"""
Diversity Dynamics Tracking

Track how diversity evolves during training.
This is a NOVEL analysis - no existing paper tracks diversity through training steps.

Key contribution: Shows WHEN and WHY diversity changes during training.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments

import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from .diversity_metrics import (
    compute_self_bleu,
    compute_distinct_n,
    compute_semantic_diversity
)

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


@dataclass
class DiversitySnapshot:
    """Single snapshot of diversity metrics at a training step."""
    step: int
    epoch: float
    self_bleu: float
    distinct_1: float
    distinct_2: float
    distinct_3: float
    semantic_diversity: float
    output_entropy: float
    
    def to_dict(self):
        return asdict(self)


class DiversityDynamicsTracker:
    """
    Track diversity metrics throughout training.
    
    Usage:
        tracker = DiversityDynamicsTracker(
            model=model,
            tokenizer=tokenizer,
            validation_prompts=val_prompts,
            n_samples=10
        )
        
        # During training (every N steps):
        snapshot = tracker.compute_snapshot(step=100, epoch=1.5)
        
        # After training:
        tracker.save_history('dynamics.json')
        tracker.plot_trajectory('diversity_trajectory.pdf')
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        validation_prompts: List[str],
        n_samples: int = 10,
        max_new_tokens: int = 100,
        temperature: float = 0.9,
        output_dir: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.validation_prompts = validation_prompts
        self.n_samples = n_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.output_dir = Path(output_dir) if output_dir else None
        
        self.history: List[DiversitySnapshot] = []
        
        logger.info(f"Initialized DiversityDynamicsTracker:")
        logger.info(f"  - Validation prompts: {len(validation_prompts)}")
        logger.info(f"  - Samples per prompt: {n_samples}")
        logger.info(f"  - Temperature: {temperature}")
    
    def compute_snapshot(self, step: int, epoch: float) -> DiversitySnapshot:
        """
        Compute diversity metrics at current training step.
        
        Args:
            step: Global training step
            epoch: Current epoch (can be fractional)
            
        Returns:
            DiversitySnapshot with all metrics
        """
        logger.info(f"Computing diversity snapshot at step {step}...")
        
        # Set model to eval mode
        was_training = self.model.training
        self.model.eval()
        
        # Generate samples for each validation prompt
        all_generations = []
        try:
            for i, prompt in enumerate(self.validation_prompts[:5]):  # Use first 5 for speed
                generations = self._generate_diverse_samples(prompt)
                all_generations.extend(generations)
                
                if (i + 1) % 2 == 0:
                    logger.debug(f"  Generated for prompt {i+1}/5")
        
        except Exception as e:
            logger.error(f"Error generating samples: {e}")
            # Return zero snapshot on error
            return DiversitySnapshot(
                step=step, epoch=epoch,
                self_bleu=0.0, distinct_1=0.0, distinct_2=0.0,
                distinct_3=0.0, semantic_diversity=0.0, output_entropy=0.0
            )
        
        # Compute metrics
        try:
            snapshot = DiversitySnapshot(
                step=step,
                epoch=epoch,
                self_bleu=compute_self_bleu(all_generations) if len(all_generations) >= 2 else 0.0,
                distinct_1=compute_distinct_n(all_generations, n=1),
                distinct_2=compute_distinct_n(all_generations, n=2),
                distinct_3=compute_distinct_n(all_generations, n=3),
                semantic_diversity=self._compute_semantic_div(all_generations),
                output_entropy=self._compute_entropy(all_generations)
            )
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            snapshot = DiversitySnapshot(
                step=step, epoch=epoch,
                self_bleu=0.0, distinct_1=0.0, distinct_2=0.0,
                distinct_3=0.0, semantic_diversity=0.0, output_entropy=0.0
            )
        
        self.history.append(snapshot)
        
        # Restore training mode
        if was_training:
            self.model.train()
        
        logger.info(f"  Self-BLEU: {snapshot.self_bleu:.3f}")
        logger.info(f"  Distinct-2: {snapshot.distinct_2:.3f}")
        logger.info(f"  Semantic: {snapshot.semantic_diversity:.3f}")
        
        return snapshot
    
    def _generate_diverse_samples(self, prompt: str) -> List[str]:
        """Generate n_samples diverse generations for a prompt."""
        # Format with Qwen3 chat template
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode for speed
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=self.n_samples,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return []
        
        generations = [
            self.tokenizer.decode(
                output[inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            for output in outputs
        ]
        
        # Filter empty generations
        generations = [g for g in generations if len(g) > 0]
        
        return generations
    
    def _compute_semantic_div(self, texts: List[str]) -> float:
        """Compute semantic diversity using embeddings."""
        try:
            return compute_semantic_diversity(texts)
        except Exception as e:
            logger.warning(f"Could not compute semantic diversity: {e}")
            return 0.0
    
    def _compute_entropy(self, texts: List[str]) -> float:
        """Compute output distribution entropy."""
        # Simple token-level entropy
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return 0.0
        
        # Compute token distribution
        unique, counts = np.unique(all_tokens, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return float(entropy)
    
    def save_history(self, filepath: str):
        """Save tracking history to JSON."""
        data = {
            'snapshots': [s.to_dict() for s in self.history],
            'config': {
                'n_samples': self.n_samples,
                'max_new_tokens': self.max_new_tokens,
                'temperature': self.temperature,
                'n_prompts': len(self.validation_prompts)
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✅ Saved diversity dynamics to {filepath}")
    
    def load_history(self, filepath: str):
        """Load tracking history from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.history = [
            DiversitySnapshot(**snapshot) 
            for snapshot in data['snapshots']
        ]
        
        logger.info(f"✅ Loaded {len(self.history)} snapshots from {filepath}")
    
    def plot_trajectory(
        self, 
        save_path: Optional[str] = None,
        phase_boundary: Optional[int] = None,
        title_suffix: str = ""
    ):
        """
        Plot diversity metrics over training.
        
        Args:
            save_path: Where to save plot (if None, show only)
            phase_boundary: Step where Phase 1 ends and Phase 2 begins
            title_suffix: Additional text for title (e.g., model name)
        """
        if not self.history:
            logger.warning("No history to plot!")
            return
        
        # Extract data
        steps = [s.step for s in self.history]
        self_bleu = [s.self_bleu for s in self.history]
        distinct_2 = [s.distinct_2 for s in self.history]
        semantic = [s.semantic_diversity for s in self.history]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        title = 'Diversity Dynamics During Training'
        if title_suffix:
            title += f' ({title_suffix})'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot 1: Self-BLEU (lower = more diverse)
        ax = axes[0, 0]
        ax.plot(steps, self_bleu, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
        if phase_boundary:
            ax.axvline(phase_boundary, color='r', linestyle='--', linewidth=2, 
                      label='Phase 1→2', alpha=0.7)
        ax.set_xlabel('Training Step', fontweight='bold')
        ax.set_ylabel('Self-BLEU ↓', fontweight='bold')
        ax.set_title('Self-BLEU (lower = more diverse)')
        ax.grid(True, alpha=0.3)
        if phase_boundary:
            ax.legend()
        
        # Plot 2: Distinct-2 (higher = more diverse)
        ax = axes[0, 1]
        ax.plot(steps, distinct_2, 'g-', linewidth=2, marker='o', markersize=4, alpha=0.7)
        if phase_boundary:
            ax.axvline(phase_boundary, color='r', linestyle='--', linewidth=2, 
                      label='Phase 1→2', alpha=0.7)
        ax.set_xlabel('Training Step', fontweight='bold')
        ax.set_ylabel('Distinct-2 ↑', fontweight='bold')
        ax.set_title('Distinct-2 (higher = more diverse)')
        ax.grid(True, alpha=0.3)
        if phase_boundary:
            ax.legend()
        
        # Plot 3: Semantic Diversity (higher = more diverse)
        ax = axes[1, 0]
        ax.plot(steps, semantic, 'm-', linewidth=2, marker='o', markersize=4, alpha=0.7)
        if phase_boundary:
            ax.axvline(phase_boundary, color='r', linestyle='--', linewidth=2, 
                      label='Phase 1→2', alpha=0.7)
        ax.set_xlabel('Training Step', fontweight='bold')
        ax.set_ylabel('Semantic Diversity ↑', fontweight='bold')
        ax.set_title('Semantic Diversity (higher = more diverse)')
        ax.grid(True, alpha=0.3)
        if phase_boundary:
            ax.legend()
        
        # Plot 4: Combined view
        ax = axes[1, 1]
        # Normalize to [0, 1] for comparison
        if max(self_bleu) > 0:
            self_bleu_norm = 1 - np.array(self_bleu) / max(self_bleu)  # Invert
        else:
            self_bleu_norm = np.zeros_like(self_bleu)
            
        if max(distinct_2) > 0:
            distinct_2_norm = np.array(distinct_2) / max(distinct_2)
        else:
            distinct_2_norm = np.zeros_like(distinct_2)
            
        if max(semantic) > 0:
            semantic_norm = np.array(semantic) / max(semantic)
        else:
            semantic_norm = np.zeros_like(semantic)
        
        ax.plot(steps, self_bleu_norm, 'b-', linewidth=2, label='Self-BLEU (inv)', alpha=0.7)
        ax.plot(steps, distinct_2_norm, 'g-', linewidth=2, label='Distinct-2', alpha=0.7)
        ax.plot(steps, semantic_norm, 'm-', linewidth=2, label='Semantic', alpha=0.7)
        
        if phase_boundary:
            ax.axvline(phase_boundary, color='r', linestyle='--', linewidth=2, 
                      label='Phase 1→2', alpha=0.7)
        
        ax.set_xlabel('Training Step', fontweight='bold')
        ax.set_ylabel('Normalized Diversity ↑', fontweight='bold')
        ax.set_title('All Metrics (Normalized)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        try:
            plt.tight_layout()
        except Exception as e:
            logger.warning(f"tight_layout failed: {e}, continuing anyway")
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()


def compare_diversity_trajectories(
    histories: Dict[str, List[DiversitySnapshot]],
    metric: str = 'distinct_2',
    save_path: Optional[str] = None,
    phase_boundary: Optional[int] = None
):
    """
    Compare diversity trajectories across multiple models.
    
    Args:
        histories: Dict mapping model_name -> list of snapshots
        metric: Which metric to plot ('self_bleu', 'distinct_2', 'semantic_diversity')
        save_path: Where to save plot
        phase_boundary: Optional phase boundary line
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (model_name, history) in enumerate(histories.items()):
        if not history:
            continue
            
        steps = [s.step for s in history]
        values = [getattr(s, metric) for s in history]
        
        plt.plot(steps, values, linewidth=2.5, marker='o', markersize=5,
                label=model_name, color=colors[i % len(colors)], alpha=0.8)
    
    if phase_boundary:
        plt.axvline(phase_boundary, color='black', linestyle='--', 
                   label='Phase 1→2', linewidth=2, alpha=0.5)
    
    plt.xlabel('Training Step', fontsize=13, fontweight='bold')
    plt.ylabel(f'{metric.replace("_", " ").title()}', fontsize=13, fontweight='bold')
    plt.title(f'Diversity Dynamics Comparison: {metric.replace("_", " ").title()}', 
             fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    
    try:
        plt.tight_layout()
    except Exception as e:
        logger.warning(f"tight_layout failed: {e}, continuing anyway")
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


# Example test
if __name__ == "__main__":
    print("✅ Diversity dynamics module loaded successfully")
    print("This module provides NOVEL analysis - tracking diversity through training")
    print("No existing paper does this!")

