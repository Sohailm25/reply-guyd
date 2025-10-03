"""
Diversity Dynamics Callback for Training

Integrates diversity tracking into the training loop without breaking existing code.
This is an OPTIONAL callback - training works fine without it.
"""

import logging
from typing import List, Optional
from transformers import TrainerCallback
from pathlib import Path

logger = logging.getLogger(__name__)


class DiversityDynamicsCallback(TrainerCallback):
    """
    Callback to track diversity dynamics during training.
    
    This is OPTIONAL - add to enable diversity tracking:
        callbacks = [DiversityDynamicsCallback(...)]
        trainer = Trainer(..., callbacks=callbacks)
    
    Without this callback, training proceeds normally.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        validation_prompts: List[str],
        tracking_frequency: int = 100,
        output_dir: str = "./output",
        n_samples: int = 10,
        max_new_tokens: int = 100,
        temperature: float = 0.9
    ):
        """
        Initialize diversity tracking callback.
        
        Args:
            model: The model being trained
            tokenizer: Tokenizer for the model
            validation_prompts: List of prompts to use for diversity tracking
            tracking_frequency: How often to compute diversity (every N steps)
            output_dir: Where to save results
            n_samples: Number of samples to generate per prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        """
        super().__init__()
        
        # Import here to avoid circular imports
        from src.evaluation.diversity_dynamics import DiversityDynamicsTracker
        
        self.tracker = DiversityDynamicsTracker(
            model=model,
            tokenizer=tokenizer,
            validation_prompts=validation_prompts,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_dir=output_dir
        )
        self.tracking_frequency = tracking_frequency
        self.output_dir = Path(output_dir)
        
        logger.info("="*60)
        logger.info("📊 Diversity Dynamics Tracking ENABLED")
        logger.info("="*60)
        logger.info(f"  Tracking frequency: Every {tracking_frequency} steps")
        logger.info(f"  Validation prompts: {len(validation_prompts)}")
        logger.info(f"  Samples per prompt: {n_samples}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info("="*60)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if state.global_step % self.tracking_frequency == 0 and state.global_step > 0:
            logger.info("\n" + "="*60)
            logger.info(f"📊 Computing diversity snapshot at step {state.global_step}...")
            logger.info("="*60)
            
            try:
                epoch = state.epoch if state.epoch is not None else 0.0
                snapshot = self.tracker.compute_snapshot(
                    step=state.global_step,
                    epoch=epoch
                )
                
                logger.info(f"✅ Diversity snapshot complete:")
                logger.info(f"   Self-BLEU: {snapshot.self_bleu:.3f} (lower = more diverse)")
                logger.info(f"   Distinct-2: {snapshot.distinct_2:.3f} (higher = more diverse)")
                logger.info(f"   Semantic: {snapshot.semantic_diversity:.3f} (higher = more diverse)")
                logger.info("="*60 + "\n")
                
            except Exception as e:
                logger.error(f"❌ Error computing diversity snapshot: {e}")
                logger.error("Continuing training...")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Save history when training completes."""
        logger.info("\n" + "="*60)
        logger.info("📊 Training complete! Saving diversity dynamics...")
        logger.info("="*60)
        
        try:
            # Save history JSON
            save_path = self.output_dir / "diversity_dynamics.json"
            self.tracker.save_history(str(save_path))
            
            # Generate plot
            plot_path = self.output_dir / "diversity_trajectory.pdf"
            self.tracker.plot_trajectory(save_path=str(plot_path))
            
            logger.info(f"✅ Diversity dynamics saved:")
            logger.info(f"   Data: {save_path}")
            logger.info(f"   Plot: {plot_path}")
            logger.info(f"   Total snapshots: {len(self.tracker.history)}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"❌ Error saving diversity dynamics: {e}")


# Test
if __name__ == "__main__":
    print("✅ Diversity callback module loaded successfully")
    print("This callback is OPTIONAL - training works without it")
    print("Add to trainer.callbacks to enable diversity tracking")

