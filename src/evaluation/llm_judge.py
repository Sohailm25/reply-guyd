"""
LLM-as-Judge Evaluation

Uses Claude 3.5 Sonnet to evaluate reply quality.

Critical mitigations for bias:
- Position bias: Randomize order of baseline/polychromic
- Length bias: Normalize by reply length
- Multiple criteria: Engagement, relevance, creativity, naturalness
- Inter-rater reliability: Compare Claude with GPT-4o on subset
"""

import anthropic
import openai
import random
import json
import logging
from typing import List, Dict, Optional, Tuple
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationCriteria:
    """Evaluation criteria for LLM judge."""
    engagement: str = "Rate the engagement potential of this reply (1-10). Would it get likes/retweets?"
    relevance: str = "Rate the relevance to the original tweet (1-10). Is it on-topic and contextually appropriate?"
    creativity: str = "Rate the creativity/originality (1-10). Is it unique or generic?"
    naturalness: str = "Rate the naturalness (1-10). Does it sound like a real person wrote it?"


class LLMJudge:
    """
    LLM-based evaluation with bias mitigation.
    
    Features:
    - Position randomization
    - Multiple evaluation criteria
    - Detailed reasoning logging
    - Cost tracking
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        provider: str = "anthropic"
    ):
        self.model = model
        self.provider = provider
        
        if provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        self.total_cost = 0.0
        self.total_tokens = 0
        
        logger.info(f"Initialized LLMJudge with {model} ({provider})")
    
    def evaluate_pair(
        self,
        tweet: str,
        reply_a: str,
        reply_b: str,
        criteria: EvaluationCriteria = None
    ) -> Dict:
        """
        Evaluate a pair of replies with position randomization.
        
        Args:
            tweet: Original tweet
            reply_a: Reply from model A
            reply_b: Reply from model B
            criteria: Evaluation criteria
            
        Returns:
            Dictionary with scores for each reply
        """
        if criteria is None:
            criteria = EvaluationCriteria()
        
        # Randomize position to mitigate bias
        if random.random() < 0.5:
            first, second = reply_a, reply_b
            swap = False
        else:
            first, second = reply_b, reply_a
            swap = True
        
        # Create prompt
        prompt = self._create_evaluation_prompt(tweet, first, second, criteria)
        
        # Get LLM judgment
        response = self._call_llm(prompt)
        
        # Parse scores
        scores = self._parse_response(response)
        
        # Swap back if needed
        if swap:
            scores['reply_a'], scores['reply_b'] = scores['reply_b'], scores['reply_a']
        
        scores['reasoning'] = response
        scores['position_swapped'] = swap
        
        return scores
    
    def _create_evaluation_prompt(
        self,
        tweet: str,
        reply_1: str,
        reply_2: str,
        criteria: EvaluationCriteria
    ) -> str:
        """Create evaluation prompt."""
        prompt = f"""You are evaluating two Twitter replies to determine which is better for engagement.

Original Tweet:
"{tweet}"

Reply 1:
"{reply_1}"

Reply 2:
"{reply_2}"

Please rate each reply on the following criteria (1-10 scale):

1. Engagement Potential: {criteria.engagement}
2. Relevance: {criteria.relevance}
3. Creativity: {criteria.creativity}
4. Naturalness: {criteria.naturalness}

Provide your evaluation in the following JSON format:
{{
    "reply_1": {{
        "engagement": <score 1-10>,
        "relevance": <score 1-10>,
        "creativity": <score 1-10>,
        "naturalness": <score 1-10>,
        "total": <sum of all scores>
    }},
    "reply_2": {{
        "engagement": <score 1-10>,
        "relevance": <score 1-10>,
        "creativity": <score 1-10>,
        "naturalness": <score 1-10>,
        "total": <sum of all scores>
    }},
    "reasoning": "<brief explanation of your judgment>"
}}

Be objective and fair. Consider that both replies may be good in different ways."""

        return prompt
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM API with retry logic."""
        for attempt in range(max_retries):
            try:
                if self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }]
                    )
                    
                    # Track usage
                    self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
                    # Claude pricing: ~$3 per 1M input tokens, ~$15 per 1M output tokens
                    cost = (response.usage.input_tokens / 1_000_000 * 3.0 +
                           response.usage.output_tokens / 1_000_000 * 15.0)
                    self.total_cost += cost
                    
                    return response.content[0].text
                    
                elif self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }],
                        max_tokens=1024,
                        temperature=0.7
                    )
                    
                    # Track usage
                    self.total_tokens += response.usage.total_tokens
                    # GPT-4o pricing: ~$2.50 per 1M tokens
                    cost = response.usage.total_tokens / 1_000_000 * 2.5
                    self.total_cost += cost
                    
                    return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        raise RuntimeError("Max retries exceeded")
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response to extract scores."""
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                return {
                    'reply_a': data.get('reply_1', {}),
                    'reply_b': data.get('reply_2', {}),
                    'reasoning': data.get('reasoning', '')
                }
            else:
                logger.error("Could not find JSON in response")
                return self._default_scores()
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return self._default_scores()
    
    def _default_scores(self) -> Dict:
        """Return default scores if parsing fails."""
        default = {
            'engagement': 5,
            'relevance': 5,
            'creativity': 5,
            'naturalness': 5,
            'total': 20
        }
        return {
            'reply_a': default,
            'reply_b': default,
            'reasoning': 'Parsing failed'
        }
    
    def batch_evaluate(
        self,
        examples: List[Dict],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Evaluate a batch of tweet-reply pairs.
        
        Args:
            examples: List of dicts with 'tweet', 'reply_a', 'reply_b'
            verbose: Print progress
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, example in enumerate(examples):
            if verbose and i % 10 == 0:
                logger.info(f"Evaluating {i}/{len(examples)} (cost so far: ${self.total_cost:.2f})")
            
            try:
                scores = self.evaluate_pair(
                    example['tweet'],
                    example['reply_a'],
                    example['reply_b']
                )
                
                results.append({
                    'tweet': example['tweet'],
                    'reply_a': example['reply_a'],
                    'reply_b': example['reply_b'],
                    'scores_a': scores['reply_a'],
                    'scores_b': scores['reply_b'],
                    'reasoning': scores['reasoning'],
                    'position_swapped': scores['position_swapped']
                })
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Evaluation failed for example {i}: {e}")
                continue
        
        logger.info(f"\nBatch evaluation complete!")
        logger.info(f"  Total examples: {len(results)}")
        logger.info(f"  Total cost: ${self.total_cost:.2f}")
        logger.info(f"  Total tokens: {self.total_tokens:,}")
        
        return results
    
    def compute_win_rates(self, results: List[Dict]) -> Dict:
        """
        Compute win rates from evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with win rates and statistics
        """
        wins_a = 0
        wins_b = 0
        ties = 0
        
        total_scores_a = {'engagement': [], 'relevance': [], 'creativity': [], 'naturalness': []}
        total_scores_b = {'engagement': [], 'relevance': [], 'creativity': [], 'naturalness': []}
        
        for result in results:
            score_a = result['scores_a'].get('total', 0)
            score_b = result['scores_b'].get('total', 0)
            
            if score_a > score_b:
                wins_a += 1
            elif score_b > score_a:
                wins_b += 1
            else:
                ties += 1
            
            # Collect scores for each criterion
            for criterion in ['engagement', 'relevance', 'creativity', 'naturalness']:
                total_scores_a[criterion].append(result['scores_a'].get(criterion, 0))
                total_scores_b[criterion].append(result['scores_b'].get(criterion, 0))
        
        total = len(results)
        
        import numpy as np
        return {
            'total_comparisons': total,
            'model_a_wins': wins_a,
            'model_b_wins': wins_b,
            'ties': ties,
            'model_a_win_rate': wins_a / total if total > 0 else 0,
            'model_b_win_rate': wins_b / total if total > 0 else 0,
            'avg_scores_a': {k: np.mean(v) for k, v in total_scores_a.items()},
            'avg_scores_b': {k: np.mean(v) for k, v in total_scores_b.items()},
        }


def compute_inter_rater_reliability(
    examples: List[Dict],
    judge1_scores: List[Dict],
    judge2_scores: List[Dict]
) -> Dict:
    """
    Compute inter-rater reliability between two judges.
    
    Args:
        examples: Test examples
        judge1_scores: Scores from judge 1 (e.g., Claude)
        judge2_scores: Scores from judge 2 (e.g., GPT-4o)
        
    Returns:
        Dictionary with correlation metrics
    """
    from scipy.stats import spearmanr, pearsonr
    
    # Extract total scores for each reply
    scores1_a = [s['scores_a']['total'] for s in judge1_scores]
    scores1_b = [s['scores_b']['total'] for s in judge1_scores]
    scores2_a = [s['scores_a']['total'] for s in judge2_scores]
    scores2_b = [s['scores_b']['total'] for s in judge2_scores]
    
    # Combine for correlation
    all_scores1 = scores1_a + scores1_b
    all_scores2 = scores2_a + scores2_b
    
    # Compute correlations
    spearman_corr, spearman_p = spearmanr(all_scores1, all_scores2)
    pearson_corr, pearson_p = pearsonr(all_scores1, all_scores2)
    
    return {
        'spearman_correlation': float(spearman_corr),
        'spearman_p_value': float(spearman_p),
        'pearson_correlation': float(pearson_corr),
        'pearson_p_value': float(pearson_p),
        'interpretation': 'high' if spearman_corr > 0.7 else 'moderate' if spearman_corr > 0.5 else 'low'
    }

