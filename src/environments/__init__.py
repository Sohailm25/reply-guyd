"""
Environment abstractions for training and evaluation.

Inspired by Prime Intellect's verifiers library, providing clean separation
between environment logic (rewards, generation, evaluation) and training code.
"""

from .base import BaseEnvironment
from .twitter_reply import TwitterReplyEnvironment

__all__ = [
    'BaseEnvironment',
    'TwitterReplyEnvironment',
]

