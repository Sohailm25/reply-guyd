"""
Deprecated shim forwarding to src.training.trainers.polychromic.

Maintained to preserve backwards compatibility for legacy imports.
"""

import warnings

from src.training.trainers.polychromic import PolychromicTrainer, PolychromicConfig  # noqa: F401

warnings.warn(
    "src.training.polychromic_trainer is deprecated; "
    "import from src.training.trainers.polychromic instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PolychromicTrainer", "PolychromicConfig"]
