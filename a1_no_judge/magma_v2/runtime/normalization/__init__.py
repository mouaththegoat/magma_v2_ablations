"""Canonical normalization layer for MAGMA v2.

Raw user layouts are converted into small internal representations before
modeling. A CSV with image paths can therefore normalize to an image manifest
instead of being treated as generic tabular features.
"""

from magma_v2.runtime.normalization.normalize import normalize_inputs

__all__ = ["normalize_inputs"]

