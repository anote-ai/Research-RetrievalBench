"""Chunking strategies: generic + structure-aware."""
from .generic import build_chunk_corpus, GENERIC_CHUNKERS
from .structure_aware import STRUCTURE_AWARE_CHUNKERS, build_structure_aware

__all__ = [
    "build_chunk_corpus",
    "GENERIC_CHUNKERS",
    "STRUCTURE_AWARE_CHUNKERS",
    "build_structure_aware",
]
