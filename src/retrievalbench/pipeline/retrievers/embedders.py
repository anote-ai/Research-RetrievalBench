"""Embedder backends: OpenAI API + local HF models.

Multiple embedders are REQUIRED for Claim 2 (CSI/ESI): chunking-vs-embedding
sensitivity can only be measured when there is variance across embedding models.

All embedders expose the same interface: embed(texts) -> np.ndarray (n, dim),
rows L2-normalized when spec.normalize=True. Local models are loaded once and
cached by model id.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

from ..config import EmbedderSpec

if TYPE_CHECKING:
    pass


class Embedder:
    """Base interface."""

    def __init__(self, spec: EmbedderSpec) -> None:
        self.spec = spec
        self.tokens_used = 0  # OpenAI backend accumulates here; local stays 0

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.spec.name


# ---------------------------------------------------------------------------
# Local model cache — sentence-transformers models are expensive to load.
# Keyed by HF model id so a grid reusing the same model across cells loads once.
# ---------------------------------------------------------------------------

_LOCAL_CACHE: dict[str, object] = {}
_LOCAL_LOCK = threading.Lock()


def _get_local_model(model_id: str):
    with _LOCAL_LOCK:
        if model_id not in _LOCAL_CACHE:
            from sentence_transformers import SentenceTransformer
            _LOCAL_CACHE[model_id] = SentenceTransformer(model_id)
        return _LOCAL_CACHE[model_id]


class LocalEmbedder(Embedder):
    """HF model via sentence-transformers (BGE / E5 / GTE / Nomic)."""

    def embed(self, texts: list[str]) -> np.ndarray:
        model = _get_local_model(self.spec.model)
        # Empty strings would error some tokenizers
        safe = [t if t.strip() else " " for t in texts]
        embs = model.encode(
            safe, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False,
        )
        embs = np.asarray(embs, dtype="float32")
        if self.spec.normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs /= np.maximum(norms, 1e-9)
        return embs


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-* via API. Accumulates token usage for cost."""

    def embed(self, texts: list[str]) -> np.ndarray:
        import time as _time
        from openai import OpenAI, APITimeoutError, APIConnectionError
        import os

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=120.0)
        safe = [t if t.strip() else " " for t in texts]
        all_embs: list[list[float]] = []
        batch_size = 500
        for i in range(0, len(safe), batch_size):
            batch = safe[i: i + batch_size]
            for attempt in range(5):
                try:
                    resp = client.embeddings.create(input=batch, model=self.spec.model)
                    self.tokens_used += resp.usage.total_tokens
                    embs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
                    all_embs.extend(embs)
                    break
                except (APITimeoutError, APIConnectionError):
                    if attempt == 4:
                        raise
                    _time.sleep(2 ** attempt)
        arr = np.asarray(all_embs, dtype="float32")
        if self.spec.normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr /= np.maximum(norms, 1e-9)
        return arr


def make_embedder(spec: EmbedderSpec) -> Embedder:
    if spec.backend == "openai":
        return OpenAIEmbedder(spec)
    if spec.backend == "local":
        return LocalEmbedder(spec)
    raise ValueError(f"unknown embedder backend: {spec.backend}")


# ---------------------------------------------------------------------------
# Registry — the default embedder pool used by the full-grid experiment.
# Small variants keep cost/memory bounded across 12 domains.
# ---------------------------------------------------------------------------

EMBEDDER_REGISTRY: dict[str, EmbedderSpec] = {
    "openai-3-small": EmbedderSpec(
        name="openai-3-small", backend="openai",
        model="text-embedding-3-small", dim=1536, cost_per_1m_tokens=0.02),
    "bge-small": EmbedderSpec(
        name="bge-small", backend="local",
        model="BAAI/bge-small-en-v1.5", dim=384, cost_per_1m_tokens=0.0),
    "e5-small": EmbedderSpec(
        name="e5-small", backend="local",
        model="intfloat/e5-small-v2", dim=384, cost_per_1m_tokens=0.0),
    "gte-small": EmbedderSpec(
        name="gte-small", backend="local",
        model="thenlper/gte-small", dim=384, cost_per_1m_tokens=0.0),
    "nomic": EmbedderSpec(
        name="nomic", backend="local",
        model="nomic-ai/nomic-embed-text-v1.5", dim=768, cost_per_1m_tokens=0.0),
}
