"""Generic chunking strategies, migrated from scripts/run_chunking_pipeline.py.

These are domain-agnostic. Structure-aware chunkers (legal_clause etc.) live
in structure_aware.py and are the mechanism behind Claim 3.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from ..config import ChunkerSpec


def chunk_fixed(doc: dict, max_tokens: int = 512, overlap: int = 50) -> list[dict]:
    words = doc["text"].split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, max(1, len(words)), step):
        cw = words[i: i + max_tokens]
        if cw:
            chunks.append({"chunk_id": f"{doc['doc_id']}_c{i}",
                           "doc_id": doc["doc_id"], "text": " ".join(cw)})
    return chunks


def chunk_sentence(doc: dict, window: int = 3) -> list[dict]:
    import nltk
    sents = [s.strip() for s in nltk.sent_tokenize(doc["text"]) if s.strip()]
    chunks = []
    for i in range(0, max(1, len(sents)), window):
        chunks.append({"chunk_id": f"{doc['doc_id']}_s{i}",
                       "doc_id": doc["doc_id"], "text": " ".join(sents[i: i + window])})
    return chunks


def chunk_recursive(doc: dict, max_tokens: int = 512) -> list[dict]:
    import nltk
    paras = [p.strip() for p in re.split(r"\n{2,}", doc["text"]) if p.strip()] or [doc["text"]]
    chunks = []
    idx = 0
    for para in paras:
        words = para.split()
        if len(words) <= max_tokens:
            chunks.append({"chunk_id": f"{doc['doc_id']}_r{idx}",
                           "doc_id": doc["doc_id"], "text": para})
            idx += 1
        else:
            for sent in nltk.sent_tokenize(para):
                chunks.append({"chunk_id": f"{doc['doc_id']}_r{idx}",
                               "doc_id": doc["doc_id"], "text": sent.strip()})
                idx += 1
    return chunks


def chunk_semantic(doc: dict, embedder, threshold: float = 0.75) -> list[dict]:
    """Embed each sentence and merge adjacent ones above `threshold` cosine."""
    import nltk
    import numpy as np
    sents = [s.strip() for s in nltk.sent_tokenize(doc["text"]) if s.strip()] or [doc["text"]]
    if len(sents) == 1:
        return [{"chunk_id": f"{doc['doc_id']}_sem0", "doc_id": doc["doc_id"], "text": sents[0]}]
    embs = embedder.embed(sents)
    chunks = []
    groups, cur = [[sents[0]]], 0
    for j in range(1, len(sents)):
        sim = float(np.dot(embs[j - 1], embs[j]))
        if sim >= threshold:
            groups[cur].append(sents[j])
        else:
            groups.append([sents[j]])
            cur += 1
    for k, g in enumerate(groups):
        chunks.append({"chunk_id": f"{doc['doc_id']}_sem{k}",
                       "doc_id": doc["doc_id"], "text": " ".join(g)})
    return chunks


# spec.name -> builder (no-embedder strategies)
_GENERIC_BUILDERS: dict[str, Callable[..., list[dict]]] = {
    "fixed_512": chunk_fixed,
    "sentence": chunk_sentence,
    "recursive": chunk_recursive,
    # "semantic" handled separately in build_chunk_corpus (needs embedder)
}


def build_chunk_corpus(corpus: list[dict], spec: ChunkerSpec, embedder=None) -> list[dict]:
    """Dispatch on spec.name. Semantic chunking requires an embedder."""
    name = spec.name
    if name in _GENERIC_BUILDERS:
        fn = _GENERIC_BUILDERS[name]
        chunks = []
        for doc in corpus:
            chunks.extend(fn(doc, **spec.params))
        return chunks
    if name == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking requires an embedder")
        chunks = []
        for doc in corpus:
            chunks.extend(chunk_semantic(doc, embedder, **spec.params))
        return chunks
    # structure-aware strategies live in structure_aware.py; dispatch there
    from .structure_aware import build_structure_aware
    return build_structure_aware(corpus, spec)


# Registry of generic chunker specs for the grid.
GENERIC_CHUNKERS: list[ChunkerSpec] = [
    ChunkerSpec(name="fixed_512", kind="generic"),
    ChunkerSpec(name="sentence", kind="generic"),
    ChunkerSpec(name="recursive", kind="generic"),
    # semantic added per-experiment (needs embedder) in run.py
]
