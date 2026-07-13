"""Structure-aware chunkers — the mechanism behind Claim 3.

Claim 3: domain-specific structure-aware chunking captures signal that generic
fixed-size chunking destroys (legal clause boundaries, financial tables,
medical section structure). Each domain gets a chunker that respects its
document structure instead of naively splitting on token count.

These are deliberately rule-based (regex + lightweight parsing) rather than
ML-based, so they are deterministic and cheap — the point is to isolate the
effect of *structural boundary preservation*, not model capacity.
"""
from __future__ import annotations

import re

from ..config import ChunkerSpec


# ---------------------------------------------------------------------------
# Legal: split on clause/section boundaries (numbered headings, ALL-CAPS terms)
# ---------------------------------------------------------------------------

# Matches common legal heading patterns:
#   "Section 3." / "3.1" / "ARTICLE II" / "WHEREAS:" / "(a)" enumeration
_LEGAL_HEADING = re.compile(
    r"(?:^|\n)\s*("
    r"(?:Section|Article|Clause|Paragraph|Recital)\s+[0-9IVXLCivxlc]+\.?"
    r"|[0-9]+\.[0-9]*(?:\.[0-9]+)*\.?"
    r"|\([a-zA-Z0-9]+\)"
    r"|(?:WHEREAS|NOW THEREFORE|IN WITNESS WHEREOF)\s*:?"
    r")",
    re.MULTILINE,
)


def chunk_legal_clause(doc: dict, max_tokens: int = 512) -> list[dict]:
    """Split on legal headings; merge fragments under ~50 tokens."""
    text = doc["text"]
    # Insert split markers at heading positions
    marked = _LEGAL_HEADING.sub(r"\n@@@\1", text)
    parts = [p.strip() for p in marked.split("\n@@@") if p.strip()]
    if not parts:
        parts = [text]

    chunks: list[dict] = []
    buf = ""
    for p in parts:
        if len(buf.split()) + len(p.split()) <= max_tokens and buf:
            buf += "\n" + p
        else:
            if buf:
                _emit(chunks, doc, buf)
            buf = p
    if buf:
        _emit(chunks, doc, buf)
    return chunks or [{"chunk_id": f"{doc['doc_id']}_law0", "doc_id": doc["doc_id"], "text": text}]


# ---------------------------------------------------------------------------
# Financial: keep markdown tables intact; split around them
# ---------------------------------------------------------------------------

# A markdown table row: | ... | ... |
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def chunk_financial_table(doc: dict, max_tokens: int = 512) -> list[dict]:
    """Keep tables as atomic chunks; chunk the prose around them normally."""
    lines = doc["text"].split("\n")
    chunks: list[dict] = []
    prose_buf: list[str] = []
    table_buf: list[str] = []
    idx = 0

    def flush_prose():
        nonlocal idx, prose_buf
        if not prose_buf:
            return
        words = " ".join(prose_buf).split()
        for i in range(0, len(words), max_tokens):
            _emit_id(chunks, doc, f"fin{idx}", " ".join(words[i:i + max_tokens]))
            idx += 1
        prose_buf = []

    def flush_table():
        nonlocal idx, table_buf
        if table_buf:
            _emit_id(chunks, doc, f"tab{idx}", "\n".join(table_buf))
            idx += 1
            table_buf = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if _MD_TABLE_ROW.match(line) or _MD_TABLE_SEP.match(line):
            flush_prose()
            table_buf.append(line)
        else:
            if table_buf and not _MD_TABLE_ROW.match(line) and not _MD_TABLE_SEP.match(line):
                flush_table()
            prose_buf.append(line)
        i += 1
    flush_prose()
    flush_table()
    return chunks or [{"chunk_id": f"{doc['doc_id']}_fin0", "doc_id": doc["doc_id"], "text": doc["text"]}]


# ---------------------------------------------------------------------------
# Medical: split on IMRAD / section headers
# ---------------------------------------------------------------------------

_MEDICAL_HEADING = re.compile(
    r"(?:^|\n)\s*("
    r"Abstract|Introduction|Background|Methods?|Materials and Methods|"
    r"Results?|Discussion|Conclusion[s]?|References?|"
    r"Patient(?:\s+Presentation)?|Diagnosis|Treatment|Prognosis|"
    r"Clinical\s+\w+|Case\s+\w+"
    r")\s*[:\n]",
    re.IGNORECASE | re.MULTILINE,
)


def chunk_medical_section(doc: dict, max_tokens: int = 512) -> list[dict]:
    marked = _MEDICAL_HEADING.sub(r"\n@@@\1", doc["text"])
    parts = [p.strip() for p in marked.split("\n@@@") if p.strip()] or [doc["text"]]
    chunks: list[dict] = []
    buf = ""
    for p in parts:
        if len(buf.split()) + len(p.split()) <= max_tokens and buf:
            buf += "\n" + p
        else:
            if buf:
                _emit(chunks, doc, buf)
            buf = p
    if buf:
        _emit(chunks, doc, buf)
    return chunks or [{"chunk_id": f"{doc['doc_id']}_med0", "doc_id": doc["doc_id"], "text": doc["text"]}]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _emit(chunks: list[dict], doc: dict, text: str) -> None:
    chunks.append({"chunk_id": f"{doc['doc_id']}_x{len(chunks)}",
                   "doc_id": doc["doc_id"], "text": text})


def _emit_id(chunks: list[dict], doc: dict, tag: str, text: str) -> None:
    chunks.append({"chunk_id": f"{doc['doc_id']}_{tag}",
                   "doc_id": doc["doc_id"], "text": text})


_BUILDERS = {
    "legal_clause": chunk_legal_clause,
    "financial_table": chunk_financial_table,
    "medical_section": chunk_medical_section,
}


def build_structure_aware(corpus: list[dict], spec: ChunkerSpec) -> list[dict]:
    fn = _BUILDERS.get(spec.name)
    if fn is None:
        raise ValueError(f"unknown structure-aware chunker: {spec.name}")
    chunks: list[dict] = []
    for doc in corpus:
        chunks.extend(fn(doc, **spec.params))
    return chunks


# Registry — map each structure-aware chunker to the domain it targets.
# run.py adds the matching structure-aware chunker to the grid for that domain.
STRUCTURE_AWARE_CHUNKERS: dict[str, ChunkerSpec] = {
    "legal": ChunkerSpec(name="legal_clause", kind="structure_aware"),
    "finance": ChunkerSpec(name="financial_table", kind="structure_aware"),
    "medical": ChunkerSpec(name="medical_section", kind="structure_aware"),
}
