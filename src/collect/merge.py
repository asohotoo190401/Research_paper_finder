"""Merge both sources into one deduplicated corpus.

The same paper often exists on arXiv AND Semantic Scholar. We match
records by, in order of reliability:

  1. arXiv id        (exact, both sources expose it)
  2. DOI             (exact, lowercased)
  3. normalized title (lowercase, alphanumerics only — catches
                       punctuation/casing differences)

Matched records are MERGED rather than dropped, so the corpus keeps the
best of both worlds: arXiv's category labels and full abstracts plus
Semantic Scholar's citation counts and venues. This cross-source
integration is the substance behind the "collect text data from more
than one source" requirement.
"""

from __future__ import annotations

import itertools
import re

_NORM_RE = re.compile(r"[^a-z0-9]+")

# Scalar fields filled from the other record when missing on the base.
FILL_FIELDS = ["s2_id", "arxiv_id", "doi", "venue", "url", "year"]
# List fields merged as an order-preserving union.
LIST_FIELDS = ["authors", "categories", "fields_of_study"]


def normalize_title(title: str | None) -> str:
    return _NORM_RE.sub("", (title or "").lower())


def keys_for(rec: dict) -> list[tuple[str, str]]:
    keys = []
    if rec.get("arxiv_id"):
        keys.append(("arxiv", rec["arxiv_id"]))
    if rec.get("doi"):
        keys.append(("doi", rec["doi"]))
    norm = normalize_title(rec.get("title"))
    if norm:
        keys.append(("title", norm))
    return keys


def merge_pair(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for field in FILL_FIELDS:
        if not merged.get(field) and extra.get(field):
            merged[field] = extra[field]
    for field in LIST_FIELDS:
        combined = list(merged.get(field) or [])
        for item in extra.get(field) or []:
            if item not in combined:
                combined.append(item)
        merged[field] = combined
    if len(extra.get("abstract") or "") > len(merged.get("abstract") or ""):
        merged["abstract"] = extra["abstract"]
    counts = [c for c in (merged.get("citation_count"),
                          extra.get("citation_count")) if c is not None]
    merged["citation_count"] = max(counts) if counts else None
    merged["sources"] = sorted(set((merged.get("sources") or [])
                                   + (extra.get("sources") or [])))
    return merged


def _finalize(doc: dict) -> dict:
    """Guarantee a stable schema and a stable doc_id for every document."""
    if doc.get("arxiv_id"):
        doc_id = f"arxiv:{doc['arxiv_id']}"
    elif doc.get("s2_id"):
        doc_id = f"s2:{doc['s2_id']}"
    else:
        doc_id = f"title:{normalize_title(doc.get('title'))[:40]}"
    for field in ("s2_id", "arxiv_id", "doi", "venue", "citation_count",
                  "year", "url"):
        doc.setdefault(field, None)
    for field in ("authors", "categories", "fields_of_study"):
        doc.setdefault(field, [])
    doc["doc_id"] = doc_id
    return doc


def merge_records(*record_lists: list[dict],
                  min_abstract_chars: int = 100) -> list[dict]:
    key_to_pos: dict[tuple[str, str], int] = {}
    docs: list[dict] = []
    for rec in itertools.chain(*record_lists):
        rec = dict(rec)
        if "source" in rec:
            rec["sources"] = [rec.pop("source")]
        pos = None
        for key in keys_for(rec):
            if key in key_to_pos:
                pos = key_to_pos[key]
                break
        if pos is None:
            docs.append(rec)
            pos = len(docs) - 1
        else:
            docs[pos] = merge_pair(docs[pos], rec)
        # Register all keys of the (possibly merged) doc, so future records
        # matching on ANY of its identifiers land on the same document.
        for key in keys_for(docs[pos]):
            key_to_pos.setdefault(key, pos)

    corpus = []
    for doc in docs:
        if len(doc.get("abstract") or "") < min_abstract_chars:
            continue  # too little text to index meaningfully
        corpus.append(_finalize(doc))
    return corpus
