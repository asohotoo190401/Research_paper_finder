"""Source 2: the Semantic Scholar Graph API (JSON REST).

Uses the bulk search endpoint, which returns up to 1,000 papers per page
plus a continuation token. Without an API key you share a global rate
pool with everyone else, so HTTP 429 responses are normal — we back off
and retry. Setting the env var S2_API_KEY (free key) makes collection
faster and steadier, but everything works without one.

API docs: https://api.semanticscholar.org/api-docs/graph
"""

from __future__ import annotations

import os
import time

import requests

from src.utils import clean_whitespace

BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = ("paperId,title,abstract,year,authors,externalIds,url,"
          "citationCount,fieldsOfStudy,venue")
BACKOFFS = [5, 15, 30, 60, 120]
MIN_ABSTRACT_CHARS = 100  # S2 lacks abstracts for many papers; skip those


def _get(params: dict) -> dict:
    headers = {}
    key = os.environ.get("S2_API_KEY")
    if key:
        headers["x-api-key"] = key
    for wait in [0] + BACKOFFS:
        if wait:
            print(f"  rate limited, waiting {wait}s ...", flush=True)
            time.sleep(wait)
        try:
            resp = requests.get(BULK_URL, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            print(f"  request error ({exc}), retrying ...", flush=True)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            continue
        resp.raise_for_status()
    raise RuntimeError(
        "Semantic Scholar kept rate-limiting; try again later or set S2_API_KEY."
    )


def parse_record(raw: dict) -> dict | None:
    """Normalize one API record; None if unusable (no title/abstract)."""
    title = clean_whitespace(raw.get("title") or "")
    abstract = clean_whitespace(raw.get("abstract") or "")
    if not title or len(abstract) < MIN_ABSTRACT_CHARS:
        return None
    ext = raw.get("externalIds") or {}
    doi = (ext.get("DOI") or "").lower() or None
    return {
        "s2_id": raw.get("paperId"),
        "arxiv_id": ext.get("ArXiv"),
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "authors": [a.get("name", "") for a in raw.get("authors") or []],
        "year": raw.get("year"),
        "venue": raw.get("venue") or None,
        "fields_of_study": raw.get("fieldsOfStudy") or [],
        "citation_count": raw.get("citationCount"),
        "url": raw.get("url"),
        "source": "semantic_scholar",
    }


def fetch_query(query: str, year_from: int = 2019,
                max_records: int = 800) -> list[dict]:
    """Page the bulk endpoint for one query until max_records usable papers."""
    params: dict = {"query": query, "fields": FIELDS, "year": f"{year_from}-"}
    out: list[dict] = []
    token = None
    while len(out) < max_records:
        if token:
            params["token"] = token
        payload = _get(dict(params))
        for raw in payload.get("data") or []:
            rec = parse_record(raw)
            if rec:
                out.append(rec)
            if len(out) >= max_records:
                break
        print(f"  '{query}': {len(out)} usable papers", flush=True)
        token = payload.get("token")
        if not token:
            break
        time.sleep(1.0)
    return out


def collect(queries: list[str], per_query: int, year_from: int = 2019) -> list[dict]:
    seen: set[str] = set()
    records: list[dict] = []
    for q in queries:
        print(f"Collecting Semantic Scholar query '{q}' ...", flush=True)
        for rec in fetch_query(q, year_from=year_from, max_records=per_query):
            if rec["s2_id"] in seen:
                continue  # the same paper can match several queries
            seen.add(rec["s2_id"])
            records.append(rec)
        time.sleep(1.0)
    return records
