"""Source 1: the arXiv API.

arXiv serves search results as an Atom XML feed (a feed-style API, unlike
Semantic Scholar's JSON REST API — together they satisfy the "more than
one source" requirement with two genuinely different API paradigms).

API docs: https://info.arxiv.org/help/api/
Etiquette: arXiv asks clients to wait ~3 seconds between requests, so we
page politely with a fixed delay.
"""

from __future__ import annotations

import re
import time

import feedparser
import requests

from src.utils import clean_whitespace

API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 100
DELAY_SECONDS = 3.0

# arXiv ids carry a version suffix (2401.00001v2). We strip it so the same
# paper matches across sources regardless of version. Anchored at the end,
# so old-style ids like "math.CV/0601001v3" keep their category prefix.
_VERSION_RE = re.compile(r"v\d+$")


def strip_version(arxiv_id: str) -> str:
    return _VERSION_RE.sub("", arxiv_id)


def parse_feed(xml_text: str) -> list[dict]:
    """Parse one Atom page into normalized paper records."""
    feed = feedparser.parse(xml_text)
    records = []
    for entry in feed.entries:
        raw_id = entry.get("id", "")
        if "/abs/" not in raw_id:
            continue
        arxiv_id = strip_version(raw_id.rsplit("/abs/", 1)[-1])
        title = clean_whitespace(entry.get("title", ""))
        abstract = clean_whitespace(entry.get("summary", ""))
        if not title or not abstract:
            continue
        published = entry.get("published", "")
        year = int(published[:4]) if published[:4].isdigit() else None
        records.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": [a.get("name", "") for a in entry.get("authors", [])],
                "year": year,
                "categories": [t.get("term", "") for t in entry.get("tags", [])],
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "source": "arxiv",
            }
        )
    return records


def _fetch_page(session: requests.Session, params: dict) -> str:
    for attempt in range(3):
        try:
            resp = session.get(API_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            print(f"  arXiv request failed ({exc}); retry {attempt + 1}/3")
            time.sleep(DELAY_SECONDS * (attempt + 1))
    raise RuntimeError("arXiv API unreachable after 3 attempts")


def fetch_category(category: str, max_results: int = 600,
                   session: requests.Session | None = None) -> list[dict]:
    """Page through the newest `max_results` papers of one arXiv category."""
    session = session or requests.Session()
    records: list[dict] = []
    for start in range(0, max_results, PAGE_SIZE):
        params = {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": min(PAGE_SIZE, max_results - start),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        batch = parse_feed(_fetch_page(session, params))
        if not batch:
            # arXiv occasionally returns a transient empty page; retry once.
            time.sleep(DELAY_SECONDS)
            batch = parse_feed(_fetch_page(session, params))
            if not batch:
                break
        records.extend(batch)
        print(f"  {category}: {len(records)} papers", flush=True)
        time.sleep(DELAY_SECONDS)
    return records


def collect(categories: list[str], per_category: int) -> list[dict]:
    session = requests.Session()
    session.headers["User-Agent"] = "PaperFinder (student text-mining project)"
    seen: set[str] = set()
    records: list[dict] = []
    for cat in categories:
        print(f"Collecting arXiv category {cat} ...", flush=True)
        for rec in fetch_category(cat, per_category, session):
            if rec["arxiv_id"] in seen:
                continue  # cross-listed papers appear in several categories
            seen.add(rec["arxiv_id"])
            records.append(rec)
    return records
