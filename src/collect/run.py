"""Collect from both sources and build data/corpus.jsonl.

Usage (from the repo root):
    python -m src.collect.run
    python -m src.collect.run --arxiv-per-category 400 --s2-per-query 600
    python -m src.collect.run --skip-arxiv        # reuse data/raw/arxiv.jsonl

Takes roughly 10 minutes with default settings (most of it is the polite
3-second delay between arXiv requests).
"""

from __future__ import annotations

import argparse
from collections import Counter

from src.collect import arxiv_collector, s2_collector
from src.collect.merge import merge_records
from src.utils import DATA_DIR, RAW_DIR, ensure_dirs, read_jsonl, write_jsonl

DEFAULT_CATEGORIES = ["cs.CL", "cs.IR", "cs.LG", "cs.CV", "cs.AI"]
DEFAULT_QUERIES = [
    "natural language processing",
    "information retrieval",
    "machine learning",
    "computer vision",
    "text mining",
]


def print_stats(corpus: list[dict]) -> None:
    n = len(corpus)
    src_counts = Counter(s for d in corpus for s in d["sources"])
    both = sum(1 for d in corpus if len(d["sources"]) > 1)
    years = [d["year"] for d in corpus if d.get("year")]
    cats = Counter(c for d in corpus for c in d.get("categories", []))
    avg_len = sum(len(d["abstract"]) for d in corpus) / max(n, 1)
    print("\n=== Corpus statistics ===")
    print(f"Documents:             {n}")
    print(f"From arXiv:            {src_counts.get('arxiv', 0)}")
    print(f"From Semantic Scholar: {src_counts.get('semantic_scholar', 0)}")
    print(f"Found in both sources: {both}")
    if years:
        print(f"Year range:            {min(years)}-{max(years)}")
    print(f"Avg abstract length:   {avg_len:.0f} chars")
    top = ", ".join(f"{c} ({k})" for c, k in cats.most_common(6))
    print(f"Top categories:        {top}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arxiv-per-category", type=int, default=600)
    parser.add_argument("--s2-per-query", type=int, default=800)
    parser.add_argument("--year-from", type=int, default=2019)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    parser.add_argument("--skip-arxiv", action="store_true",
                        help="reuse data/raw/arxiv.jsonl instead of fetching")
    parser.add_argument("--skip-s2", action="store_true",
                        help="reuse data/raw/semantic_scholar.jsonl instead of fetching")
    args = parser.parse_args()
    ensure_dirs()

    arxiv_path = RAW_DIR / "arxiv.jsonl"
    if args.skip_arxiv:
        arxiv_records = read_jsonl(arxiv_path) if arxiv_path.exists() else []
        print(f"Reusing {len(arxiv_records)} arXiv records from {arxiv_path}")
    else:
        arxiv_records = arxiv_collector.collect(args.categories,
                                                args.arxiv_per_category)
        write_jsonl(arxiv_path, arxiv_records)
        print(f"Saved {len(arxiv_records)} arXiv records -> {arxiv_path}")

    s2_path = RAW_DIR / "semantic_scholar.jsonl"
    if args.skip_s2:
        s2_records = read_jsonl(s2_path) if s2_path.exists() else []
        print(f"Reusing {len(s2_records)} Semantic Scholar records from {s2_path}")
    else:
        s2_records = s2_collector.collect(args.queries, args.s2_per_query,
                                          year_from=args.year_from)
        write_jsonl(s2_path, s2_records)
        print(f"Saved {len(s2_records)} Semantic Scholar records -> {s2_path}")

    corpus = merge_records(arxiv_records, s2_records)
    corpus_path = DATA_DIR / "corpus.jsonl"
    write_jsonl(corpus_path, corpus)
    print(f"Wrote merged corpus -> {corpus_path}")
    print_stats(corpus)


if __name__ == "__main__":
    main()
