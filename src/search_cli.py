"""Interactive command-line search over the corpus.

    python -m src.search_cli                     # interactive, Boolean model
    python -m src.search_cli --query "graph AND neural"   # one-shot
    python -m src.search_cli --model boolean --top 20

The --model flag is a registry that Day 3 extends with "tfidf" and
"bm25"; the CLI itself won't need to change.
"""

import argparse
import sys
import time

from .indexing import InvertedIndex, build_index
from .models import boolean
from .utils import DATA_DIR, read_jsonl


def _run_boolean(index: InvertedIndex, docs: list[dict], query: str, top: int):
    """Boolean is unranked: return matches in corpus order, no scores."""
    matches = boolean.search(index, query)
    hits = [(doc_id, None) for doc_id in sorted(matches)[:top]]
    return hits, len(matches)


# name -> runner(index, docs, query, top) -> (hits, total_matches)
MODELS = {
    "boolean": _run_boolean,
}


def _format_hit(rank: int, record: dict, score) -> str:
    title = (record.get("title") or "(untitled)").strip()
    if len(title) > 90:
        title = title[:87] + "..."
    year = record.get("year") or "----"
    source = "+".join(record.get("sources") or ["?"])
    score_part = f"  score={score:.4f}" if score is not None else ""
    return f"{rank:3d}. [{year}] {title}  ({source}){score_part}"


def _answer(model_name: str, index, docs, query: str, top: int) -> None:
    runner = MODELS[model_name]
    try:
        started = time.perf_counter()
        hits, total = runner(index, docs, query, top)
        ms = (time.perf_counter() - started) * 1000
    except boolean.QuerySyntaxError as exc:
        print(f"  ! query error: {exc}")
        return
    if total == 0:
        print("  no matches (note: lowercase and/or/not are stopwords; "
              "operators must be UPPERCASE)")
        return
    unranked = " (unranked model; showing first {})".format(len(hits)) \
        if model_name == "boolean" and total > len(hits) else ""
    print(f"  {total} match{'es' if total != 1 else ''} in {ms:.1f} ms{unranked}")
    for rank, (doc_id, score) in enumerate(hits, start=1):
        print(_format_hit(rank, docs[doc_id], score))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Search the PaperFinder corpus")
    parser.add_argument("--model", choices=sorted(MODELS), default="boolean")
    parser.add_argument("--corpus", default=str(DATA_DIR / "corpus.jsonl"))
    parser.add_argument("--top", type=int, default=10, help="results to show")
    parser.add_argument("--query", help="run one query and exit")
    args = parser.parse_args(argv)

    docs = read_jsonl(args.corpus)
    started = time.perf_counter()
    index = build_index(docs)
    print(f"Indexed {index.N:,} documents / {index.vocabulary_size:,} terms "
          f"in {time.perf_counter() - started:.1f}s. Model: {args.model}.")

    if args.query:
        _answer(args.model, index, docs, args.query, args.top)
        return

    print("Operators: AND, OR, NOT, parentheses (uppercase). "
          "Type \\q (or Ctrl-D) to quit.")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query in {"\\q", "quit", "exit"}:
            return
        if query:
            _answer(args.model, index, docs, query, args.top)


if __name__ == "__main__":
    main()
