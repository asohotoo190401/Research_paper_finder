"""Interactive command-line search over the corpus.

    python -m src.search_cli                              # BM25 (default)
    python -m src.search_cli --model tfidf
    python -m src.search_cli --model boolean --query "graph AND neural"

All three retrieval models sit behind one interface: construct with the
index, then .search(query, k) -> (top hits, total matches). Boolean
takes operator queries (AND/OR/NOT, uppercase); the ranked models take
free text -- an uppercase operator typed there is simply lowercased into
a stopword and vanishes, which is exactly what you want.
"""

import argparse
import time

from .indexing import InvertedIndex, build_index
from .models import boolean
from .models.bm25 import BM25
from .models.vsm import VectorSpaceModel
from .utils import DATA_DIR, read_jsonl


class BooleanModel:
    """Adapter giving Boolean retrieval the same interface as the
    ranked models. Matches carry no score -- the model is unranked by
    definition, so we show them in corpus order."""

    def __init__(self, index: InvertedIndex):
        self.index = index

    def search(self, query: str, k: int = 10):
        matches = boolean.search(self.index, query)
        return [(doc_id, None) for doc_id in sorted(matches)[:k]], len(matches)


MODELS = {
    "boolean": BooleanModel,
    "tfidf": VectorSpaceModel,
    "bm25": BM25,
}


def _format_hit(rank: int, record: dict, score) -> str:
    title = (record.get("title") or "(untitled)").strip()
    if len(title) > 90:
        title = title[:87] + "..."
    year = record.get("year") or "----"
    source = "+".join(record.get("sources") or ["?"])
    score_part = f"  score={score:.4f}" if score is not None else ""
    return f"{rank:3d}. [{year}] {title}  ({source}){score_part}"


def _answer(model, docs: list[dict], query: str, top: int) -> None:
    try:
        started = time.perf_counter()
        hits, total = model.search(query, top)
        ms = (time.perf_counter() - started) * 1000
    except boolean.QuerySyntaxError as exc:
        print(f"  ! query error: {exc}")
        return
    if total == 0:
        print("  no matches")
        return
    note = ""
    if isinstance(model, BooleanModel) and total > len(hits):
        note = f" (unranked model; showing first {len(hits)})"
    print(f"  {total} match{'es' if total != 1 else ''} in {ms:.1f} ms{note}")
    for rank, (doc_id, score) in enumerate(hits, start=1):
        print(_format_hit(rank, docs[doc_id], score))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Search the PaperFinder corpus")
    parser.add_argument("--model", choices=sorted(MODELS), default="bm25")
    parser.add_argument("--corpus", default=str(DATA_DIR / "corpus.jsonl"))
    parser.add_argument("--top", type=int, default=10, help="results to show")
    parser.add_argument("--query", help="run one query and exit")
    args = parser.parse_args(argv)

    docs = read_jsonl(args.corpus)
    started = time.perf_counter()
    index = build_index(docs)
    model = MODELS[args.model](index)
    print(f"Indexed {index.N:,} documents / {index.vocabulary_size:,} terms "
          f"in {time.perf_counter() - started:.1f}s. Model: {args.model}.")

    if args.query:
        _answer(model, docs, args.query, args.top)
        return

    if args.model == "boolean":
        print("Operators: AND, OR, NOT, parentheses (uppercase). "
              "Type \\q (or Ctrl-D) to quit.")
    else:
        print("Free-text queries; results ranked by relevance. "
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
            _answer(model, docs, query, args.top)


if __name__ == "__main__":
    main()
