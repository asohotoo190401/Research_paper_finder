"""Quantitative evaluation of the three retrieval models.

Methodology (the standard "pooling" approach, scaled to a course
project): for a set of test queries, the top results of ALL models are
pooled and judged for topical relevance; the judged sets live in
data/judgments.json, keyed by stable corpus doc_ids so they survive
re-indexing. Each model is then scored against those judgments with the
classic ranked-retrieval metrics:

* P@k  (precision at k): what fraction of the top k is relevant?
  The "first page quality" metric.
* R@k  (recall at k): what fraction of all judged-relevant documents
  appears in the top k? (Relative to the judged pool -- unjudged
  documents are treated as non-relevant, the standard caveat.)
* AP   (average precision): the mean of P@i over every rank i that
  holds a relevant document; rewards putting relevant results EARLY.
  MAP is the mean of AP over all queries -- the headline number.

Boolean retrieval has no ranking, so its matches are scored in corpus
order. That is not unfair; it is the point: the metrics quantify
exactly what unranked retrieval costs you.

    python -m src.evaluate              # summary table
    python -m src.evaluate --per-query  # per-query breakdown
"""

import argparse
import json

from .indexing import build_index
from .search_cli import MODELS
from .utils import DATA_DIR, read_jsonl

JUDGMENTS_PATH = DATA_DIR / "judgments.json"
DEPTH = 100  # ranking depth retrieved per query for AP


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for doc in ranking[:k] if doc in relevant) / k


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for doc in ranking[:k] if doc in relevant) / len(relevant)


def average_precision(ranking: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, doc in enumerate(ranking, start=1):
        if doc in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def load_judgments(path=JUDGMENTS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_model(model, docs: list[dict], judgments: list[dict]) -> dict:
    """Average P@5, P@10, R@10 and AP for one model over all queries."""
    per_query = []
    for entry in judgments:
        relevant = set(entry["relevant"])
        hits, _ = model.search(entry["query"], DEPTH)
        ranking = [docs[doc_idx]["doc_id"] for doc_idx, _ in hits]
        per_query.append({
            "query": entry["query"],
            "P@5": precision_at_k(ranking, relevant, 5),
            "P@10": precision_at_k(ranking, relevant, 10),
            "R@10": recall_at_k(ranking, relevant, 10),
            "AP": average_precision(ranking, relevant),
        })
    n = len(per_query)
    means = {metric: sum(q[metric] for q in per_query) / n
             for metric in ("P@5", "P@10", "R@10", "AP")}
    return {"means": means, "per_query": per_query}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the retrieval models")
    parser.add_argument("--per-query", action="store_true")
    args = parser.parse_args()

    docs = read_jsonl(DATA_DIR / "corpus.jsonl")
    index = build_index(docs)
    judgments = load_judgments()
    n_rel = sum(len(e["relevant"]) for e in judgments)
    print(f"{len(judgments)} queries, {n_rel} judged-relevant documents\n")

    print(f"{'model':<10} {'P@5':>7} {'P@10':>7} {'R@10':>7} {'MAP':>7}")
    for name in ("boolean", "tfidf", "bm25"):
        model = MODELS[name](index)
        result = evaluate_model(model, docs, judgments)
        m = result["means"]
        print(f"{name:<10} {m['P@5']:>7.3f} {m['P@10']:>7.3f} "
              f"{m['R@10']:>7.3f} {m['AP']:>7.3f}")
        if args.per_query:
            for q in result["per_query"]:
                print(f"    {q['AP']:.3f} AP  P@5={q['P@5']:.2f}  {q['query']}")


if __name__ == "__main__":
    _main()
