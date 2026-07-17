"""PaperFinder web app: the search engine with a face.

    python -m src.app          # dev server at http://127.0.0.1:5000

Design notes:

* Everything heavy -- reading the corpus, building the inverted index,
  precomputing the VSM document norms -- happens ONCE at import time.
  Under a WSGI host (PythonAnywhere) the module is imported when the
  worker starts, so every request after that only pays for scoring:
  a few milliseconds.
* The app reuses the exact MODELS registry the CLI uses; the web layer
  adds nothing but presentation, which is how it should be.
* Boolean query syntax errors are caught and shown as a friendly
  message instead of a stack trace.
"""

import time

from flask import Flask, render_template, request

from .indexing import build_index
from .models.boolean import QuerySyntaxError
from .search_cli import MODELS
from .utils import DATA_DIR, read_jsonl

DOCS = read_jsonl(DATA_DIR / "corpus.jsonl")
INDEX = build_index(DOCS)
ENGINES = {name: cls(INDEX) for name, cls in MODELS.items()}
STATS = {
    "docs": INDEX.N,
    "terms": INDEX.vocabulary_size,
    "sources": "arXiv + Semantic Scholar",
}
RESULTS_PER_PAGE = 20

app = Flask(__name__)


@app.route("/")
def search():
    query = request.args.get("q", "").strip()
    model_name = request.args.get("model", "bm25")
    if model_name not in ENGINES:
        model_name = "bm25"

    results, total, elapsed_ms, error = None, 0, 0.0, None
    if query:
        try:
            started = time.perf_counter()
            hits, total = ENGINES[model_name].search(query, RESULTS_PER_PAGE)
            elapsed_ms = (time.perf_counter() - started) * 1000
            results = [(DOCS[doc_idx], score) for doc_idx, score in hits]
        except QuerySyntaxError as exc:
            error = f"Query error: {exc}"

    return render_template(
        "index.html",
        q=query,
        model=model_name,
        models=["bm25", "tfidf", "boolean"],
        results=results,
        total=total,
        elapsed_ms=elapsed_ms,
        error=error,
        stats=STATS,
    )


if __name__ == "__main__":
    app.run(debug=True)
