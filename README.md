# PaperFinder — a research paper search engine

Final project for the Text Mining / Information Retrieval course.

**Problem.** Relevant research is scattered across platforms, and keyword search on any single one misses papers that live elsewhere or are described with different vocabulary. PaperFinder collects paper metadata from two independent sources — the arXiv API and the Semantic Scholar Graph API — integrates and deduplicates them into a single corpus, and provides ranked full-text search over titles and abstracts using three retrieval models (Boolean, TF-IDF vector space, BM25), plus K-means topic clustering and a quantitative + qualitative evaluation.

## Pipeline

```
data collection (arXiv API + Semantic Scholar API)
        → merge & deduplicate (cross-source integration)
        → preprocessing (tokenize, normalize, stopwords, stem/lemma)
        → inverted index & TF-IDF vectors
        → retrieval (Boolean | VSM | BM25) + K-means clustering
        → evaluation (P@k, Recall, MAP + qualitative)
        → Flask web app
```

## Project structure

```
src/
  collect/     data collectors, cross-source merge, collection CLI
  ...          (preprocessing, indexing, models, evaluation, app — added day by day)
tests/         offline unit tests with embedded API fixtures
data/
  raw/         per-source raw records (JSONL)
  corpus.jsonl merged, deduplicated corpus
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q              # offline unit tests
python -m src.collect.run        # build the corpus (~10 minutes)
```

Optional: export a free Semantic Scholar API key as `S2_API_KEY` for faster,
steadier collection (https://www.semanticscholar.org/product/api). Works
without one too — the client backs off and retries on rate limits.

*Documentation grows with the project — full methodology, evaluation results,
and deployment guide land before submission.*
