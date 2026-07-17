# PaperFinder - a research paper search engine

Final project for the Text Mining / Information Retrieval course.

**Live demo:** https://sohotoo19akhilesh.pythonanywhere.com

**Problem.** Relevant research is scattered across platforms, and keyword
search on any single one misses papers that live elsewhere or are described
with different vocabulary. PaperFinder collects paper metadata from two
independent sources (the arXiv API and the Semantic Scholar Graph API),
integrates and deduplicates them into a single corpus, and provides ranked
full-text search over titles and abstracts using three retrieval models
(Boolean, TF-IDF vector space, BM25), plus K-means topic clustering and a
quantitative + qualitative evaluation.

## Pipeline

```
data collection (arXiv API + Semantic Scholar API)
    -> merge & deduplicate (cross-source integration)
    -> preprocessing (tokenize, normalize, stopwords, stemming)
    -> inverted index (tf, df, document lengths)
    -> retrieval (Boolean | TF-IDF | BM25) + K-means clustering
    -> evaluation (P@k, Recall, MAP + qualitative)
    -> Flask web app
```

## Project structure

```
src/
  collect/        data collectors, cross-source merge, collection CLI
  preprocess.py   shared text pipeline (tokenize, stopwords, Porter stemmer)
  indexing.py     inverted index with tf, df, and document lengths
  models/         boolean.py (query parser), vsm.py (TF-IDF), bm25.py
  search_cli.py   interactive search from the terminal
  cluster.py      K-means topic clustering over TF-IDF vectors
  evaluate.py     P@k / Recall / MAP against pooled relevance judgments
  app.py          Flask web app (templates/ holds the UI)
tests/            60 offline unit + integration tests (no network needed)
data/
  raw/            per-source raw records (JSONL)
  corpus.jsonl    merged, deduplicated corpus (6,303 papers)
  judgments.json  pooled relevance judgments for 8 evaluation queries
```

## Quickstart

```
python -m venv .venv
source .venv/bin/activate            # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
python -m pytest -q                  # 60 passed
python -m src.search_cli             # interactive search (BM25 default)
python -m src.cluster --k 8          # topic clustering report
python -m src.evaluate               # metrics table
python -m src.app                    # web app at http://127.0.0.1:5000
python -m src.collect.run            # (optional) rebuild the corpus, ~10 min
```

## Methodology and results

### 1. Data collection and integration

Two collectors with polite rate limiting and retry-with-backoff pull recent
AI/ML/NLP/IR papers: the arXiv Atom API (2,475 records) and the Semantic
Scholar Graph bulk-search API (3,831 records). Records are normalized into
one schema and deduplicated by a three-level cascade: shared arXiv id, then
DOI, then normalized title. The merged corpus holds 6,303 papers; 3
cross-source duplicates were detected and merged. The overlap is small by
construction: the arXiv pull is newest-first (2,819 papers from 2026) and
Semantic Scholar's index lags very recent preprints and covers additional
venues. Integration here means a unified schema and a working dedup
mechanism across two structurally different APIs (Atom XML feed vs JSON
REST), not a large intersection.

### 2. Preprocessing

One pipeline is used for documents and queries alike: lowercase and
tokenize on alphanumeric runs, remove classic English stopwords, then apply
Porter stemming (cached per unique token for speed). Sharing the pipeline
is what makes a query for "networks" match documents indexed under the stem
"network".

### 3. Indexing

The inverted index maps each term to its postings {doc_id: tf} and also
stores document lengths, so document frequency, tf, and length statistics
are all available to the ranked models without re-indexing. Statistics for
this corpus: 28,798 unique terms, average document length 155.7 tokens,
build time about 2 seconds, queries answered in under 10 ms. At this scale
the index is rebuilt in memory at startup; persistence would add complexity
without benefit.

### 4. Retrieval models

* **Boolean**: a recursive-descent parser supports AND / OR / NOT and
  parentheses with standard precedence; evaluation is set intersection,
  union, and complement over postings. Precise and explainable, but
  unranked.
* **TF-IDF vector space**: weights w = (1 + log10 tf) * log10(N/df),
  ranking by cosine similarity with precomputed document norms. A term
  occurring in every document has idf 0 and contributes nothing.
* **BM25** (k1 = 1.5, b = 0.75): the Lucene/Elasticsearch default. Adds
  term-frequency saturation (contribution asymptotes at k1 + 1) and
  tunable document-length normalization, with a non-negative idf variant.

### 5. Topic clustering

K-means over TF-IDF vectors built with the same preprocessing pipeline
(sublinear tf, L2 normalization so Euclidean distance behaves like cosine,
min_df = 2). A silhouette scan over k = 4..12 is nearly flat (about 0.003
to 0.004), which is typical for overlapping short-text topics; k = 8 was
chosen for readability and the scan is reported rather than hidden. The
eight clusters found (sizes and top centroid stems):

| size | topic (top stems) |
|-----:|-------------------|
| 1220 | general ML: model, learn, data, predict, optim, train |
| 1069 | AI in research/industry: research, studi, ai, system, technolog |
|  812 | computer vision: imag, detect, vision, network, deep, featur |
|  809 | LLM agents: agent, llm, reason, evalu, task, benchmark |
|  725 | multimodal generation: imag, gener, modal, visual, semant |
|  644 | text mining / sentiment: text, sentiment, mine, analysi, extract |
|  536 | retrieval / RAG: retriev, queri, rag, document, answer, augment |
|  488 | clinical ML: patient, clinic, diseas, health, medic, cancer |

That a coherent retrieval/RAG cluster emerges unsupervised from the corpus
of a retrieval project is a satisfying sanity check.

### 6. Evaluation

Pooled judgments: for 8 test queries, the top results of all three models
were pooled and judged for topical relevance at title level, giving 82
relevant documents (data/judgments.json, keyed by stable doc ids). Metrics
at ranking depth 100:

| model   |   P@5 |  P@10 |  R@10 |   MAP |
|---------|------:|------:|------:|------:|
| boolean | 0.600 | 0.450 | 0.457 | 0.434 |
| tfidf   | 0.750 | 0.650 | 0.663 | 0.638 |
| bm25    | 0.775 | 0.625 | 0.640 | **0.699** |

The ordering matches theory: both ranked models clearly beat unranked
Boolean (which requires every query term and returns matches in arbitrary
order), and BM25 leads on MAP and P@5 because it places relevant documents
earliest. TF-IDF is marginally better at depth 10 on this small query set.
Qualitatively, Boolean's failure modes are visible in the pools: it missed
obviously relevant papers lacking one query term and surfaced a
speech-diarization paper for a hallucination query. A classic vocabulary
trap also appears: "code generation" retrieves "code-mixed" linguistics
papers, a homonym no term-matching model can resolve.

Caveats, stated plainly: judgments are title-level and made by the project
author, the query set is small, and recall is relative to the judged pool
(standard pooling caveat). The numbers support the model comparison; they
are not absolute effectiveness claims.

## Web app and deployment

`src/app.py` is a small Flask layer over the same model registry the CLI
uses. The corpus is loaded and the index built once at import time
(WSGI-friendly); each request then costs only milliseconds of scoring. The
app is deployed on PythonAnywhere (free tier): clone the repo in a Bash
console, create a virtualenv with flask + nltk, point a manually configured
web app's WSGI file at `src.app:app`, and reload. The stemmer needs no
NLTK data downloads, which keeps the server setup to two pip installs.

## Limitations and future work

Phrase queries and term proximity are unsupported; stemming occasionally
surprises (top cluster terms are stems); the corpus is a snapshot skewed
toward 2026 arXiv; evaluation would firm up with more queries, more
judges, and abstract-level judging. Natural extensions: incremental corpus
refresh, query expansion, learned ranking on top of BM25 candidates, and
cluster labels generated from representative titles instead of stems.
