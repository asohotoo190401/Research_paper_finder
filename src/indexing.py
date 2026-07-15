"""The inverted index -- the core data structure of every search engine.

A linear scan answers a query in time proportional to the size of the
corpus. An inverted index answers it in time proportional to the number
of *matching* documents, because it maps each term directly to the
documents that contain it (its "postings").

We index each document's title + abstract, and we store more than the
bare document ids: for every (term, document) pair we keep the term
frequency (tf), and per document we keep its length in tokens. Boolean
retrieval (Day 2) only needs the ids, but TF-IDF and BM25 (Day 3) need
tf, document frequency (df = postings list length), document lengths and
the average document length -- storing them now means the ranked models
plug in later without re-indexing.

Build time for the full corpus is a couple of seconds, so we simply
rebuild in memory at startup instead of persisting the index to disk;
at this scale, persistence would add complexity without buying anything.
"""

from collections import Counter
from dataclasses import dataclass, field

from .preprocess import preprocess
from .utils import clean_whitespace


def doc_text(record: dict) -> str:
    """The searchable text of a corpus record: title + abstract."""
    title = record.get("title") or ""
    abstract = record.get("abstract") or ""
    return clean_whitespace(f"{title} {abstract}")


@dataclass
class InvertedIndex:
    # term -> {doc_id -> term frequency}; df(term) == len(postings[term])
    postings: dict[str, dict[int, int]]
    # doc_id -> document length in tokens (after preprocessing)
    doc_lengths: list[int]
    N: int  # number of documents
    avgdl: float  # average document length (needed by BM25)
    all_docs: frozenset[int] = field(default_factory=frozenset)

    def df(self, term: str) -> int:
        """Document frequency: in how many documents does `term` occur?"""
        return len(self.postings.get(term, ()))

    def docs_containing(self, term: str) -> set[int]:
        """The set of doc ids whose text contains `term` (for Boolean ops)."""
        return set(self.postings.get(term, ()))

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)


def build_index(docs: list[dict]) -> InvertedIndex:
    """Build the inverted index over a list of corpus records.

    Documents are identified by their position in `docs` (0..N-1); the
    caller keeps the same list around to map ids back to records.
    """
    postings: dict[str, dict[int, int]] = {}
    doc_lengths: list[int] = []
    for doc_id, record in enumerate(docs):
        tokens = preprocess(doc_text(record))
        doc_lengths.append(len(tokens))
        for term, tf in Counter(tokens).items():
            postings.setdefault(term, {})[doc_id] = tf
    N = len(docs)
    avgdl = (sum(doc_lengths) / N) if N else 0.0
    return InvertedIndex(
        postings=postings,
        doc_lengths=doc_lengths,
        N=N,
        avgdl=avgdl,
        all_docs=frozenset(range(N)),
    )


def _main() -> None:
    """`python -m src.indexing` -- build the index and print statistics."""
    import time

    from .utils import DATA_DIR, read_jsonl

    docs = read_jsonl(DATA_DIR / "corpus.jsonl")
    start = time.perf_counter()
    index = build_index(docs)
    elapsed = time.perf_counter() - start

    print(f"Documents indexed : {index.N:,}")
    print(f"Vocabulary size   : {index.vocabulary_size:,} unique terms")
    print(f"Avg doc length    : {index.avgdl:.1f} tokens")
    print(f"Build time        : {elapsed:.1f}s")
    top = sorted(index.postings.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
    print("Highest-df terms  :", ", ".join(f"{t} ({len(p)})" for t, p in top))


if __name__ == "__main__":
    _main()
