"""The vector space model: TF-IDF weights + cosine similarity.

Boolean retrieval answers "which documents match?" -- this model answers
the question users actually have: "which documents match *best*?" Both
documents and the query become vectors in a space with one dimension per
vocabulary term, and documents are ranked by the cosine of the angle
between their vector and the query's.

Each vector component is a TF-IDF weight:

    w(term, doc) = (1 + log10 tf) * log10(N / df)

* tf (term frequency): how often the term occurs in the document. The
  log damps it -- ten occurrences signal more relevance than one, but
  not ten times more.
* idf (inverse document frequency): log10(N/df) measures how *rare* the
  term is across the corpus. Rare terms are discriminating; a term that
  appears in every document has idf = 0 and contributes nothing at all.

Cosine similarity divides the dot product by both vector lengths, which
is what keeps long documents from winning simply by containing more
words. Document norms are precomputed once at startup (one pass over
the index); scoring a query then only touches the postings of the
query's own terms -- the inverted index doing its job again.
"""

import math
from collections import Counter

from ..indexing import InvertedIndex
from ..preprocess import preprocess


class VectorSpaceModel:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self._doc_norms = self._compute_doc_norms()

    def _idf(self, term: str) -> float:
        df = self.index.df(term)
        return math.log10(self.index.N / df) if df else 0.0

    def _compute_doc_norms(self) -> list[float]:
        """Euclidean length of every document's TF-IDF vector."""
        sq_norms = [0.0] * self.index.N
        for term, plist in self.index.postings.items():
            idf = self._idf(term)
            if idf == 0.0:  # term in every doc: zero weight everywhere
                continue
            for doc_id, tf in plist.items():
                weight = (1.0 + math.log10(tf)) * idf
                sq_norms[doc_id] += weight * weight
        return [math.sqrt(sq) for sq in sq_norms]

    def search(self, query: str, k: int = 10):
        """Rank documents by cosine similarity to the query.

        Returns (hits, total): the top-k list of (doc_id, score) sorted
        by descending score, and the total number of documents with a
        nonzero score.
        """
        scores: dict[int, float] = {}
        q_sq_norm = 0.0
        for term, qtf in Counter(preprocess(query)).items():
            idf = self._idf(term)
            if idf == 0.0:  # unknown term, or term with no signal
                continue
            q_weight = (1.0 + math.log10(qtf)) * idf
            q_sq_norm += q_weight * q_weight
            for doc_id, tf in self.index.postings[term].items():
                d_weight = (1.0 + math.log10(tf)) * idf
                scores[doc_id] = scores.get(doc_id, 0.0) + q_weight * d_weight
        if not scores:
            return [], 0
        q_norm = math.sqrt(q_sq_norm)
        ranked = sorted(
            ((doc_id, dot / (q_norm * self._doc_norms[doc_id]))
             for doc_id, dot in scores.items()
             if self._doc_norms[doc_id] > 0.0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:k], len(ranked)
