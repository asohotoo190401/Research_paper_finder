"""Okapi BM25 -- the ranking function behind Lucene, Elasticsearch and
most production search engines. It keeps TF-IDF's core intuitions but
fixes two of its weaknesses with two tunable parameters:

    score(q, d) = sum over query terms t of
        idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))

* k1 (default 1.5) controls *term-frequency saturation*. In TF-IDF, tf's
  contribution grows without bound (log-damped, but unbounded). In BM25
  it asymptotically approaches k1 + 1: the jump from tf=1 to tf=2
  matters a lot, from tf=20 to tf=21 barely at all. A document can't win
  just by repeating a word.
* b (default 0.75) controls *length normalization*: how strongly a
  document's length (dl) relative to the corpus average (avgdl) deflates
  its tf. b=1 is full normalization, b=0 ignores length entirely --
  cosine gives you no such dial.

We use the idf variant Lucene uses,

    idf(t) = ln( (N - df + 0.5) / (df + 0.5) + 1 )

which stays non-negative even for terms in more than half the corpus.
Everything the formula needs -- tf, df, document lengths, avgdl -- was
stored in the inverted index on Day 2, so this file is pure math over
existing data structures.
"""

import math
from collections import Counter

from ..indexing import InvertedIndex
from ..preprocess import preprocess


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def _idf(self, term: str) -> float:
        df = self.index.df(term)
        if df == 0:
            return 0.0
        return math.log((self.index.N - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, k: int = 10):
        """Rank documents by BM25 score against the query.

        Returns (hits, total): the top-k list of (doc_id, score) sorted
        by descending score, and the total number of scored documents.
        Repeated query terms count with their query-side frequency.
        """
        scores: dict[int, float] = {}
        k1, b = self.k1, self.b
        avgdl = self.index.avgdl or 1.0
        for term, qtf in Counter(preprocess(query)).items():
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for doc_id, tf in self.index.postings[term].items():
                dl = self.index.doc_lengths[doc_id]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + \
                    qtf * idf * tf * (k1 + 1.0) / denom
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:k], len(ranked)
