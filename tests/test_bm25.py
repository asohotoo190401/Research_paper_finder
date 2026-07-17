"""Unit tests for src.models.bm25: saturation, length norm, idf."""

import pytest

from src.indexing import build_index
from src.models.bm25 import BM25

# Docs 0-2: "cat" with tf = 1, 2, 3, padded with unique filler words so
# all three have the SAME length (isolates the tf effect from length
# normalization). Doc 3 is a long document with tf("cat") = 1.
# "dog" appears in one doc, "bird" in three (rare vs. common-ish).
DOCS = [
    {"title": "cat zaa zab zac zad zae zaf zag zah zai", "abstract": ""},
    {"title": "cat cat zba zbb zbc zbd zbe zbf zbg zbh", "abstract": ""},
    {"title": "cat cat cat zca zcb zcc zcd zce zcf zcg", "abstract": ""},
    {"title": "cat", "abstract": " ".join(f"zd{i:02d}" for i in range(39))},
    {"title": "dog bird zea zeb zec zed zee zef zeg zeh", "abstract": ""},
    {"title": "bird zfa zfb zfc zfd zfe zff zfg zfh zfi", "abstract": ""},
    {"title": "bird zga zgb zgc zgd zge zgf zgg zgh zgi", "abstract": ""},
]


@pytest.fixture(scope="module")
def index():
    return build_index(DOCS)


@pytest.fixture(scope="module")
def model(index):
    return BM25(index)


def _score_of(hits, doc_id):
    return dict(hits)[doc_id]


def test_equal_length_docs_same_tf_pattern(index):
    # Guard for the fixture itself: docs 0-2 must have identical lengths,
    # or the saturation test below wouldn't isolate tf.
    assert index.doc_lengths[0] == index.doc_lengths[1] == index.doc_lengths[2]
    assert index.doc_lengths[3] == 40


def test_tf_saturates_with_diminishing_returns(model):
    # More occurrences -> higher score, but each extra occurrence adds
    # less than the previous one (the k1 asymptote at work).
    hits, _ = model.search("cat")
    s1, s2, s3 = (_score_of(hits, d) for d in (0, 1, 2))
    assert s1 < s2 < s3
    assert (s2 - s1) > (s3 - s2)


def test_longer_documents_are_penalized(model):
    # Same tf("cat") = 1, but doc 3 is four times longer than doc 0.
    hits, _ = model.search("cat")
    assert _score_of(hits, 0) > _score_of(hits, 3)


def test_b_zero_switches_length_normalization_off(index):
    flat = BM25(index, b=0.0)
    hits, _ = flat.search("cat")
    assert _score_of(hits, 0) == pytest.approx(_score_of(hits, 3))


def test_rarer_terms_contribute_more(model):
    # Doc 4 matches via rare "dog" AND common-ish "bird"; docs 5-6 match
    # via "bird" alone. And within the query, the rare term must be
    # worth more than the common one: check via idf directly.
    assert model._idf("dog") > model._idf("bird") > 0.0
    hits, total = model.search("dog bird")
    assert total == 3
    assert hits[0][0] == 4


def test_query_terms_are_stemmed_like_documents(model):
    hits, total = model.search("cats")
    assert total == 4 and _score_of(hits, 2) > 0.0


def test_unknown_terms_are_ignored(model):
    assert model.search("blockchain") == ([], 0)
    # ...and mixing one in doesn't break a good term.
    hits, total = model.search("blockchain cat")
    assert total == 4


def test_ranking_is_sorted_and_capped(model):
    hits, total = model.search("cat", k=2)
    scores = [score for _, score in hits]
    assert len(hits) == 2 and total == 4
    assert scores == sorted(scores, reverse=True)
