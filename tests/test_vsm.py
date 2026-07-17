"""Unit tests for src.models.vsm: TF-IDF weighting and cosine ranking."""

import pytest

from src.indexing import build_index
from src.models.vsm import VectorSpaceModel

# "shared" appears in every doc (idf = 0); "banana" in exactly one.
# Docs 1-3 share a second weighted term ("kiwi") so that cosine can
# actually distinguish them -- a doc whose ONLY weighted term is the
# query term always scores exactly 1.0 regardless of tf.
DOCS = [
    {"title": "shared banana", "abstract": ""},
    {"title": "shared cherry kiwi", "abstract": ""},
    {"title": "shared cherry cherry cherry kiwi", "abstract": ""},
    {"title": "shared cherry kiwi mango papaya guava lychee durian", "abstract": ""},
]


@pytest.fixture(scope="module")
def model():
    return VectorSpaceModel(build_index(DOCS))


def _ranked_ids(hits):
    return [doc_id for doc_id, _ in hits]


def test_hand_computed_cosine_is_exactly_one(model):
    # Doc 0's only nonzero-weight term is "banana" (its other term,
    # "shared", has idf 0). The query "banana" therefore points in
    # exactly the same direction as doc 0's vector: cosine = 1.0.
    hits, total = model.search("banana")
    assert total == 1
    assert hits[0][0] == 0
    assert hits[0][1] == pytest.approx(1.0)


def test_term_in_every_document_has_no_signal(model):
    # idf = log(N/N) = 0: the term contributes nothing, matches nothing.
    hits, total = model.search("shared")
    assert (hits, total) == ([], 0)


def test_higher_tf_ranks_higher(model):
    # Docs 1 and 2 both contain "cherry"; doc 2 says it three times.
    hits, _ = model.search("cherry")
    ranking = _ranked_ids(hits)
    assert ranking.index(2) < ranking.index(1)


def test_cosine_punishes_dilution(model):
    # Docs 1 and 3 each contain "cherry" and "kiwi" once, but doc 3
    # buries them among five other rare (heavily weighted) terms, so its
    # vector points far less toward the query: length normalization.
    hits, _ = model.search("cherry")
    ranking = _ranked_ids(hits)
    assert ranking.index(1) < ranking.index(3)


def test_scores_sorted_descending_and_bounded(model):
    hits, _ = model.search("cherry banana")
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < score <= 1.0 + 1e-9 for score in scores)


def test_query_terms_are_stemmed_like_documents(model):
    hits, total = model.search("bananas")
    assert total == 1 and hits[0][0] == 0


def test_k_limits_results(model):
    hits, total = model.search("cherry", k=2)
    assert len(hits) == 2 and total == 3


def test_unknown_terms_match_nothing(model):
    assert model.search("blockchain") == ([], 0)
