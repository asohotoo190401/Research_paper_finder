"""Unit tests for src.models.boolean: parser, precedence, set semantics."""

import pytest

from src.indexing import build_index
from src.models.boolean import QuerySyntaxError, search

DOCS = [
    {"title": "Deep learning for vision", "abstract": "convolutional networks"},
    {"title": "Reinforcement learning agents", "abstract": "reward optimization"},
    {"title": "Quantum error correction", "abstract": "stabilizer codes"},
    {"title": "Deep reinforcement learning", "abstract": "policy gradients"},
]


@pytest.fixture(scope="module")
def index():
    return build_index(DOCS)


def test_single_term(index):
    assert search(index, "learning") == {0, 1, 3}


def test_adjacent_terms_imply_and(index):
    assert search(index, "deep learning") == {0, 3}
    assert search(index, "deep AND learning") == {0, 3}


def test_or(index):
    assert search(index, "quantum OR vision") == {0, 2}


def test_not(index):
    assert search(index, "learning AND NOT reinforcement") == {0}
    assert search(index, "NOT learning") == {2}


def test_operator_precedence_not_over_and_over_or(index):
    # a OR b AND c  ==  a OR (b AND c)
    assert search(index, "quantum OR deep AND reinforcement") == {2, 3}
    assert search(index, "(quantum OR deep) AND reinforcement") == {3}


def test_query_terms_go_through_the_document_pipeline(index):
    # The corpus says "agents"; both singular and plural queries match
    # because query terms are stemmed exactly like document terms.
    assert search(index, "agent") == {1}
    assert search(index, "agents") == {1}


def test_lowercase_and_is_a_stopword_not_an_operator(index):
    assert search(index, "deep and learning") == {0, 3}


def test_query_of_only_stopwords_matches_nothing(index):
    assert search(index, "the of a") == set()


def test_unknown_term_matches_nothing(index):
    assert search(index, "blockchain") == set()


@pytest.mark.parametrize("bad", ["deep AND (learning", "AND deep", "deep OR", "()"])
def test_malformed_queries_raise(index, bad):
    with pytest.raises(QuerySyntaxError):
        search(index, bad)
