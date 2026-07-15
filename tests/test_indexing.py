"""Unit tests for src.indexing on a three-document toy corpus."""

import pytest

from src.indexing import build_index, doc_text

DOCS = [
    {"title": "Neural networks for image classification",
     "abstract": "Deep neural networks classify images."},
    {"title": "Quantum computing", "abstract": "Qubits and entanglement."},
    {"title": "Graph neural networks", "abstract": "Message passing on graphs."},
]


@pytest.fixture(scope="module")
def index():
    return build_index(DOCS)


def test_doc_text_joins_title_and_abstract_and_survives_none():
    assert doc_text({"title": "A", "abstract": None}) == "A"
    assert doc_text(DOCS[1]) == "Quantum computing Qubits and entanglement."


def test_corpus_level_statistics(index):
    assert index.N == 3
    assert index.all_docs == frozenset({0, 1, 2})
    assert index.avgdl == pytest.approx(sum(index.doc_lengths) / 3)
    assert all(length > 0 for length in index.doc_lengths)


def test_postings_hold_term_frequencies(index):
    # "neural" appears twice in doc 0 (title + abstract) and once in doc 2.
    assert index.postings["neural"] == {0: 2, 2: 1}
    assert index.df("neural") == 2
    assert index.docs_containing("neural") == {0, 2}


def test_index_terms_are_stemmed(index):
    # Documents say "networks"/"graphs"; the index stores the stems.
    assert "networks" not in index.postings
    assert index.docs_containing("network") == {0, 2}
    assert index.docs_containing("graph") == {2}


def test_stopwords_never_reach_the_index(index):
    for stopword in ("for", "and", "on"):
        assert stopword not in index.postings


def test_unknown_term_has_empty_postings(index):
    assert index.df("blockchain") == 0
    assert index.docs_containing("blockchain") == set()
