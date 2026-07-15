"""Unit tests for src.preprocess -- all offline, pure functions."""

from src.preprocess import STOPWORDS, preprocess, tokenize


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("State-of-the-Art: GPT-4!") == [
        "state", "of", "the", "art", "gpt", "4",
    ]


def test_tokenize_keeps_alphanumeric_terms_whole():
    assert tokenize("BM25 outperforms TF-IDF") == ["bm25", "outperforms", "tf", "idf"]


def test_tokenize_handles_empty_and_none():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_stopwords_are_removed():
    assert preprocess("the cat and the hat", stem=False) == ["cat", "hat"]
    assert "the" in STOPWORDS and "and" in STOPWORDS


def test_stemming_collapses_morphological_variants():
    assert preprocess("network networks networking") == [
        "network", "network", "network",
    ]


def test_full_pipeline_on_a_sentence():
    out = preprocess("Retrieval of documents, and retrieving a document.")
    assert out == ["retriev", "document", "retriev", "document"]


def test_flags_can_disable_stages():
    text = "The Networks"
    assert preprocess(text, remove_stopwords=False, stem=False) == ["the", "networks"]
    assert preprocess(text, stem=False) == ["networks"]
