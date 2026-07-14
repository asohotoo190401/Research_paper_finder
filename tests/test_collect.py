"""Unit tests for the collection layer (parsers + merge/dedup).

Run with:  python -m pytest -q
These use embedded API-response fixtures, so they run fully offline.
"""

from src.collect.arxiv_collector import parse_feed, strip_version
from src.collect.merge import merge_records, normalize_title
from src.collect.s2_collector import parse_record

ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <title>Neural Ranking Models
      for Document Retrieval</title>
    <summary>We study neural ranking models for full-text document
      retrieval and compare them against classical lexical baselines such
      as BM25 across several benchmark collections.</summary>
    <published>2024-01-02T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <category term="cs.IR"/>
    <category term="cs.CL"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cs/0112017v1</id>
    <title>A Study of Old-Style Identifiers</title>
    <summary>This entry exists to test parsing of pre-2007 arXiv
      identifiers, which contain a subject prefix and a slash before the
      numeric part of the identifier string.</summary>
    <published>2001-12-11T00:00:00Z</published>
    <author><name>Grace Hopper</name></author>
    <category term="cs.DL"/>
  </entry>
</feed>"""

S2_SAMPLE = {
    "paperId": "abc123",
    "title": "Neural Ranking Models for Document Retrieval",
    "abstract": ("We study neural ranking models for full-text document "
                 "retrieval and compare them against classical lexical "
                 "baselines such as BM25 across several benchmark collections."),
    "year": 2024,
    "authors": [{"authorId": "1", "name": "Ada Lovelace"}],
    "externalIds": {"ArXiv": "2401.00001", "DOI": "10.1234/EXAMPLE"},
    "url": "https://www.semanticscholar.org/paper/abc123",
    "citationCount": 42,
    "fieldsOfStudy": ["Computer Science"],
    "venue": "SIGIR",
}


def test_parse_arxiv_feed():
    records = parse_feed(ATOM_SAMPLE)
    assert len(records) == 2
    first = records[0]
    assert first["arxiv_id"] == "2401.00001"          # version stripped
    assert "\n" not in first["title"]                  # whitespace cleaned
    assert "  " not in first["title"]
    assert first["year"] == 2024
    assert "cs.IR" in first["categories"]
    assert first["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert records[1]["arxiv_id"] == "cs/0112017"      # old-style id survives


def test_strip_version_handles_all_id_styles():
    assert strip_version("2401.00001v2") == "2401.00001"
    assert strip_version("cs/0112017v1") == "cs/0112017"
    assert strip_version("math.CV/0601001v3") == "math.CV/0601001"


def test_parse_s2_record():
    rec = parse_record(S2_SAMPLE)
    assert rec["arxiv_id"] == "2401.00001"
    assert rec["doi"] == "10.1234/example"             # lowercased
    assert rec["citation_count"] == 42
    assert rec["venue"] == "SIGIR"
    short = dict(S2_SAMPLE, abstract="too short")
    assert parse_record(short) is None                 # unusable -> dropped


def test_merge_dedup_across_sources():
    arxiv_recs = parse_feed(ATOM_SAMPLE)
    s2_rec = parse_record(S2_SAMPLE)
    corpus = merge_records(arxiv_recs, [s2_rec], min_abstract_chars=10)
    assert len(corpus) == 2                            # 3 records -> 2 docs
    merged = next(d for d in corpus if d["arxiv_id"] == "2401.00001")
    assert set(merged["sources"]) == {"arxiv", "semantic_scholar"}
    assert merged["citation_count"] == 42              # taken from S2
    assert "cs.IR" in merged["categories"]             # kept from arXiv
    assert merged["venue"] == "SIGIR"                  # filled from S2
    assert merged["doc_id"] == "arxiv:2401.00001"


def test_merge_by_normalized_title():
    a = {"title": "Sparse Retrieval: A Survey!", "abstract": "x" * 50,
         "source": "arxiv", "arxiv_id": "2402.11111"}
    b = {"title": "sparse retrieval - a survey", "abstract": "y" * 80,
         "source": "semantic_scholar", "s2_id": "zzz"}
    assert normalize_title(a["title"]) == normalize_title(b["title"])
    corpus = merge_records([a], [b], min_abstract_chars=10)
    assert len(corpus) == 1
    assert len(corpus[0]["abstract"]) == 80            # longer abstract wins
