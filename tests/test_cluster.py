"""Unit tests for src.cluster on a tiny two-topic corpus."""

from src.cluster import cluster_matrix, representatives, top_terms, vectorize

# Two unmistakable topics: cooking (docs 0-3) and astronomy (docs 4-7).
DOCS = [
    {"title": "baking bread with sourdough yeast", "abstract": "flour dough oven"},
    {"title": "pasta sauce recipes", "abstract": "tomato garlic basil dough"},
    {"title": "roasting vegetables in the oven", "abstract": "garlic olive flour"},
    {"title": "sourdough starter guide", "abstract": "yeast flour dough oven"},
    {"title": "galaxy formation and dark matter", "abstract": "telescope stars orbit"},
    {"title": "exoplanet detection with telescopes", "abstract": "orbit stars transit"},
    {"title": "black hole accretion disks", "abstract": "galaxy telescope orbit"},
    {"title": "mapping stars in the milky way", "abstract": "galaxy telescope 2024"},
]


def test_two_topics_separate_cleanly():
    matrix, _ = vectorize(DOCS)
    kmeans = cluster_matrix(matrix, k=2, seed=42)
    labels = kmeans.labels_
    cooking = {labels[i] for i in range(4)}
    astronomy = {labels[i] for i in range(4, 8)}
    assert len(cooking) == 1 and len(astronomy) == 1
    assert cooking != astronomy


def test_top_terms_describe_the_topics_and_skip_digits():
    matrix, vectorizer = vectorize(DOCS)
    kmeans = cluster_matrix(matrix, k=2, seed=42)
    terms = top_terms(kmeans, vectorizer, n=5)
    flat = {t for cluster in terms for t in cluster}
    assert "galaxi" in flat or "telescop" in flat   # stems, on purpose
    assert "dough" in flat or "flour" in flat
    assert "2024" not in flat                        # digits filtered


def test_representatives_come_from_their_own_cluster():
    matrix, _ = vectorize(DOCS)
    kmeans = cluster_matrix(matrix, k=2, seed=42)
    for cluster_id in (0, 1):
        for doc_idx in representatives(matrix, kmeans, cluster_id, n=2):
            assert kmeans.labels_[doc_idx] == cluster_id
