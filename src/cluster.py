"""K-means topic clustering: what IS in this corpus?

Retrieval answers "find me X"; clustering answers the complementary
question, "what topics does this collection contain?", with no query at
all (unsupervised learning). We cluster documents in TF-IDF space:

* The TF-IDF matrix is built with the SAME preprocessing pipeline as
  retrieval (our `preprocess` is plugged into scikit-learn's vectorizer
  as the analyzer), with sublinear tf matching our vector space model.
* Vectors are L2-normalized (scikit-learn's default), which makes
  K-means' Euclidean distance behave like cosine similarity, consistent
  with how retrieval compares documents.
* min_df=2 drops terms that occur in a single document; they cannot
  help group documents and would double the matrix width.

K-means then finds k centroids minimizing within-cluster distance. We
read each cluster through two lenses: the highest-weighted terms of its
centroid (what the topic is "about" -- note these are stems), and the
member documents closest to the centroid (the most typical papers).

Choosing k is a judgment call; `--scan` reports the silhouette score
(how well-separated clusters are, higher is better) over a range of k
so the choice can be justified rather than asserted.
"""

import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from .indexing import doc_text
from .preprocess import preprocess
from .utils import DATA_DIR, read_jsonl


def vectorize(docs: list[dict]):
    """TF-IDF matrix over the corpus, using the shared preprocessing."""
    vectorizer = TfidfVectorizer(analyzer=preprocess, sublinear_tf=True, min_df=2)
    matrix = vectorizer.fit_transform(doc_text(d) for d in docs)
    return matrix, vectorizer


def cluster_matrix(matrix, k: int, seed: int = 42) -> KMeans:
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(matrix)
    return kmeans


def top_terms(kmeans: KMeans, vectorizer, n: int = 10) -> list[list[str]]:
    """The n highest-weighted centroid terms for every cluster.

    Pure-number tokens (years, section numbers) are skipped: they can
    carry TF-IDF weight but say nothing about a cluster's topic.
    """
    terms = np.asarray(vectorizer.get_feature_names_out())
    result = []
    for centroid in kmeans.cluster_centers_:
        picked = [terms[i] for i in np.argsort(centroid)[::-1]
                  if not terms[i].isdigit()][:n]
        result.append(picked)
    return result


def representatives(matrix, kmeans: KMeans, cluster_id: int, n: int = 3):
    """Indices of the cluster's members closest to its centroid."""
    members = np.where(kmeans.labels_ == cluster_id)[0]
    distances = kmeans.transform(matrix[members])[:, cluster_id]
    return members[np.argsort(distances)][:n].tolist()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Cluster the corpus into topics")
    parser.add_argument("--k", type=int, default=8, help="number of clusters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--scan", nargs=2, type=int, metavar=("KMIN", "KMAX"),
                        help="report silhouette scores for a range of k")
    args = parser.parse_args()

    docs = read_jsonl(DATA_DIR / "corpus.jsonl")
    matrix, vectorizer = vectorize(docs)
    print(f"TF-IDF matrix: {matrix.shape[0]:,} docs x {matrix.shape[1]:,} terms")

    if args.scan:
        kmin, kmax = args.scan
        print(f"{'k':>3}  {'silhouette':>10}")
        for k in range(kmin, kmax + 1):
            kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=4)
            labels = kmeans.fit_predict(matrix)
            score = silhouette_score(matrix, labels, sample_size=2000,
                                     random_state=0)
            print(f"{k:>3}  {score:>10.4f}")
        return

    kmeans = cluster_matrix(matrix, args.k, args.seed)
    terms_per_cluster = top_terms(kmeans, vectorizer)
    order = np.argsort(-np.bincount(kmeans.labels_))  # biggest cluster first
    for cluster_id in order:
        size = int((kmeans.labels_ == cluster_id).sum())
        print(f"\nCluster {cluster_id} -- {size} papers")
        print("  terms:", ", ".join(terms_per_cluster[cluster_id]))
        for doc_id in representatives(matrix, kmeans, cluster_id, args.examples):
            print(f"  * [{docs[doc_id].get('year')}] {docs[doc_id]['title'][:88]}")


if __name__ == "__main__":
    _main()
