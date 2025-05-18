import pytest
import numpy as np
from doclens.clustering import cluster_documents, knn_index


@pytest.fixture
def sample_vectors():
    """Sample document vectors for testing"""
    np.random.seed(42)
    return np.random.rand(100, 300)  # 100 documents, 300 dimensions


def test_clustering(sample_vectors):
    n_clusters = 5
    kmeans, labels = cluster_documents(sample_vectors, n_clusters=n_clusters)

    assert len(labels) == sample_vectors.shape[0]
    assert len(np.unique(labels)) == n_clusters
    assert all(label >= 0 for label in labels)


def test_knn_index(sample_vectors):
    knn = knn_index(sample_vectors)

    # Test single query
    query = np.random.rand(300)
    distances, indices = knn.kneighbors(query.reshape(1, -1))

    assert len(indices[0]) == min(5, len(sample_vectors))  # Default k=5
    assert all(0 <= idx < len(sample_vectors) for idx in indices[0])
    assert all(dist >= 0 for dist in distances[0])  # Distances should be non-negative
