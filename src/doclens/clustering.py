from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def knn_index(embeddings, n_neighbors=5):
    """Create KNN index for similarity search

    Args:
        embeddings (_typnp.arraye_): Document embedding matrix
        n_neighbors (int, optional): Number of neighbors to consider. Defaults to 5.

    Returns:
        KNeighborsClassifier: Fitted KNN model
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine")
    knn.fit(embeddings)
    return knn


def cluster_documents(embeddings, n_clusters=5, random_state=42):
    """Cluster documents using KMeans

    Args:
        embeddings (np.array): Document embedding matrix
        n_clusters (int, optional): Number of clusters. Defaults to 5.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (KMeans model, document cluster labels)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels


def get_cluster_centers(kmeans_model):
    """Get cluster centers from KMeans model.

    Args:
        kmeans_model: Fitted KMeans model

    Returns:
        np.array: Cluster centers
    """
    return kmeans_model.cluster_centers_
