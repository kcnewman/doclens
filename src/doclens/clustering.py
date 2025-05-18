from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster_documents(tfidf_matrix, n_clusters=5):
    """Cluster documents using KMeans"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    return kmeans, cluster_labels


def reduce_dimensions(tfidf_matrix, n_components=2):
    """Reduce dimensionality using PCA"""
    pca = PCA(n_components=n_components)
    reduced_matrix = pca.fit_transform(tfidf_matrix.toarray())

    return pca, reduced_matrix
