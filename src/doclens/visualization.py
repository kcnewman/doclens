import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def reduce_and_plot(tfidf_matrix, labels=None, title="Document Clusters"):
    """Reduce dimensions and plot document clusters"""
    # Reduce dimensions
    pca = PCA(n_components=2)
    coords = pca.fit_transform(tfidf_matrix.toarray())

    # Create plot
    plt.figure(figsize=(10, 8))

    if labels is not None:
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis")
        plt.colorbar(scatter)
    else:
        plt.scatter(coords[:, 0], coords[:, 1])

    plt.title(title)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")

    return plt
