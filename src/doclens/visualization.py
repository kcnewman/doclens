import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os


def setup_visualization_dir(vis_dir="../../results/visualizations"):
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir


def reduce_dimensions(embeddings, n_components=2):
    """Reduce dimensions using PCA.

    Args:
        embeddings (np.array): Document embeddings matrix
        n_components (int): Number of components to keep

    Returns:
        np.array: Reduced embeddings
    """
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings, pca


def plot_document_clusters(reduced_embeddings, labels, save_path=None):
    """Plot document clusters in 2D space.

    Args:
        reduced_embeddings (np.array): PCA-reduced embeddings
        labels (np.array): Cluster labels
        save_path (str, optional): Path to save visualization
    """
    plt.figure(figsize=(10, 8))

    unique_clusters = np.unique(labels)
    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(unique_clusters)))
    for cluster_id, color in zip(unique_clusters, colors):
        cluster_points = reduced_embeddings[labels == cluster_id]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=50,
            c=[color],
            alpha=0.7,
            label=f"Cluster {cluster_id}",
        )

    plt.title("Document Clusters (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_sentiment_distribution(sentiments, save_path=None):
    """Plot sentiment distribution.

    Args:
        sentiments (list): Sentiment labels (0 for negative, 1 for positive)
        save_path (str, optional): Path to save visualization
    """
    plt.figure(figsize=(8, 6))
    unique_sentiments, counts = np.unique(sentiments, return_counts=True)
    labels = ["Negative", "Positive"]

    plt.bar(labels, counts, color=["red", "green"])
    plt.title("Sentiment Distribution")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)

    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha="center")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_search_results(query, results, sentiments, save_path=None):
    """Visualize search results with sentiments.

    Args:
        query (str): Search query
        results (list): List of (doc_index, similarity) tuples
        sentiments (list): Sentiment predictions for results
        save_path (str, optional): Path to save visualization
    """
    plt.figure(figsize=(10, 6))

    indices = [r[0] for r in results]
    similarities = [r[1] for r in results]

    colors = ["red" if s == 0 else "green" for s in sentiments]

    plt.barh(indices, similarities, color=colors)
    plt.title(f'Search Results for: "{query}"')
    plt.xlabel("Similarity Score")
    plt.ylabel("Document Index")
    plt.xlim(0, 1)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Positive"),
        Patch(facecolor="red", label="Negative"),
    ]
    plt.legend(handles=legend_elements)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
