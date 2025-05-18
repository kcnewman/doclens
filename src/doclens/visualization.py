import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os


def setup_visualization_dir(vis_dir="results/visualizations"):
    """Setup visualization directory with absolute path"""
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    abs_vis_dir = os.path.join(root_dir, vis_dir)
    os.makedirs(abs_vis_dir, exist_ok=True)
    return abs_vis_dir


def reduce_dimensions(embeddings, n_components=None, variance_threshold=0.8):
    """Reduce dimensions using PCA while preserving specified variance."""
    embeddings_normalized = (embeddings - np.mean(embeddings, axis=0)) / np.std(
        embeddings, axis=0
    )
    pca_init = PCA(n_components=min(embeddings.shape[0], embeddings.shape[1]))
    pca_init.fit(embeddings_normalized)
    cumsum = np.cumsum(pca_init.explained_variance_ratio_)
    n_components = (
        np.argmax(cumsum >= variance_threshold) + 1
        if np.any(cumsum >= variance_threshold)
        else embeddings.shape[1]
    )
    variance_2d = pca_init.explained_variance_ratio_[:2].sum() * 100

    print(
        f"Using {n_components} components to preserve {variance_threshold * 100:.1f}% variance"
    )
    print(f"First 2 components explain {variance_2d:.1f}% of total variance")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_normalized)

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

    plt.title("Document Clusters (First 2 PCA Components)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
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
    """Visualize search results with sentiments."""
    plt.figure(figsize=(10, 6))

    if not results:
        plt.text(0.5, 0.5, "No results found", ha="center", va="center")
        plt.title(f'Search Results for: "{query}"')
    else:
        indices = [str(r[0]) for r in results]
        similarities = [r[1] for r in results]
        colors = ["red" if s == 0 else "green" for s in sentiments]
        y_pos = np.arange(len(indices))
        plt.barh(y_pos, similarities, color=colors)
        plt.yticks(y_pos, indices)
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
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()
