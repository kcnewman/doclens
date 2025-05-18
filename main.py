import os
import argparse
import pandas as pd
import warnings
import logging
from contextlib import suppress

from doclens.utils import LoadData, load_glove_embeddings
from doclens.search import get_corpus_embedding, search_documents
from doclens.clustering import cluster_documents, knn_index
from doclens.sentiment import SentimentPredictor
from doclens.visualization import (
    setup_visualization_dir,
    reduce_dimensions,
    plot_document_clusters,
    plot_sentiment_distribution,
    visualize_search_results,
)


def configure_cpu():
    """Configure CPU cores and suppress related warnings"""
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("loky").setLevel(logging.ERROR)


configure_cpu()


def setup_directories():
    vis_dir = setup_visualization_dir()
    os.makedirs("./results/models", exist_ok=True)
    return vis_dir


def prepare_data():
    print("Loading data...")
    data_loader = LoadData()
    X_train, X_test, y_train, y_test = data_loader.load()
    return X_train, X_test, y_train, y_test


def load_embeddings(glove_path):
    print(f"Loading GloVe embeddings from {glove_path}...")
    return load_glove_embeddings(glove_path)


def train_sentiment_model(X_train, y_train):
    print("Training sentiment model...")
    sentiment_model = SentimentPredictor()
    sentiment_model.train(X_train, y_train)

    model_path = os.path.join("./results/models", "sentiment_model.joblib")
    vectorizer_path = os.path.join("./results/models", "tfidf_vectorizer.joblib")
    sentiment_model.save_model(model_path, vectorizer_path)
    print(f"Sentiment model saved to {model_path}")

    return sentiment_model


def process_search_query(
    query,
    document_vectors,
    ind2doc_dict,
    en_embeddings,
    sentiment_model,
    corpus,
    vis_dir,
):
    print(f"\nSearching for: '{query}'")
    results = search_documents(query, document_vectors, ind2doc_dict, en_embeddings)
    valid_results = [(idx, sim) for idx, sim in results if idx in corpus.index]

    if not valid_results:
        print("No matching documents found.")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(vis_dir, f"search_results_{timestamp}.png")
        visualize_search_results(query, [], [], save_path)
        return [], []

    result_indices = [idx for idx, _ in valid_results]
    result_docs = [str(corpus[idx]) for idx in result_indices]
    result_sentiments = sentiment_model.predict(result_docs)

    print("\nSearch Results:")
    print("-" * 60)
    for i, (idx, sim) in enumerate(valid_results):
        sentiment = "Positive" if result_sentiments[i] == 1 else "Negative"
        print(f"Document {idx} (Similarity: {sim:.4f}, Sentiment: {sentiment})")
        doc_preview = (
            corpus[idx][:100] + "..." if len(corpus[idx]) > 100 else corpus[idx]
        )
        print(f"Preview: {doc_preview}")
        print("-" * 60)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(vis_dir, f"search_results_{timestamp}.png")
    visualize_search_results(query, valid_results, result_sentiments, save_path)
    print(f"Search results visualization saved to {save_path}")

    return result_indices, result_sentiments


def main():
    parser = argparse.ArgumentParser(
        description="DocLens: Document Search + Clustering + Sentiment Prediction"
    )
    parser.add_argument(
        "--glove",
        type=str,
        default="data/glove/glove.6B.300d.txt",
        help="Path to GloVe embeddings file",
    )
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    args = parser.parse_args()

    vis_dir = setup_directories()
    X_train, X_test, y_train, y_test = prepare_data()
    corpus = pd.concat([X_train, X_test])
    corpus_labels = pd.concat([y_train, y_test])
    en_embeddings = load_embeddings(args.glove)

    print("Computing document embeddings...")
    document_vectors, ind2doc_dict = get_corpus_embedding(corpus, en_embeddings)

    sentiment_model = train_sentiment_model(X_train, y_train)

    print("Creating KNN index...")
    knn = knn_index(document_vectors)

    print(f"Clustering documents into {args.clusters} clusters...")
    kmeans, cluster_labels = cluster_documents(
        document_vectors, n_clusters=args.clusters
    )

    print("Reducing dimensions with PCA...")
    reduced_embeddings, pca = reduce_dimensions(
        document_vectors, variance_threshold=0.8
    )
    print("\nGenerating initial visualizations...")
    cluster_vis_path = os.path.join(vis_dir, "document_clusters.png")
    if not os.path.exists(cluster_vis_path):
        plot_document_clusters(reduced_embeddings, cluster_labels, cluster_vis_path)
        print(f"Document clusters visualization saved to {cluster_vis_path}")
    sentiment_vis_path = os.path.join(vis_dir, "sentiment_distribution.png")
    if not os.path.exists(sentiment_vis_path):
        plot_sentiment_distribution(corpus_labels, sentiment_vis_path)
        print(f"Sentiment distribution visualization saved to {sentiment_vis_path}")

    print("\nDocument analysis system ready!")
    print("Enter your search queries below (type 'quit' to exit)")
    print("-" * 60)

    # Interactive search loop
    while True:
        query = input("\nEnter search query: ")
        if query.lower() == "quit":
            break
        process_search_query(
            query,
            document_vectors,
            ind2doc_dict,
            en_embeddings,
            sentiment_model,
            corpus,
            vis_dir,
        )


if __name__ == "__main__":
    main()
