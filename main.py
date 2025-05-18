import os
import argparse
import pandas as pd
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
    result_indices = [idx for idx, _ in results]
    result_docs = [corpus[idx] for idx in result_indices]

    result_sentiments = sentiment_model.predict(result_docs)

    print("\nSearch Results:")
    print("-" * 60)
    for i, (idx, sim) in enumerate(results):
        sentiment = "Positive" if result_sentiments[i] == 1 else "Negative"
        print(f"Document {idx} (Similarity: {sim:.4f}, Sentiment: {sentiment})")

        doc_preview = (
            corpus[idx][:100] + "..." if len(corpus[idx]) > 100 else corpus[idx]
        )
        print(f"Preview: {doc_preview}")
        print("-" * 60)

    save_path = os.path.join(vis_dir, "search_results.png")
    visualize_search_results(query, results, result_sentiments, save_path)
    print(f"Search results visualization saved to {save_path}")

    return result_indices, result_sentiments


def main():
    parser = argparse.ArgumentParser(
        description="DocLens: Document Search + Clustering + Sentiment Prediction"
    )
    parser.add_argument(
        "--glove",
        type=str,
        default="glove/glove.6B.300d.txt",
        help="Path to GloVe embeddings file",
    )
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument(
        "--query", type=str, default="product review", help="Initial search query"
    )
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
    reduced_embeddings, pca = reduce_dimensions(document_vectors)
    print(f"Explained variance: {sum(pca.explained_variance_ratio_):.2f}")

    cluster_vis_path = os.path.join(vis_dir, "document_clusters.png")
    plot_document_clusters(reduced_embeddings, cluster_labels, cluster_vis_path)
    print(f"Document clusters visualization saved to {cluster_vis_path}")

    sentiment_vis_path = os.path.join(vis_dir, "sentiment_distribution.png")
    plot_sentiment_distribution(corpus_labels, sentiment_vis_path)
    print(f"Sentiment distribution visualization saved to {sentiment_vis_path}")

    process_search_query(
        args.query,
        document_vectors,
        ind2doc_dict,
        en_embeddings,
        sentiment_model,
        corpus,
        vis_dir,
    )

    while True:
        query = input("\nEnter search query (or 'quit' to exit): ")
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
