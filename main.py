import os
import argparse
import numpy as np
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
    X_train,X_test,y_train,y_test = data_loader.load()
    return X_train,X_test,y_train,y_test

def load_embeddings(glove_path):
    print(f"Loading GloVe embeddings from {glove_path}...")
    return load_glove_embeddings(glove_path)

def train_sentiment_model(X_train, y_train):
    print("Training sentiment model...")
    sentiment_model = SentimentPredictor()
    sentiment_model.train(X_train,y_train)
    
    model_path = os.path.join("./results/models", "sentiment_model.joblib")
    vectorizer_path = os.path.join("./results/models", "tfidf_vectorizer.joblib")
    sentiment_model.save_model(model_path,vectorizer_path)
    print(f"Sentiment model saved to {model_path}")
    
    return sentiment_model

def process_search(query, document_vectors,ind2doc_dict,en_embeddings)