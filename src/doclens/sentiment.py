import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from .utils import TextPreprocessor


def build_freqs(texts, ys):
    """Build frequency tables
    Args:
        text: list of document texts
        ys: List of sentiment labels

    Output:
        freqs (dict): Dictionary mapping each (word, sentiment) pair to its frequency
    """
    yls = list(np.squeeze(ys).tolist()) if not isinstance(ys, list) else ys
    freqs = {}
    processor = TextPreprocessor()
    for y, text in zip(yls, texts):
        tokens = processor.clean_text(text)
        for word in tokens:
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs


def extract_features(texts, freqs):
    """
    Extract features from text using frequency dictionary.

    Args:
        texts (list: List of document texts
        freqs (dict): Dictionary of word frequencies.

    Returns:
        X (np.ndarray): Feature matrix
    """
    processor = TextPreprocessor()
    tokens_l = [processor.clean_text(text) for text in texts]
    vocab = sorted(set([word for word, _ in freqs.keys()]))
    word_idx = {word: i for i, word in enumerate(vocab)}
    n_samples = len(tokens_l)
    n_features = len(vocab)
    X = np.zeros((n_samples, n_features))
    for i, tokens in enumerate(tokens_l):
        for word in tokens:
            if word in word_idx:
                X[i, word_idx[word]] += 1
    return X


class SentimentPredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=5000)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, x_train, y_train):
        """Train sentiment prediction model

        Args:
            x_train (list): List of document text
            y_train (list): List of sentiment label
        """
        X_tfidf = self.vectorizer.fit_transform(x_train)

        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_tfidf, y_train)

    def predict(self, texts):
        """Predict sentiment for given texts.

        Args:
            texts (list): List of document text

        Return:
            texts (list): Predicted sentiment label
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict(X_tfidf)

    def predict_prob(self, texts):
        """Predict sentiment probabilities

        Args:
            texts (list): List of document texts

        Returns:
            np.array: Predicted sentiment probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict_proba(X_tfidf)

    def save_model(self, model_path, vectorizer_path=None):
        """Save trained model to disk

        Args:
            model_path (str): Path to save model
            vectorizer_path (str, optional): Path to save vectorizer. Defaults to None.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        if vectorizer_path:
            joblib.dump(self.vectorizer, vectorizer_path)

    def load_model(self, model_path, vectorizer_path=None):
        """Load trained model

        Args:
            model_path (str): Path to saved model
            vectorizer_path (str, optional): Path to saved vectorizer. Defaults to None.
        """
        self.model = joblib.load(model_path)

        if vectorizer_path and os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
