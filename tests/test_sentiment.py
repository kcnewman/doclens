import pytest
import numpy as np
from doclens.sentiment import train_sentiment_model, predict_sentiment


def test_sentiment_training():
    docs = ["This is great!", "This is terrible.", "I love this.", "I hate this."]
    labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

    model, vectorizer = train_sentiment_model(docs, labels)
    assert model is not None
    assert vectorizer is not None


def test_sentiment_prediction():
    # Training data
    train_docs = ["I love this", "I hate this"]
    train_labels = [1, 0]

    # Test data
    test_docs = ["This is wonderful"]

    model, vectorizer = train_sentiment_model(train_docs, train_labels)
    predictions, probabilities = predict_sentiment(model, vectorizer, test_docs)

    assert len(predictions) == 1
    assert probabilities.shape == (1, 2)
    assert predictions[0] in [0, 1]
