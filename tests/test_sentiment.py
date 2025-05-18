import pytest
import numpy as np
from doclens.sentiment import SentimentPredictor
import os
import tempfile


@pytest.fixture
def sample_data():
    """Sample training data for sentiment analysis"""
    X_train = [
        "This product is excellent!",
        "Terrible experience, do not buy",
        "I love this product",
        "Waste of money",
    ]
    y_train = [1, 0, 1, 0]  # 1 for positive, 0 for negative
    return X_train, y_train


@pytest.fixture
def trained_model(sample_data):
    """Create a trained sentiment model"""
    X_train, y_train = sample_data
    model = SentimentPredictor()
    model.train(X_train, y_train)
    return model


def test_sentiment_training(sample_data):
    X_train, y_train = sample_data
    model = SentimentPredictor()
    model.train(X_train, y_train)

    # Test prediction on training data
    predictions = model.predict(X_train)
    assert len(predictions) == len(y_train)
    assert all(pred in [0, 1] for pred in predictions)


def test_sentiment_prediction(trained_model):
    test_texts = [
        "This is wonderful",
        "This is horrible",
    ]
    predictions = trained_model.predict(test_texts)

    assert len(predictions) == 2
    assert predictions[0] == 1  # Should predict positive
    assert predictions[1] == 0  # Should predict negative


def test_model_save_load(trained_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        vectorizer_path = os.path.join(tmpdir, "vectorizer.joblib")

        # Save model
        trained_model.save_model(model_path, vectorizer_path)

        # Load model
        new_model = SentimentPredictor()
        new_model.load_model(model_path, vectorizer_path)

        # Test predictions match
        test_text = ["This is a test"]
        assert trained_model.predict(test_text) == new_model.predict(test_text)
