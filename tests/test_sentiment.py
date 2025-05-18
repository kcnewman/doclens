import pytest
from doclens.sentiment import SentimentPredictor


@pytest.fixture
def trained_model():
    """Create and train a sentiment model with sample data"""
    X_train = [
        # Strong positive examples
        "This product is excellent and amazing!",
        "I love this product, it's wonderful!",
        "Outstanding quality and fantastic service",
        "Perfect experience, highly recommended",
        "Brilliant product, exceeded expectations",
        # Mixed but positive examples
        "Good product despite minor issues",
        "Generally positive experience with some drawbacks",
        "Mostly satisfied with the purchase",
        # Strong negative examples
        "Terrible experience, do not buy",
        "Worst purchase ever, complete waste",
        "Absolutely horrible product, avoid",
        "Poor quality, disappointing results",
        "Dreadful service, complete disaster",
        # Mixed but negative examples
        "Has some good features but overall disappointing",
        "Not worth it despite few positives",
        "Could be better, many issues outweigh benefits",
    ]

    # 1 for positive (first 8 examples), 0 for negative (last 8 examples)
    y_train = [1] * 8 + [0] * 8

    model = SentimentPredictor()
    model.train(X_train, y_train)
    return model


def test_sentiment_prediction(trained_model):
    """Test basic sentiment prediction"""
    test_pairs = [
        ("This is absolutely wonderful and amazing", 1),  # Strong positive
        ("This is horrible and terrible", 0),  # Strong negative
        ("The product works as expected", 1),  # Mild positive
        ("Would not recommend this product", 0),  # Clear negative
    ]

    for text, expected in test_pairs:
        prediction = trained_model.predict([text])[0]
        assert prediction == expected, f"Failed on: {text}"


def test_sentiment_prediction_edge_cases(trained_model):
    """Test sentiment prediction on edge cases"""
    edge_cases = [
        ("Good but has some issues", 1),  # Mixed but mostly positive
        ("Bad product despite some good features", 0),  # Mixed but mostly negative
        ("Not bad at all", 1),  # Negation of negative
        ("Not particularly good", 0),  # Negation of positive
        ("Just okay", 1),  # Neutral but slightly positive
        ("Could be better", 0),  # Mild negative
    ]

    for text, expected in edge_cases:
        prediction = trained_model.predict([text])[0]
        assert prediction == expected, f"Failed on: {text}"
