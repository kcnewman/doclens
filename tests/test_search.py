import pytest
from doclens.search import create_tfidf_matrix, find_top_k_similar_docs


def test_tfidf_creation():
    docs = ["This is document one", "This is document two", "This is document three"]
    vectorizer, matrix = create_tfidf_matrix(docs)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == len(vectorizer.get_feature_names_out())


def test_similarity_search():
    docs = [
        "Python is a programming language",
        "Java is also a programming language",
        "Dogs are pets",
    ]
    vectorizer, matrix = create_tfidf_matrix(docs)
    query = "What programming languages exist?"
    indices, similarities = find_top_k_similar_docs(query, matrix, vectorizer, k=2)

    assert len(indices) == 2
    assert len(similarities) == 2
    # First two docs should be more similar than the third
    assert all(i in [0, 1] for i in indices)
