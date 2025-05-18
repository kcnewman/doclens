import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from doclens.search import search_documents


@pytest.fixture
def sample_embeddings():
    """Create sample word embeddings for testing"""
    vocab = [
        "python",
        "programming",
        "language",
        "java",
        "dogs",
        "pets",
        "are",
        "is",
        "a",
        "also",
    ]
    return {word: np.random.rand(300) for word in vocab}


def create_tfidf_matrix(docs):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


def test_tfidf_creation():
    docs = ["This is document one", "This is document two", "This is document three"]
    vectorizer, matrix = create_tfidf_matrix(docs)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == len(vectorizer.get_feature_names_out())


def test_similarity_search(sample_embeddings):
    docs = [
        "Python is a programming language",
        "Java is also a programming language",
        "Dogs are pets",
    ]
    ind2doc_dict = {i: doc for i, doc in enumerate(docs)}

    doc_vectors = np.array(
        [
            np.mean(
                [
                    sample_embeddings.get(word.lower(), np.zeros(300))
                    for word in doc.split()
                ],
                axis=0,
            )
            for doc in docs
        ]
    )

    query = "What programming languages exist?"
    results = search_documents(query, doc_vectors, ind2doc_dict, sample_embeddings)

    assert len(results) <= 5
    assert all(isinstance(idx, int) and isinstance(sim, float) for idx, sim in results)
    assert all(0 <= sim <= 1 for _, sim in results)
    assert all(idx in [0, 1] for idx, _ in results[:2])
