import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from doclens.search import search_documents


def create_tfidf_matrix(docs):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


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
    ind2doc_dict = {i: doc for i, doc in enumerate(docs)}
    en_embeddings = {}
    results = search_documents(query, matrix, ind2doc_dict, en_embeddings, top_k=2)
    indices, similarities = zip(*results)

    assert len(indices) == 2
    assert len(similarities) == 2
    assert all(i in [0, 1] for i in indices)
