import numpy as np
from .utils import TextPreprocessor


def get_doc_embedding(doc, en_embeddings):
    """Get embeddings for a single document

    Args:
        doc (str): Document text
        en_embeddings (dict): Word embeddings dictionary
    Returns:
        np.array: Documemnt embedding vector
    """
    processor = TextPreprocessor()
    doc_embedding = np.zeros(300)
    processed_doc = processor.clean_text(doc)
    if not processed_doc:
        return doc_embedding
    count = 0
    for word in processed_doc:
        if word in en_embeddings:
            doc_embedding += en_embeddings[word]
            count += 1
    if count > 0:
        doc_embedding = doc_embedding / count
    return doc_embedding


def get_corpus_embedding(documents, en_embeddings):
    """Get embeddings for a corpus of documents.

    Args:
        documents (list): List of document texts
        en_embeddings (dict): Word embedding dictionary

    Returns:
        tuple: (document_vectors, index_to_document_dict)
    """
    embedding_dim = 300
    document_vec = np.zeros((len(documents), embedding_dim))
    ind2doc_dict = {}
    for i, doc in enumerate(documents):
        embedding = get_doc_embedding(doc, en_embeddings)
        document_vec[i] = embedding
        ind2doc_dict[i] = doc

    return document_vec, ind2doc_dict


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors.

    Args:
        vec1 (np.array): First vector
        vec2 (np.array): Second vector
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


def search_documents(query, document_vectors, ind2doc_dict, en_embeddings, top_k=3):
    """Search for most relevant documents for a given query

    Args:
        query (str): Query text
        document_vectors (np.array): Document embedding matrix
        ind2doc_dict (dict): Index to document mapping
        en_embeddings (dict): Word embedding dictionary
        top_k (int, optional): Number of results. Defaults to 3.

    Returns:
        list: List of (doc_index, similarity_score) tuples
    """
    query_embedding = get_doc_embedding(query, en_embeddings)

    similarities = []
    for i, doc_vec in enumerate(document_vectors):
        sim = cosine_similarity(query_embedding, doc_vec)
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
