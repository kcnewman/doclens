import numpy as np
from .utils import get_doc_embedding, knn_index
from sklearn.metrics.pairwise import cosine_similarity

def cosine_search(query, documents, doc_embeddings, en_embeddings, top_n=3):
    """_summary_

    Args:
        query (_type_): _description_
        documents (_type_): _description_
        doc_embeddings (_type_): _description_
        en_embeddings (_type_): _description_
        top_n (int, optional): _description_. Defaults to 3.
    """
    query_embedding = get_doc_embedding(query,en_embeddings).reshape(1,-1)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_idx = similarities.argsort()[-top_n:][::-1]
    top_scores = similarities[top_idx]
    return top_idx,top_scores

def knn_search(query_embedding,knn_index=knn_index,n_results=3):
    """_summary_

    Args:
        query_embedding (_type_): _description_
        knn_index (_type_): _description_
        n_results (int, optional): _description_. Defaults to 3.
    """
    distance, idx = knn_index.kneighbors(query_embedding.reshape(1,-1),n_neighbors=n_results)
    
    return distance[0], idx[0]

    
    


