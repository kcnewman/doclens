import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.neighbors import NearestNeighbors


def process_docs(doc):
    """_summary_

    Args:
        doc (_type_): _description_
    """
    stemmer = PorterStemmer()
    stopwords_en = stopwords.words("english")
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    doc = doc.lower()
    doc = re.sub(r"[^\w\s]", "", doc)
    doc = re.sub(r"^RT[\s]+", "", doc)
    doc = re.sub(r"https?://[^\s\n\r]+", "", doc)
    doc = re.sub(r"#", "", doc)

    doc_tokens = tokenizer.tokenize(doc)
    docs_clean = []
    for doc in doc_tokens:
        if doc not in stopwords_en and doc not in string.punctuation:
            stem_doc = stemmer.stem(doc)
            docs_clean.append(stem_doc)

    return docs_clean


def load_glove_embeddings(glove_dir):
    en_embeddings = {}
    with open(glove_dir, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            en_embeddings[word] = vector
    return en_embeddings


def get_doc_embedding(doc, en_embeddings, process_doc=process_docs):
    """_summary_

    Args:
        doc (_type_): _description_
        en_embeddings (_type_): _description_
        process_doc (_type_, optional): _description_. Defaults to process_docs.
    """
    doc_embedding = np.zeros(300)
    processed_doc = process_doc(doc)
    if not processed_doc:
        return doc_embedding

    for word in processed_doc:
        doc_embedding += en_embeddings.get(word, np.zeros(300))
        doc_embedding = doc_embedding / max(len(processed_doc), 1)
    return doc_embedding


def get_corpus_embedding(documents, en_embeddings, embedding_func=get_doc_embedding):
    """_summary_

    Args:
        documents (_type_): _description_
        en_embeddings (_type_): _description_
        embedding_func (_type_, optional): _description_. Defaults to get_doc_embedding.
    """
    embedding_dim = 300
    document_vec = np.zeros((len(documents), embedding_dim))
    ind2doc_dict = {}
    for i, doc in enumerate(documents):
        embedding = embedding_func(doc, en_embeddings)
        document_vec[i] = embedding
        ind2doc_dict[i] = embedding

        return document_vec, ind2doc_dict


def knn_index(embeddings, n_neighbors=5):
    """_summary_

    Args:
        embeddings (_type_): _description_
        n_neighbors (int, optional): _description_. Defaults to 5.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine")
    knn.fit(embeddings)
    return knn


def build_freqs(texts, ys):
    """Build frequency tables
    Args:
        text: A list of str
        ys: An array matching sentiment
    Output:
        freqs (dict): Dictionary mapping each word(word, sentiment) pair to its frequency
    """
    yls = list(np.squeeze(ys).tolist()) if not isinstance(ys, list) else ys
    freqs = {}
    for y, text in zip(yls, texts):
        tokens = process_docs(text)
        for word in tokens:
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs


def extract_features(train_x, freqs):
    """
    Extract features from an array of texts.

    Args:
        train_x (list or np.array): a list or array of texts (tweets).
        freqs (dict): a dictionary mapping (word, sentiment) pairs to their frequency counts.

    Returns:
        X (np.ndarray): a 2D numpy array with shape (number_of_texts, vocabulary_size)
    """
    tokens_l = [process_docs(tweet) for tweet in train_x]
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

# from .search import create_tfidf_matrix, find_top_k_similar_docs
# from .clustering import cluster_documents, reduce_dimensions
# from .sentiment import train_sentiment_model, predict_sentiment
# from .visualization import reduce_and_plot

# def preprocess_texts(texts):
#     # Tokenization, lowercasing, stopword removal, etc.
#     pass

# class DocumentAnalyzer:
#     def __init__(self):
#         self.vectorizer = None
#         self.tfidf_matrix = None
#         self.sentiment_model = None
#         self.sentiment_vectorizer = None
#         self.kmeans = None
#         self.cluster_labels = None

#     def fit(self, documents, sentiment_labels=None):
#         """Fit the analyzer with documents and optionally sentiment labels"""
#         # Create TF-IDF matrix
#         self.vectorizer, self.tfidf_matrix = create_tfidf_matrix(documents)

#         # Train sentiment model if labels provided
#         if sentiment_labels is not None:
#             self.sentiment_model, self.sentiment_vectorizer = train_sentiment_model(
#                 documents, sentiment_labels
#             )

#         # Cluster documents
#         self.kmeans, self.cluster_labels = cluster_documents(self.tfidf_matrix)

#     def analyze_query(self, query, k=3):
#         """Analyze a query: find similar docs, predict sentiment, visualize"""
#         # Find similar documents
#         similar_indices, similarities = find_top_k_similar_docs(
#             query,
#             self.tfidf_matrix,
#             self.vectorizer,
#             k
#         )

#         results = {
#             'indices': similar_indices,
#             'similarities': similarities,
#         }

#         # Predict sentiment if model exists
#         if self.sentiment_model is not None:
#             sentiments, probs = predict_sentiment(
#                 self.sentiment_model,
#                 self.sentiment_vectorizer,
#                 [query]
#             )
#             results['sentiment'] = sentiments[0]
#             results['sentiment_proba'] = probs[0]

#         return results

#     def visualize(self):
#         """Visualize document clusters"""
#         return reduce_and_plot(self.tfidf_matrix, self.cluster_labels)
