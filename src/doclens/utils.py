import os
import re
import string
import numpy as np
import pandas as pd
from typing import List
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.neighbors import NearestNeighbors


class TextPreprocessor:
    def __init__(self):
        self.tokenizer = TweetTokenizer(
            preserve_case=False, strip_handles=True, reduce_len=True
        )
        self.stopwords_en = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, treebank_tag):
        """Map Treebank POS tags to WordNet POS tags."""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN  # Default fallback

    def clean_text(self, text: str) -> List[str]:
        """Clean and lemmatize a single document."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"^rt[\s]+", "", text)
        text = re.sub(r"https?://[^\s\n\r]+", "", text)
        text = re.sub(r"#", "", text)

        tokens = self.tokenizer.tokenize(text)
        pos_tags = pos_tag(tokens)

        clean_tokens = []
        for token, tag in pos_tags:
            if token not in self.stopwords_en and token not in string.punctuation:
                lemma = self.lemmatizer.lemmatize(token, pos=self.get_wordnet_pos(tag))
                clean_tokens.append(lemma)
        return clean_tokens

    def preprocess_docs(
        self, docs: List[str], return_as_text: bool = False
    ) -> List[str | List[str]]:
        """Preprocess a list of documents. Return joined strings or token lists."""
        processed = []
        for doc in docs:
            tokens = self.clean_text(doc)
            processed.append(" ".join(tokens) if return_as_text else tokens)
        return processed


def load_glove_embeddings(glove_dir):
    en_embeddings = {}
    with open(glove_dir, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            en_embeddings[word] = vector
    return en_embeddings


def get_doc_embedding(doc, en_embeddings):
    """_summary_

    Args:
        doc (_type_): _description_
        en_embeddings (_type_): _description_
        process_doc (_type_, optional): _description_. Defaults to process_docs.
    """
    processor = TextPreprocessor()
    doc_embedding = np.zeros(300)
    processed_doc = processor.preprocess_docs(doc)
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
    processor = TextPreprocessor()
    for y, text in zip(yls, texts):
        tokens = processor.preprocess_docs(text)
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
    processor = TextPreprocessor()
    tokens_l = [processor.preprocess_docs(tweet) for tweet in train_x]
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


class LoadData:
    def __init__(self, data_dir="../../data/raw"):
        self.train_path = os.path.join(data_dir, "amazon_polarity_train_sample.csv")
        self.test_path = os.path.join(data_dir, "amazon_polarity_test_sample.csv")

        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            raise FileNotFoundError(
                "Both 'train.csv' and 'test.csv' must exist in the specified directory."
            )

    def getdata(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        X_train = train_df[["title", "content"]]
        y_train = train_df["label"]

        x_test = test_df[["title", "content"]]
        y_test = test_df["label"]
        return X_train, x_test, y_train, y_test

    def combine_text_features(self, df):
        df["title"] = df["title"].fillna("")
        df["content"] = df["content"].fillna("")
        return df["title"] + " " + df["content"]

    def load(self):
        X_train_raw, X_test_raw, y_train, y_test = self.getdata()
        X_train = self.combine_text_features(X_train_raw)
        X_test = self.combine_text_features(X_test_raw)
        return X_train, X_test, y_train, y_test
