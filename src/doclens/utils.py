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
        return wordnet.NOUN

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
    """Load GloVe embeddings from file

    Args:
        glove_dir (path): Path to GloVe file

    Returns:
        en_embeddings: A dictionary mapping words to vector representations
    """
    en_embeddings = {}
    with open(glove_dir, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            en_embeddings[word] = vector
    return en_embeddings


class LoadData:
    def __init__(self, data_dir="../../data/raw"):
        self.train_path = os.path.join(data_dir, "amazon_polarity_train_sample.csv")
        self.test_path = os.path.join(data_dir, "amazon_polarity_test_sample.csv")

        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            raise FileNotFoundError(
                "Both 'train and test files must exist in {data_dir}"
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
