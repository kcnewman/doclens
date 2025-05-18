import pandas as pd
from utils import build_freqs, extract_features
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("../../data/raw/amazon_polarity_train_sample.csv")
train_x = data["content"]
freqs = build_freqs(train_x, train_y)

X_train = extract_features(train_x, freqs)
model = LogisticRegression(max_iter=1000)
