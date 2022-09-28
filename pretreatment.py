from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from configs import DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, LABELS_PATH
from utils import save_pkl, tokenizer


def label_encoder(y, labels_path):
    le = LabelEncoder()
    labels = le.fit_transform(y).tolist()
    save_pkl(labels_path, le)
    print("category in labels:", le.classes_)
    print("category distribution in labels:", Counter(y).most_common())
    return labels


def load_data():
    datasets = pd.read_csv(DATA_PATH)
    X = [tokenizer(text) for text in datasets["text"].tolist()]
    y = label_encoder(datasets["label"].tolist(), LABELS_PATH)
    return X, y


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = pd.DataFrame({"text": X_train, "label": y_train})
    test_data = pd.DataFrame({"text": X_test, "label": y_test})
    train_data.to_csv(TRAIN_DATA_PATH, index=False, encoding="utf-8")
    test_data.to_csv(VAL_DATA_PATH, index=False, encoding="utf-8")
