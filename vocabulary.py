import sys

import pandas as pd
from tqdm import tqdm

from configs import TRAIN_DATA_PATH, TRAIN_DATA_IDXS_PATH, VAL_DATA_PATH, VAL_DATA_IDXS_PATH, VOCAB_PATH, CONFIG
from utils import load_json, save_json


class Vocabulary(object):
    def __init__(self):
        self.token2idx = {"<K>": 0, "<unk>": 1}
        self.idx2token = {0: "<K>", 1: "<unk>"}
        self.idx = 2

    def add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = self.idx
            self.idx2token[self.idx] = token
            self.idx += 1

    def add_sequence(self, sequence):
        for token in sequence.split():
            self.add_token(token)

    def seq2ver(self, sequence, fix_length=32):
        idxs = [self.token2idx.get(token, self.token2idx["<unk>"]) for token in sequence.split()]
        if len(idxs) >= fix_length:
            idxs = idxs[:fix_length]
        else:
            idxs.extend([self.token2idx["<K>"]] * (fix_length - len(idxs)))
        return idxs

    def save_dict(self, dict_path):
        save_json(dict_path, self.token2idx)

    def load_dict(self, dict_path):
        self.token2idx = load_json(dict_path)
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.idx = len(self.token2idx)

    def __call__(self, word):
        return self.token2idx.get(word, self.token2idx["<unk>"])

    def __len__(self):
        return self.idx


def load_data(data_path):
    datasets = pd.read_csv(data_path)
    X = datasets["text"].tolist()
    y = datasets["label"].tolist()
    return X, y


def build_vocab():
    train_X, train_y = load_data(TRAIN_DATA_PATH)

    vocabulary_obj = Vocabulary()
    for text in tqdm(train_X, file=sys.stdout):
        vocabulary_obj.add_sequence(text)
    vocabulary_obj.save_dict(VOCAB_PATH)
    return vocabulary_obj


if __name__ == "__main__":
    vocabulary_obj = build_vocab()

    train_X, train_y = load_data(TRAIN_DATA_PATH)
    train_data_idxs = [
        {
            "text_idx": vocabulary_obj.seq2ver(text, fix_length=CONFIG["fix_length"]),
            "label": label,
        }
        for text, label in tqdm(zip(train_X, train_y), total=len(train_y), file=sys.stdout)
    ]
    save_json(TRAIN_DATA_IDXS_PATH, train_data_idxs)

    val_X, val_y = load_data(VAL_DATA_PATH)
    val_data_idxs = [
        {
            "text_idx": vocabulary_obj.seq2ver(text, fix_length=CONFIG["fix_length"]),
            "label": label,
        }
        for text, label in tqdm(zip(val_X, val_y), total=len(val_y), file=sys.stdout)
    ]
    save_json(VAL_DATA_IDXS_PATH, val_data_idxs)
