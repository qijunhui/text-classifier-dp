import json

import dill
import jieba


def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False)
    print(f"[{filepath}] data saving...")


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as fr:
        data = json.load(fp=fr)
    print(f"[{filepath}] data loading...")
    return data


def save_pkl(filepath, data):
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):
    with open(filepath, "rb") as fr:
        data = dill.load(fr)
    print(f"[{filepath}] data loading...")
    return data


def tokenizer(text):
    return " ".join(jieba.lcut(text))
