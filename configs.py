import os

import torch


def makedir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)


# 根目录
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# 数据
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "takeaway_comment_8k.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.csv")
TRAIN_DATA_IDXS_PATH = os.path.join(DATA_DIR, "train_data_idxs.json")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val_data.csv")
VAL_DATA_IDXS_PATH = os.path.join(DATA_DIR, "val_data_idxs.json")

# 模型
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
VOCAB_PATH = os.path.join(OUTPUTS_DIR, "vocab.json")
LABELS_PATH = os.path.join(OUTPUTS_DIR, "labels.pkl")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "model.pkl")

# 模型覆盖
IS_COVER = True

# 优先使用gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型相关配置
CONFIG = {
    "fix_length": 32,
    "batch_size": 128,
    "lr": 0.003,
    "epoch": 100,
    "min_epoch": 5,
    "patience": 0.0002,
    "patience_num": 10,
}

# makedir
makedir(OUTPUTS_DIR)
