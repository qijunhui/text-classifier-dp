from collections import Counter

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from configs import TRAIN_DATA_IDXS_PATH, VAL_DATA_IDXS_PATH, VOCAB_PATH, LABELS_PATH, MODEL_PATH, DEVICE, CONFIG
from datasets import Datasets
from models.linear_model import Model
from train import train
from utils import load_json, load_pkl


def label_weight(data):
    labels = [datum["label"] for datum in data]
    counts = [count for label, count in sorted(Counter(labels).most_common(), key=lambda x: x[0])]
    labels_weight = dict(zip(load_pkl(LABELS_PATH).classes_, [round(count / sum(counts), 4) for count in counts]))
    print(f"label weight: {labels_weight}")
    return [count / sum(counts) for count in counts]


def train_run():
    train_data = load_json(TRAIN_DATA_IDXS_PATH)
    val_data = load_json(VAL_DATA_IDXS_PATH)

    train_datasets = Datasets(train_data)
    val_datasets = Datasets(val_data)

    train_loader = DataLoader(
        train_datasets,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_datasets,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = Model(
        vocab_size=len(load_json(VOCAB_PATH)),
        outputs_size=len(load_pkl(LABELS_PATH).classes_),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(label_weight(train_data)).to(DEVICE))

    train(train_loader, val_loader, model, optimizer, criterion)


def test_run():
    val_data = load_json(VAL_DATA_IDXS_PATH)

    val_datasets = Datasets(val_data)

    val_loader = DataLoader(
        val_datasets,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = (
        Model(
            vocab_size=len(load_json(VOCAB_PATH)),
            outputs_size=len(load_pkl(LABELS_PATH).classes_),
        )
        .to(DEVICE)
        .cpu()
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    val_acc_records = []
    y_true = []
    y_pred = []
    for idx, batch_data in enumerate(val_loader):
        inputs, targets = batch_data
        outputs = model(inputs)
        # print("真实值:", targets.reshape(-1).tolist())
        # print("预测值:", torch.argmax(outputs, dim=1).tolist())
        y_true.extend(targets.reshape(-1).tolist())
        y_pred.extend(torch.argmax(outputs, dim=1).tolist())
        val_acc_records.append(accuracy_score(targets.reshape(-1), torch.argmax(outputs, dim=1)))
    print(classification_report(y_true, y_pred, target_names=load_pkl(LABELS_PATH).classes_, digits=4, zero_division=0))
    val_acc = round(sum(val_acc_records) / len(val_acc_records), 4)
    print(f"[val] acc: {val_acc}")


if __name__ == "__main__":
    train_run()
    test_run()
