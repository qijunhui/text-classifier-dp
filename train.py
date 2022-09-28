import os
import sys

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from configs import OUTPUTS_DIR, MODEL_PATH, IS_COVER, DEVICE, CONFIG


def train_epoch(train_loader, model, optimizer, criterion, epoch):
    model.train()
    train_acc_records = []
    train_loss_records = []
    for idx, batch_data in enumerate(tqdm(train_loader, file=sys.stdout)):
        inputs, targets = batch_data

        outputs = model(inputs.to(DEVICE))
        loss = criterion(outputs, targets.reshape(-1).to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc_records.append(accuracy_score(targets.reshape(-1), torch.argmax(outputs, dim=1).cpu()))
        train_loss_records.append(loss.item())

    train_acc = round(sum(train_acc_records) / len(train_acc_records), 4)
    train_loss = round(sum(train_loss_records) / len(train_loss_records), 4)
    print(f"[train] Epoch: {epoch} / {CONFIG['epoch']}, acc: {train_acc}, loss: {train_loss}")
    return train_acc, train_loss


def evaluate(val_loader, model, epoch):
    model.eval()
    val_acc_records = []
    for idx, batch_data in enumerate(val_loader):
        inputs, targets = batch_data

        outputs = model(inputs.to(DEVICE))

        val_acc_records.append(accuracy_score(targets.reshape(-1), torch.argmax(outputs, dim=1).cpu()))

    val_acc = round(sum(val_acc_records) / len(val_acc_records), 4)
    print(f"[val]   Epoch: {epoch} / {CONFIG['epoch']}, acc: {val_acc}")
    return val_acc


def train(train_loader, val_loader, model, optimizer, criterion):
    best_val_acc = 0
    patience_counter = 0
    for epoch in range(1, CONFIG["epoch"] + 1):
        train_acc, train_loss = train_epoch(train_loader, model, optimizer, criterion, epoch)
        val_acc = evaluate(val_loader, model, epoch)

        if (val_acc - best_val_acc) > CONFIG["patience"]:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                MODEL_PATH
                if IS_COVER
                else os.path.join(OUTPUTS_DIR, f"{epoch}-train_acc{train_acc}-val_acc{val_acc}-model.pkl"),
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if (patience_counter >= CONFIG["patience_num"] and epoch > CONFIG["min_epoch"]) or epoch == CONFIG["epoch"]:
            print(f"best val acc: {best_val_acc}, training finished!")
            break
