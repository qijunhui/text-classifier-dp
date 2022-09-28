import torch
from torch.utils.data import Dataset

from utils import load_json


class Datasets(Dataset):
    def __init__(self, data, is_expand=1):
        self.data = data * is_expand

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.data[idx]["text_idx"]),
            torch.LongTensor([self.data[idx]["label"]]),
        )


if __name__ == "__main__":
    from configs import TRAIN_DATA_IDXS_PATH

    train_data = load_json(TRAIN_DATA_IDXS_PATH)

    train_datasets = Datasets(train_data)
    print("训练数据量:", len(train_datasets))
    for idx, datum in enumerate(train_datasets):
        if idx >= 1:
            break
        print(datum[0].shape, datum[1].shape)
        print(datum)
