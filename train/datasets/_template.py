from time import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SampleData(Dataset):
    def __init__(self):
        super(SampleData).__init__()
        self.data = ""

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        pass


def sample_ds():
    if torch.cuda.is_available():
        num_workers = 16
    else:
        num_workers = 0
    ds = SampleData()
    dl = DataLoader(ds, batch_size=10,
                    num_workers=num_workers, pin_memory=False,
                    drop_last=False, shuffle=True)

    for batch in dl:
        # print(f"{batch['labels'].squeeze()}")
        pass


if __name__ == '__main__':
    start = time()
    sample_ds()
    print(f"Time {(time() -  start) / 60} minutes")
