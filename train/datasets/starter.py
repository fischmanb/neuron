from time import time

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import re

from preprocessing import projectroot
from google.cloud import storage
import json
import io
import numpy as np
import librosa

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
                    drop_last=False, shuffle=False)

    for batch in dl:
        # print(f"{batch['labels'].squeeze()}")
        pass


if __name__ == '__main__':
    sample_ds()