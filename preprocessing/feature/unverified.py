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


class RandomCut(nn.Module):
    """Augmentation technique that randomly cuts start or end of audio"""

    def __init__(self, max_cut=10):
        super(RandomCut, self).__init__()
        self.max_cut = max_cut

    def forward(self, x):
        """Randomly cuts from start or end of batch"""
        side = torch.randint(0, 1, (1,))
        cut = torch.randint(1, self.max_cut, (1,))
        if side == 0:
            return x[:-cut, :, :]
        elif side == 1:
            return x[cut:, :, :]


class SpecAugment(nn.Module):
    """Augmentation technique to add masking on the time or frequency domain"""

    def __init__(self, rate, policy=3, freq_mask=2, time_mask=4):
        super(SpecAugment, self).__init__()

        self.rate = rate

        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        policies = {1: self.policy1, 2: self.policy2, 3: self.policy3}
        self._forward = policies[policy]

    def forward(self, x):
        return self._forward(x)

    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug(x)
        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug2(x)
        return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)


