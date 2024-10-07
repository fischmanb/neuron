import pandas as pd
import torch

from preprocessing import preprocessingroot
from train import trainroot


def mfcc_eda():
    tfile = torch.load(preprocessingroot / "audio_segments" / "mfcc" / "1ef79f10-1a02-6f12-9dcd-b85a23fed4bb.pt", weights_only=True)
    a=1


def labels_eda():
    filepath = trainroot / 'datasets' / 'filtered_combined_labels_09-28.pkl.xz'
    df = pd.read_pickle(filepath, compression='xz')
    a=1


if __name__ == '__main__':
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 900)
    labels_eda()  # this is just used to test locally
    # mfcc_eda()
