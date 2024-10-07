"""
Data and annotations for patient diagnosis dataset for train/val
"""
import os
import pickle
from pathlib import Path
from time import time
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing import preprocessingroot
from train import trainroot


class DiagClip(Dataset):
    """First Dataset, minimum length 1sec"""
    def __init__(self, split="val", num_classes=1):
        """
        File columns: 'timestamp', 'text', 'speaker', 'voice', 'samplerate', 'original-file', 'creation-date',
        'full-audio-duration(seconds)', 'patient_id', 'diagnosis', 'GUID', 'common_disorders'

        :param split: 'train' or 'val'
        :param num_classes: num_classes
        """
        super(DiagClip).__init__()
        self.phases = ["train", "val"]
        self.split = split
        self.guids = []

        if not os.path.exists(trainroot / 'datasets' / 'trainval.pkl'):
            self.cache_trainval()

        with open(trainroot / 'datasets' / 'trainval.pkl', 'rb') as f:
            self.guids = pickle.load(f)[split]

        # self.idx_to_label = {0: 'Trauma- and Stressor-Related Disorders', 1: 'Bipolar Disorders', 2: 'Obsessive-Compulsive and Related Disorders', 3: 'Depressive Disorders', 4: 'Schizophrenia Spectrum and Other Psychotic Disorders', 5: 'Substance-Related and Addictive Disorders', 6: 'Anxiety Disorders', 7: 'Neurodevelopmental Disorders', 8: 'Neurocognitive Disorders'}
        self.idx_to_label = {0: 'Anxiety Disorders', 1: 'Trauma- and Stressor-Related Disorders', 2: 'Substance-Related and Addictive Disorders', 3: 'Neurocognitive Disorders', 4: 'ADHD', 5: 'Depressive Disorders', 6: 'Bipolar Disorders', 7: 'Schizophrenia Spectrum and Other Psychotic Disorders', 8: 'Obsessive-Compulsive and Related Disorders'}
        self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}
        self.classlabels = list(self.idx_to_label.values())
        self.num_classes = num_classes

        full_df = pd.read_pickle(trainroot / 'datasets' / 'filtered_combined_labels_09-28.pkl.xz', compression='xz')
        diagnoses = set().union(*full_df['common_disorders'])  # elements of the lists of diagnoses broken down and unique
        assert diagnoses == set(self.classlabels)
        full_df['GUID'] = full_df['GUID'].astype(str)
        self.df = full_df[full_df['GUID'].isin(self.guids)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        diagnoses = self.df.iloc[idx]['common_disorders']
        diagnoses = [self.label_to_idx[x] for x in diagnoses]
        labels = [0] * len(self.idx_to_label)
        for diagnosis_idx in diagnoses:
            labels[diagnosis_idx] = 1

        tpath = os.path.join("/data/traindata/mfcc", f"{self.df.iloc[idx]['GUID']}.pt")
        mfcc_tensor = torch.load(tpath, weights_only=True)
        mfcc_tensor = torch.nn.functional.pad(mfcc_tensor, (1, 999 - mfcc_tensor.shape[2]))
        return {
            "vecs": mfcc_tensor,
            "transcript": self.df.iloc[idx]['text'],
            "labels": torch.tensor(labels)
        }

    @staticmethod
    def cache_trainval(task_name="mfcc"):
        """Restrict individual patient sessions to all train or all val; do not spread patients across train/val"""
        df = pd.read_pickle(trainroot / 'datasets' / 'filtered_combined_labels_09-28.pkl.xz', compression='xz')
        df['GUID'] = df['GUID'].astype(str)
        task_folder = os.path.join("/data/traindata", task_name)
        assert os.path.exists(task_folder)
        files_folders = glob.glob(os.path.join(task_folder, '**', '*'), recursive=True)
        cachefiles = [f for f in files_folders if os.path.isfile(f) and not f.startswith(".")]
        cachefiles = [Path(x).stem for x in cachefiles]

        cachefiles = [x.split("_")[1] for x in cachefiles]
        print(cachefiles[0])
        exit()
        assert len(cachefiles) > 0, f"There aren't any local audio segment files for the task {task_name}"
        df = df[df['GUID'].isin(cachefiles)]
        df = df.sort_values(by="patient_id").reset_index(drop=True)
        tlen = int(0.8 * len(df))
        trainval_guids = df['GUID'].to_list()
        data = {"train": trainval_guids[:tlen], "val": trainval_guids[tlen:]}

        print(f"Final lengths train {len(data['train'])} val {len(data['val'])}")

        with open(trainroot / 'datasets' / 'trainval.pkl', 'wb') as f:
            pickle.dump(data, f)


def sample_ds():
    if torch.cuda.is_available():
        num_workers = 14
    else:
        num_workers = 0
    ds = DiagClip(split="train")
    dl = DataLoader(ds, batch_size=2,
                    num_workers=num_workers, pin_memory=False,
                    drop_last=False, shuffle=True)

    for batch in dl:
        # print(f"{batch['labels'].squeeze()}")
        print(f"{batch['vecs'].shape}")
        pass


if __name__ == '__main__':
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 900)
    sample_ds()
