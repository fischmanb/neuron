"""
Sept 21 2024 audio data has been segmented and corresponding label information inserted into ~6000 feather files
Parallelized training ideal situation all ~1M labels in single feather file and sound files in a random access bucket
"""
import io
import os
import tempfile
import time
from random import shuffle

import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from train import projectroot


class GcpStorage:
    def __init__(self):
        self.storage_client = storage.Client()

    def list_files(self, bucket_name, prefix='', suffix='', sort=False):
        """
        List the files in a bucket, excluding linux hidden files that start with ._
        https://cloud.google.com/storage/docs/listing-objects#storage-list-objects-python
        :param bucket_name: str, bucket name
        :param prefix: str, see gcp docs
        :param suffix: str, wildcard for blob filepath ending such as **.wav
        :param sort: bool, sort alpha order otherwise use a shuffle to intentionally randomize order
        :return: list, bucket contents filenames
        """
        blobs = self.storage_client.list_blobs(bucket_name, prefix=prefix, match_glob=suffix)
        flist = [blob.name for blob in blobs if not blob.name.split('/')[-1].startswith("._")]
        if sort:
            flist = sorted(flist)
        else:
            shuffle(flist)
        return flist

    def download_blob_to_memory(self, bucket_name, blob_name):
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob_data = blob.download_as_bytes()
        return io.BytesIO(blob_data)

    def download_blob_to_disk_temp(self, bucket_name, blob_name, local_file_path="/Users/dm/neuro/"):
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with tempfile.NamedTemporaryFile(dir=local_file_path, suffix=".feather", delete=False) as temp_file:
            temp_file_path = temp_file.name
            blob.download_to_filename(temp_file_path)
        return temp_file_path


def dump_pickle(last_index, df):
    # print(f"Dumping at {last_index}")
    df.to_pickle(projectroot / 'datasets' / 'util' / f'aggregated_{str(last_index).zfill(4)}.pkl.xz',  compression='xz')


def extract_labels():
    """
    Consolidate the label data, by combining many files to a handful. Leave the audio_array data alone. Slow script.
    """
    start_time = time.time()
    gstor = GcpStorage()
    feather_files = gstor.list_files("extracted-features", prefix="Extracted-training-data-v2", suffix="**.feather", sort=True)
    feather_files = sorted(feather_files)

    feather_file = ''
    last_index = 0
    dfnew = pd.DataFrame()
    for fidx in tqdm(range(last_index, len(feather_files))):
        feather_file = feather_files[fidx]
        # print(f"Processing {fidx}: {feather_file}")
        dtemp = gstor.download_blob_to_disk_temp("extracted-features", feather_file)
        df = pd.read_feather(dtemp)
        df = df.drop("audio_array", axis=1)
        dfnew = pd.concat([dfnew, df], ignore_index=True)
        assert ".feather" in dtemp
        os.remove(dtemp)
        time.sleep(2)
        last_index = fidx
        if fidx % 2500 == 0 and fidx != 0:
            dump_pickle(last_index, dfnew)
            dfnew = pd.DataFrame()

    dump_pickle(last_index, dfnew)
    print(f"Last file processed {feather_file} at index {last_index}")
    elapsed_time = time.time() - start_time
    print(f"{elapsed_time / 60:.2f} minutes")


def consolidate_labels():
    """
    Combine multiple train/datasets/util/lablels pkl.xz files into one. Step Two after extract_labels.
    """
    directory = projectroot / 'datasets' / 'util' / 'labels'

    dataframes = []

    for filename in os.listdir(directory):
        if filename.endswith('.pkl.xz'):
            file_path = os.path.join(directory, filename)
            df = pd.read_pickle(file_path, compression='xz')
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_pickle(projectroot / 'datasets' / 'util' / 'combined_labels.pkl.xz', compression='xz')


def extract_audio():
    """
    Extract audio npy to local disk
    We don't currently know what these npy arrays represent or how they were generated
    """
    start_time = time.time()
    gstor = GcpStorage()
    feather_files = gstor.list_files("extracted-features", prefix="Extracted-training-data-v2", suffix="**.feather", sort=True)
    feather_files = sorted(feather_files)

    feather_file = ''
    last_index = 0
    for fidx in tqdm(range(last_index, len(feather_files))):
        feather_file = feather_files[fidx]
        # print(f"Processing {fidx}: {feather_file}")
        dtemp = gstor.download_blob_to_disk_temp("extracted-features", feather_file)
        df = pd.read_feather(dtemp)
        arrdata = df['audio_array'].to_numpy()
        # TODO do stuff here or breakpoint
        assert ".feather" in dtemp
        os.remove(dtemp)
        last_index = fidx

    print(f"Last file processed {feather_file} at index {last_index}")
    elapsed_time = time.time() - start_time
    print(f"{elapsed_time / 60:.2f} minutes")


def count_files_gcp(audio_bucket="private-management-files"):
    """count files in a bucket 84,236
        TODO list the filenames and ensure the files referenced in labels are all in here
    """
    client = storage.Client()
    bucket = client.bucket(audio_bucket)
    blobs = bucket.list_blobs()
    file_count = sum(1 for _ in blobs)
    print(f"{file_count} audio files in the GCP bucket {audio_bucket}")

    return file_count


def clip_audio(subset=True):
    """
    Load an audio file and clip it between two timestamps.
    Subset is for the initial training where all 85,000 audio files won't be preprocessed
    """
    # Load annotations dataframe
    # sort by original filenames


if __name__ == '__main__':
    # extract_labels()
    # consolidate_labels()
    # extract_audio()
    clip_audio()
