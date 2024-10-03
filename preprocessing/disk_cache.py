"""
Cache features such as MFCC to disk
"""
import io
import json
import os
import re
import tempfile

import ffmpeg
import torchaudio
from google.cloud import storage
from random import shuffle
from preprocessing import preprocessingroot
import pickle


class GcpStorage:
    def __init__(self):
        self.storage_client = storage.Client(project="NeuroPad")

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

    def download_blob_to_disk(self, bucket_name, blob_name, local_file_path, alt_filename='', verbose=False):
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob_data = blob.download_as_bytes()
        with open(local_file_path, 'wb') as file:
            file.write(blob_data)
        if verbose:
            print(f"Downloaded {blob_name} to {local_file_path}")

    def download_blob_to_disk_chris(self, bucket_name, blob_name, local_file_path, alt_filename='', verbose=False):
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob_data = blob.download_as_bytes()
        final_file_path = f"{local_file_path}/{alt_filename}" if alt_filename else local_file_path

        with open(final_file_path, 'wb') as file:
            file.write(blob_data)

        if verbose:
            print(f"Downloaded {blob_name} to {final_file_path}")

    def download_blob_to_disk_temp(self, bucket_name, blob_name, local_file_path="/Users/dm/neuro/"):
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with tempfile.NamedTemporaryFile(dir=local_file_path, suffix=".feather", delete=False) as temp_file:
            temp_file_path = temp_file.name
            blob.download_to_filename(temp_file_path)
        return temp_file_path


def main():
    """Download npy arrays to local disk for analysis or training"""
    savepath = preprocessingroot / "audio_segments" / "mfcc"
    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)
    gstor = GcpStorage()
    mfcc_bucket = "alphaneuro-audio-segments"
    mfcc_bucket_prefix = "unique-segments-09-26_features"
    # TODO update the pipeline code to match the original specs agreed on by team to make list_files faster

    cache_file = 'cache.pkl'

    if not os.path.exists(cache_file):
        segments_list = gstor.list_files(mfcc_bucket, prefix=mfcc_bucket_prefix)
        segments_list = reversed(segments_list)
        with open(cache_file, 'wb') as f:
            pickle.dump(segments_list, f)

    with open(cache_file, 'rb') as f:
        segments_list = pickle.load(f)

    segments_list = list(segments_list)
    print(f"Loaded segments list {len(list(segments_list))} files")
    print("test")
    print(segments_list[0])
    print(segments_list[-1])
    for item in segments_list:
        print(item)

    for blob_fullpath in segments_list:
        if blob_fullpath.split('/')[-1] == 'mfcc.pt':
            alt_filename = f"{blob_fullpath.split('/')[1]}.pt"
            if os.path.exists(f"{savepath}/{alt_filename}"):
                # print(f"Skipping {alt_filename}")
                pass
            else:
                gstor.download_blob_to_disk_chris(mfcc_bucket, blob_fullpath,
                                                  local_file_path=savepath,
                                                  alt_filename=alt_filename, verbose=False)


if __name__ == '__main__':
    main()
