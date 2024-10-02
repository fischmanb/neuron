import os

import torch
import io
from google.cloud import storage
import torchaudio


class GcpStorage:
    def __init__(self):
        self.storage_client = storage.Client()

    def resample(self, waveform, orig_sample_rate=None, new_sample_rate=8000):
        """TODO unverified"""
        transform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=new_sample_rate)
        transformed = transform(waveform)
        return transformed

    def list_files(self, bucket_name, prefix='', suffix=''):
        """
        List the files in a bucket, excluding linux hidden files that start with ._
        https://cloud.google.com/storage/docs/listing-objects#storage-list-objects-python
        :param bucket_name: str, bucket name
        :param prefix: str, see gcp docs
        :param suffix: str, wildcard for blob filepath ending such as **.wav
        :return: list, bucket contents filenames
        """
        blobs = self.storage_client.list_blobs(bucket_name, prefix=prefix, match_glob=suffix)
        return [blob.name for blob in blobs if not blob.name.split('/')[-1].startswith("._")]

    def download_blob_to_memory(self, bucket_name, blob_name):
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob_data = blob.download_as_bytes()
        return io.BytesIO(blob_data)

    def download_blob_to_disk(self, bucket_name, blob_name, local_file_path):
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob_data = blob.download_as_bytes()
        with open(local_file_path, 'wb') as file:
            file.write(blob_data)
        print(f"Downloaded {blob_name} to {local_file_path}")


def save_model(savedir, class_names, task, run_name, epoch, model, optimizer, lr, save_filename=None):
    """
    Save metrics, model params, optim, sched and also a full network graph and weights.

    Parameters
    ----------
    task: which finetune such as diagnosis
    savedir: str absolute path to save in
    class_names: list
        Comes from list that is in the correct order to be indexed for labeling inference output
    run_name : str
    epoch  :  int current epoch (starting with 1)
    model
    optimizer
    scheduler
    """
    if not save_filename:
        save_filename = f"{task}-{run_name}-ep{epoch}.pt"
    save_fullpath = str(os.path.join(savedir, save_filename))
    print(f"Saving model {save_fullpath}")

    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

    if type(class_names) is list:
        converted_classnames = ','.join(class_names)
    else:
        converted_classnames = str(class_names)

    torch.save({
        'epoch': epoch,
        'class_names': converted_classnames,
        'model_info': model.__class__,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': str(lr),
        'optimizer_info': str(optimizer),
    }, save_fullpath)
    return save_filename
