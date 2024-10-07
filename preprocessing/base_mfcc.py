import io
import json
import os
import re
import tempfile

import ffmpeg
import torchaudio
from google.cloud import storage
from silero_vad import load_silero_vad, get_speech_timestamps
from feature.core import MelFrequency


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

    def extract_label_id(self, filename):
        """audio files and JSON annotations are matched by an ID that are separated with a hyphen by convention"""
        match = re.search(r'-(.*)\.', filename)
        if match:
            return match.group(1)
        return None

    def get_ann(self, json_bucket_name, json_file):
        # TODO refactor this function

        bucket = self.storage_client.get_bucket(json_bucket_name)
        blob = bucket.blob(json_file)

        if not blob.exists():
            return None

        json_content = blob.download_as_text()
        data = json.loads(json.loads(json_content))

        if 'transcription' not in data:
            return None

        processed_segments = []
        for segment in data['transcription']:
            if 'timestamp' in segment and isinstance(segment['timestamp'], list) and len(segment['timestamp']) == 2:
                processed_segments.append({
                    'start': segment['timestamp'][0],
                    'end': segment['timestamp'][1],
                    'speaker': segment.get('speaker', ''),
                    'voice': segment.get('voice', ''),
                    'text': segment.get('text', '')
                })
        return processed_segments

    def m4a_to_wav(self, m4amem, remove=True, verbose=False):
        """Convert the in-memory m4a file to wav format using a temporary file."""
        try:
            m4amem.seek(0)

            with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".m4a", delete=False) as temp_m4a_file:
                temp_m4a_file.write(m4amem.read())
                temp_m4a_file_path = temp_m4a_file.name

            out, err = (
                ffmpeg
                .input(temp_m4a_file_path)  # Read from the temporary file on disk
                .output('pipe:1', format='wav', ar='16000', ac=1)  # Set sample rate to 16kHz and mono (1 channel)
                .run(capture_stdout=True, capture_stderr=True)
            )
            if remove:
                os.remove(temp_m4a_file_path)
            if verbose:
                print("FFmpeg conversion logs:\n", err.decode('utf-8'))

            wav_file_in_memory = io.BytesIO(out)
            if wav_file_in_memory is not None:
                wav_file_in_memory.seek(0)  # Ensure the cursor is at the beginning
                waveform, sample_rate = torchaudio.load(wav_file_in_memory)
                if verbose:
                    print(f"Waveform shape: {waveform.shape}")
                    print(f"Sample rate: {sample_rate}")

                return waveform, sample_rate
            return None, None

        except ffmpeg.Error as e:
            raise Exception(f"Error during conversion: {e.stderr.decode('utf-8')}")

    def verify_wav_file_from_memory(self, wav_file_in_memory):
        wav_file_in_memory.seek(0)
        waveform, sample_rate = torchaudio.load(wav_file_in_memory)

        print(f"Sample rate: {sample_rate}")
        print(f"Waveform shape: {waveform.shape}")  # [channels, samples]

        if waveform.numel() == 0:
            raise ValueError("The waveform is empty.")

        if sample_rate != 16000:
            print(f"Warning: Unexpected sample rate {sample_rate} Hz")

        if waveform.size(0) not in [1, 2]:
            raise ValueError(f"Unexpected number of channels: {waveform.size(0)}")

        duration = waveform.size(1) / sample_rate
        print(f"Duration: {duration:.2f} seconds")

        print("WAV file integrity verified.")
        return True

    def verify_m4a_file_from_memory(self, m4a_file_in_memory, verbose=False):
        """verified working from memory"""
        try:
            m4a_file_in_memory.seek(0)

            out, err = (
                ffmpeg
                .input('pipe:0', format='m4a')  # Read from the BytesIO object
                .output('null', f='null')  # Null output means we're just verifying
                .run(input=m4a_file_in_memory.read(), capture_stdout=True, capture_stderr=True)
            )
            if verbose:
                print("File is valid and can be processed without errors")
            return True
        except ffmpeg.Error as e:
            raise Exception(f"Error verifying m4a file: {e.stderr.decode('utf-8')}")


def process_bucket(verify_m4a=True, verbose=False, annotate=False):
    """
    Iterate through the audio files, get segments, convert to wav
    """
    gstor = GcpStorage()
    vad_model = load_silero_vad()

    audio_bucket = "private-management-files"
    if annotate:
        json_bucket = "processed-json-files-v2"

    audio_filenames = gstor.list_files(audio_bucket, suffix="**.m4a")
    if annotate:
        json_filenames = gstor.list_files(json_bucket, suffix="**.json")
    for audio_filename in audio_filenames:
        label_id = gstor.extract_label_id(audio_filename)
        if annotate:
            annotation_matches = [x for x in json_filenames if f"-{label_id}." in x]
        if len(annotation_matches) == 0:
            pass  # TODO log the event of missing annotation JSON for this audio file
        elif len(annotation_matches) > 1:
            raise Exception(f"Multiple annotation files matched for label ID {label_id} audio file {audio_filename}")
        else:
            ann = gstor.get_ann(json_bucket, annotation_matches[0])
            m4amem = gstor.download_blob_to_memory(audio_bucket, audio_filename)
            if verify_m4a:
                if gstor.verify_m4a_file_from_memory(m4amem):
                    if verbose:
                        print("M4A file verification successful.")
                else:
                    raise Exception("M4A file verification failed.")

            wav, sr = gstor.m4a_to_wav(m4amem)
            speech_timestamps = get_speech_timestamps(wav, vad_model)
            print(speech_timestamps)
            # preemphasis, spectral gating

            # patient_features = []
            # for segment in json_data:
            #     if segment['voice'] == 'patient':
            #         start_sample = int(segment['start'] * sr)
            #         end_sample = int(segment['end'] * sr)
            #         segment_audio = audio_data[start_sample:end_sample]
            #
            #         features = extract_features(segment_audio, sr, f"{segment['start']}_{segment['end']}")
            #         if features:
            #             features['start'] = segment['start']
            #             features['end'] = segment['end']
            #             patient_features.append(features)

def test_mfcc_bucket(verify_m4a=True, verbose=False):
    """
    Iterate through the audio files, get segments, convert to wav
    """
    gstor = GcpStorage()

    mfcc_bucket = "alphaneuro-audio-segments"

    filenames = gstor.list_files(mfcc_bucket, suffix="**.m4a")

    print(filenames)


def sample_mfcc():
    filepath = "a sample wav file full path"
    waveform, sample_rate = torchaudio.load(filepath, normalize=True)
    melfreq = MelFrequency(sample_rate=sample_rate)
    mfcc_tensor = melfreq.transform(waveform)
    print(mfcc_tensor.shape)


if __name__ == '__main__':
    # sample_mfcc()
    process_bucket()
    # test_mfcc_bucket()

