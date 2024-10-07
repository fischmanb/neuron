

import numpy as np
import librosa
import json
import tempfile
from google.cloud import storage
from tqdm import tqdm
import multiprocessing
from functools import partial
import os
import logging
import ffmpeg
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_audio_from_mp4(mp4_file):
    """
    Extract audio from MP4 file using ffmpeg.
    """
    try:
        out, _ = (
            ffmpeg
            .input(mp4_file)
            .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='22050')
            .run(capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    except ffmpeg.Error as e:
        logger.error(f"Error extracting audio from MP4: {e.stderr.decode()}")
        return None

def extract_mfcc_spectrograms(audio_data, sr, n_mfcc=20, n_fft=2048, hop_length=512, duration=10):
    """
    Extract MFCC, delta, and delta-delta spectrograms of fixed size.
    """
    # Ensure consistent audio length
    target_length = duration * sr
    if len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
    else:
        audio_data = audio_data[:target_length]

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Stack the spectrograms
    mfcc_spectrograms = np.stack([mfccs, delta_mfccs, delta2_mfccs], axis=-1)
    
    return mfcc_spectrograms

def process_audio_file(blob, input_bucket, sr=22050, duration=10):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            blob.download_to_filename(temp_file.name)
            audio_data = extract_audio_from_mp4(temp_file.name)
        
        if audio_data is None:
            logger.error(f"Failed to extract audio from {blob.name}")
            return (blob.name, None)
        
        mfcc_spectrograms = extract_mfcc_spectrograms(audio_data, sr, duration=duration)
        
        return (blob.name, mfcc_spectrograms)
    except Exception as e:
        logger.error(f"Error processing {blob.name}: {str(e)}")
        return (blob.name, None)
    finally:
        if 'temp_file' in locals():
            os.remove(temp_file.name)

def process_audio_files_batch(input_bucket_name, audio_directory_path, result_bucket_name, num_files=None, sr=22050, duration=10, batch_size=10):
    storage_client = storage.Client()
    input_bucket = storage_client.bucket(input_bucket_name)
    result_bucket = storage_client.bucket(result_bucket_name)

    blobs = list(input_bucket.list_blobs(prefix=audio_directory_path))
    audio_blobs = [blob for blob in blobs if blob.name.lower().endswith('.mp4')]
    
    if num_files:
        audio_blobs = audio_blobs[:num_files]

    logger.info(f"Processing {len(audio_blobs)} MP4 files")

    for i in tqdm(range(0, len(audio_blobs), batch_size)):
        batch_blobs = audio_blobs[i:i+batch_size]
        
        with multiprocessing.Pool() as pool:
            results = pool.map(partial(process_audio_file, input_bucket=input_bucket, sr=sr, duration=duration), batch_blobs)

        batch_results = {name: spectrograms for name, spectrograms in results if spectrograms is not None}

        # Save the batch results
        result_blob_name = f'mfcc_spectrograms/batch_{i//batch_size:04d}.npz'
        result_blob = result_bucket.blob(result_blob_name)
        
        # Create a temporary file to store the numpy array
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            np.savez_compressed(temp_file.name, **batch_results)
            temp_file.flush()
            result_blob.upload_from_filename(temp_file.name)
        
        # Clean up the temporary file
        os.remove(temp_file.name)

    logger.info(f"Processed {len(audio_blobs)} MP4 files.")

# Usage example:
input_bucket_name = "private-management-files"
audio_directory_path = "Voice Memos 14965 FILES"
result_bucket_name = "analysis_results"

# Process all files (or specify a number)
process_audio_files_batch(input_bucket_name, audio_directory_path, result_bucket_name, num_files=None, batch_size=100)
