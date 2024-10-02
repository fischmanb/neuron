# import argparse
# import asyncio
# import asyncio
# import cProfile
# import gc
# import io
# import io
# import io
# import json
# import json
# import json
# import logging
# import logging
# import logging
# import multiprocessing
# import os
# import os
# import pickle
# import pstats
# import random
# import re
# import re
# import shutil
# import shutil
# import subprocess
# import subprocess
# import sys
# import tempfile
# import time
# import time
# import warnings
# from collections import Counter
# from collections import defaultdict
# from collections import defaultdict
# from concurrent.futures import ProcessPoolExecutor
# from concurrent.futures import ProcessPoolExecutor
# from datetime import datetime, timedelta
# from datetime import datetime, timedelta
# from itertools import groupby
# from multiprocessing import Pool
# from pathlib import Path
#
# import aiofiles
# import antropy as ant
# import asyncpg
# import functions_framework
# import librosa
# import librosa
# import matplotlib.pyplot as plt
# import nest_asyncio
# import nolds
# import numpy as np
# import numpy as np
# import numpy as np
# import pandas as pd
# import pandas as pd
# import pandas as pd
# import parselmouth
# import psutil
# import pyarrow as pa
# import pyarrow as pa
# import pyarrow as pa
# import pyarrow.parquet as pq
# import pyarrow.parquet as pq
# import pyarrow.parquet as pq
# import pywt
# import pywt
# import pywt
# import scipy.io.wavfile
# import seaborn as sns
# import simpleaudio as sa
# import soundfile as sf
# import soundfile as sf
# import torch
# import torch
# import torchaudio
# import webrtcvad
# from google.api_core.exceptions import Forbidden
# from google.cloud import storage
# from google.cloud import storage
# from google.cloud import storage
# from google.cloud import storage
# from memory_profiler import profile
# from modules.audioFile import AudioFile
# from numba import jit
# from numba import jit
# from numpy import dot
# from numpy.linalg import norm
# from parselmouth.praat import call
# from pyAudioAnalysis import audioSegmentation as aS
# from pyannote.audio import Inference
# from pyannote.audio import Model
# from pyannote.audio import Pipeline
# from pyannote.audio.pipelines.utils.hook import ProgressHook
# from pydub import AudioSegment
# from pydub import AudioSegment
# from scipy import signal
# from scipy.io.wavfile import write as writeWav
# from scipy.signal import welch
# from scipy.signal import welch
# from scipy.signal import wiener, lfilter
# from scipy.spatial import cKDTree
# from scipy.spatial.distance import cosine
# from scipy.spatial.distance import squareform, pdist
# from scipy.stats import entropy
# from scipy.stats import entropy
# from scipy.stats import entropy, skew, kurtosis
# from scipy.stats import skew, kurtosis, entropy
# from scipy.stats import skew, kurtosis, entropy
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.mixture import GaussianMixture
# from torch import nn
# from tqdm import tqdm
# from tqdm import tqdm
# from tqdm import tqdm
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor
#
#
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Get the start_index from environment variable
# start_index = int(os.environ.get('START_INDEX', 0))
#
# # Set other parameters
# num_files = 20  # Total number of files to process in this run
# batch_size = 10  # Size of each batch
#
# # Generate output bucket name
# output_bucket = f"gcs-output-bucket-{start_index}"
#
# #  Create the bucket if it doesn't exist
# storage_client = storage.Client()
# try:
#     bucket = storage_client.get_bucket(output_bucket)
#     logger.info(f"Bucket {output_bucket} already exists")
# except Exception:
#     try:
#         bucket = storage_client.create_bucket(output_bucket)
#         logger.info(f"Bucket {output_bucket} created successfully")
#     except Exception as e:
#         logger.error(f"Failed to create bucket {output_bucket}: {str(e)}")
#
# # Display the parameters
# print(f"""
# <b>Processing Parameters:</b><br>
# Start Index: {start_index}<br>
# Number of Files: {num_files}<br>
# Batch Size: {batch_size}<br>
# Output Bucket: {output_bucket}
# """)
#
# print(f"Starting processing with start_index: {start_index}")
# # %%
# # 8.24 - Extract features from patient segemts only - O(M log M ) only
#
#
# logger = logging.getLogger(__name__)
#
# nest_asyncio.apply()
#
# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     handlers=[logging.StreamHandler()])
#
# # Create a logger for this module
# logger = logging.getLogger(__name__)
#
# # Ensure that the logger is not being filtered
# logger.setLevel(logging.INFO)
#
# # Test the logger
# logger.info("Logging has been configured.")
#
# # GCP bucket settings
# # AUDIO_BUCKET_NAME = "private-management-files"
# # JSON_BUCKET_NAME = "processed-json-files-v2"
# # OUTPUT_BUCKET_NAME = "diarized_patient_feature_analysis"
# csv_file_path = 'extraction_6k_07_30.csv'
#
# # Set the path to the Parquet file
# path_to_parquet_file = 'aggregated-speaker-embeddings-with-docs-voice-and-cluster.gzip'
#
# # Ensure the file exists
# if not os.path.exists(path_to_parquet_file):
#     raise FileNotFoundError(f"The Parquet file '{path_to_parquet_file}' does not exist in the current directory.")
#
# # Constants
# # MAX_WORKERS = 16
# TARGET_PROCESSING_TIME = 4 * 60 * 60  # 4 hours in seconds
# PROCESSED_FILES_PATH = "processed_files.json"
#
# CONFIDENCE_THRESHOLD = 0.7  # Adjust this value as needed
#
#
# class Checkpoint:
#     def __init__(self, last_processed_index=0, processed_count=0, current_file_progress=None):
#         self.last_processed_index = last_processed_index
#         self.processed_count = processed_count
#         self.current_file_progress = current_file_progress or {}
#
#     def __getitem__(self, item):
#         return getattr(self, item)
#
#     def __setitem__(self, key, value):
#         setattr(self, key, value)
#
#     def __str__(self):
#         return f"Checkpoint(last_processed_index={self.last_processed_index}, processed_count={self.processed_count}, current_file_progress={self.current_file_progress})"
#
#
# # Utility Functions
# async def load_processed_files(bucket_name, result_bucket_path):
#     logger.info(f"Loading processed files from {result_bucket_path}/processed_files.json in bucket {bucket_name}")
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(f"{result_bucket_path}/processed_files.json")
#
#     try:
#         exists = await asyncio.to_thread(blob.exists)
#         if exists:
#             logger.info("Blob exists. Downloading content.")
#             content = await asyncio.to_thread(blob.download_as_text)
#             processed_files = set(json.loads(content))
#             logger.info(f"Loaded processed files: {processed_files}")
#             return processed_files
#         else:
#             logger.warning(f"Blob does not exist at {result_bucket_path}/processed_files.json")
#             return set()  # Return an empty set if the file doesn't exist
#     except Exception as e:
#         logger.error(f"Error loading processed files: {e}")
#         return set()  # Return an empty set in case of an error
#
#
# async def save_processed_files(bucket_name, result_bucket_path, processed_files):
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(f"{result_bucket_path}/processed_files.json")
#
#     try:
#         content = json.dumps(list(processed_files))
#         await asyncio.to_thread(blob.upload_from_string, content)
#         logger.info(f"Processed files saved to {result_bucket_path}/processed_files.json")
#     except Exception as e:
#         logger.error(f"Error saving processed files: {e}")
#
#
# # async def load_checkpoint(bucket_name, result_bucket_path):
# # client = storage.Client()
# # bucket = client.bucket(bucket_name)
# # blob = bucket.blob(f"{result_bucket_path}/checkpoint.pickle")
#
# # retry_count = 3
# # for attempt in range(retry_count):
# # try:
# # if await asyncio.to_thread(blob.exists):
# # content = await asyncio.to_thread(blob.download_as_bytes)
# # return pickle.loads(content)
# # else:
# # logger.warning(f"Checkpoint file does not exist at {result_bucket_path}/checkpoint.pickle. Initializing new checkpoint.")
# # return Checkpoint()
# # except Forbidden as e:
# # logger.error(f"Access forbidden for {result_bucket_path}/checkpoint.pickle: {e}")
# # if attempt < retry_count - 1:
# # await asyncio.sleep(2 ** attempt)  # Exponential backoff
# # else:
# # return Checkpoint()
# # except Exception as e:
# # logger.error(f"Error loading checkpoint: {e}")
# # return Checkpoint()
#
# async def load_checkpoint(bucket_name, result_bucket_path):
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(f"{result_bucket_path}/checkpoint.pickle")
#
#     try:
#         if await asyncio.to_thread(blob.exists):
#             content = await asyncio.to_thread(blob.download_as_bytes)
#             loaded_data = pickle.loads(content)
#             logger.info(f"Loaded data type: {type(loaded_data)}, content: {loaded_data}")
#
#             if isinstance(loaded_data, Checkpoint):
#                 logger.info("Loaded data is a Checkpoint instance")
#                 checkpoint = loaded_data
#             else:
#                 logger.warning("Loaded data is not a Checkpoint instance. Creating new Checkpoint from loaded data.")
#                 checkpoint = Checkpoint(
#                     last_processed_index=loaded_data.get('last_processed_index', 0),
#                     processed_count=loaded_data.get('processed_count', 0),
#                     current_file_progress=loaded_data.get('current_file_progress', {})
#                 )
#         else:
#             logger.warning(f"Checkpoint file does not exist. Initializing new checkpoint.")
#             checkpoint = Checkpoint()
#
#         logger.info(f"Final checkpoint: {checkpoint}")
#         return checkpoint
#     except Exception as e:
#         logger.error(f"Error loading checkpoint: {e}")
#         logger.info("Initializing new checkpoint due to error.")
#         return Checkpoint()
#
#
# async def save_checkpoint(bucket_name, result_bucket_path, checkpoint):
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(f"{result_bucket_path}/checkpoint.pickle")
#     try:
#         await asyncio.to_thread(blob.upload_from_string, pickle.dumps(checkpoint))
#         logger.info(f"Checkpoint saved to {result_bucket_path}/checkpoint.pickle")
#     except Exception as e:
#         logger.error(f"Error saving checkpoint: {e}")
#
#
# def select_random_subset(file_pairs, num_files):
#     return random.sample(file_pairs, min(num_files, len(file_pairs)))
#
#
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)
#
# # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # logger = logging.getLogger(__name__)
#
# # def load_diarization_json(json_file_path):
# # with open(json_file_path, 'r') as f:
# # data = json.load(f)
#
# # processed_segments = []
# # if 'transcription' in data:
# # for segment in data['transcription']:
# # processed_segments.append({
# # 'start': segment['timestamp'][0],
# # 'end': segment['timestamp'][1],
# # 'speaker': segment.get('speaker', ''),
# # 'voice': segment.get('voice', ''),
# # 'text': segment.get('text', '')
# # })
#
# # return processed_segments
#
#
# def numpy_to_python(obj):
#     if isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     return obj
#
#
# def monitor_cpu_usage():
#     cpu_percent = psutil.cpu_percent(interval=1)
#     logger.info(f"Current CPU usage: {cpu_percent}%")
#     return cpu_percent
#
#
# def monitor_disk_io():
#     io_counters = psutil.disk_io_counters()
#     logger.info(f"Disk read: {io_counters.read_bytes / (1024 * 1024):.2f} MB, "
#                 f"Disk write: {io_counters.write_bytes / (1024 * 1024):.2f} MB")
#
#
# def monitor_network():
#     net_io = psutil.net_io_counters()
#     logger.info(f"Network sent: {net_io.bytes_sent / (1024 * 1024):.2f} MB, "
#                 f"Network received: {net_io.bytes_recv / (1024 * 1024):.2f} MB")
#
#
# def load_filtering_criteria(parquet_file_path):
#     df = pd.read_parquet(parquet_file_path)
#     return df[['original-file', 'num-unique-speakers', 'cluster', 'full-audio-duration(seconds)']]
#
#
# def save_checkpoint_to_gcs(bucket_name, blob_name, data):
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#     blob.upload_from_string(json.dumps(data, default=numpy_to_python))
#
#
# processed_files = set()
#
#
# def check_if_processed(filename):
#     # Check both with and without the .json extension
#     return filename in processed_files or f"{filename}.json" in processed_files
#
#
# def mark_as_processed(filename, bucket_name, result_bucket_path, processed_files):
#     processed_files.add(f"{filename}.json")
#
#     # Write the updated set to the file in GCS
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(f"{result_bucket_path}/successful_extractions.txt")
#
#     content = "\n".join(processed_files)
#     blob.upload_from_string(content)
#
#     logger.info(f"Marked {filename} as processed and updated successful_extractions.txt")
#
#
# def load_processed_files(bucket_name, result_bucket_path):
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(f"{result_bucket_path}/successful_extractions.txt")
#
#     if blob.exists():
#         content = blob.download_as_text()
#         for line in content.split('\n'):
#             if line.strip():
#                 filename = line.split(':')[0].strip()
#                 # Ensure the .json extension is present
#                 if not filename.endswith('.json'):
#                     filename += '.json'
#                 processed_files.add(filename)
#     logger.info(f"Loaded {len(processed_files)} already processed files.")
#
#
# def load_diarization_json(json_bucket_name, json_file):
#     try:
#         client = storage.Client()
#         bucket = client.get_bucket(json_bucket_name)
#         blob = bucket.blob(json_file)
#
#         if not blob.exists():
#             logging.error(f"JSON file {json_file} does not exist in bucket {json_bucket_name}")
#             return None
#
#         json_content = blob.download_as_text()
#         # logging.info(f"Raw JSON content: {json_content[:1000]}...")  # Log first 1000 characters
#         # print(f"Raw JSON content: {json_content[:1000]}")
#
#         try:
#             data = json.loads(json.loads(json_content))
#         except json.JSONDecodeError as e:
#             logging.error(f"Failed to parse JSON: {str(e)}")
#             return None
#
#         logging.info(f"Parsed JSON data type: {type(data)}")
#
#         if not isinstance(data, dict):
#             logging.error(f"Parsed JSON is not a dictionary. Actual type: {type(data)}")
#             return None
#
#         if 'transcription' not in data:
#             logging.error("'transcription' key not found in JSON data")
#             return None
#
#         if not isinstance(data['transcription'], list):
#             logging.error(f"'transcription' is not a list. Actual type: {type(data['transcription'])}")
#             return None
#
#         processed_segments = []
#         for i, segment in enumerate(data['transcription']):
#             # logging.info(f"Processing segment {i}: {segment}")
#
#             # Check if 'timestamp' exists and is in the expected format
#             if 'timestamp' in segment:
#                 if isinstance(segment['timestamp'], list) and len(segment['timestamp']) == 2:
#                     start, end = segment['timestamp']
#                 elif isinstance(segment['timestamp'], dict) and 'start' in segment['timestamp'] and 'end' in segment[
#                     'timestamp']:
#                     start = segment['timestamp']['start']
#                     end = segment['timestamp']['end']
#                 else:
#                     logging.warning(f"Unexpected 'timestamp' format in segment {i}: {segment['timestamp']}")
#                     continue
#             else:
#                 logging.warning(f"No 'timestamp' found in segment {i}")
#                 continue
#
#             try:
#                 processed_segments.append({
#                     'start': float(start),
#                     'end': float(end),
#                     'speaker': segment.get('speaker', ''),
#                     'voice': segment.get('voice', ''),
#                     'text': segment.get('text', '')
#                 })
#             except (TypeError, ValueError) as e:
#                 logging.error(f"Error processing segment {i}: {str(e)}")
#
#         if not processed_segments:
#             logging.warning(f"No diarized segments found in JSON: {json_file}")
#
#         return processed_segments
#
#     except Exception as e:
#         logging.error(f"Error loading diarized segments from {json_file}: {str(e)}")
#         logging.exception("Traceback:")  # log the full traceback
#         return None
#
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logger = logging.getLogger(__name__)
#
#
# class SpeakerEmbedding(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.device = torch.device("cpu")  # Force CPU usage
#         self.vec2wav_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM",
#                                                    use_auth_token="hf_")
#         self.vec2wav_model = self.vec2wav_model.to(self.device)
#         self.vec2wav = Inference(self.vec2wav_model, window="whole", duration=3.0, step=1.0)
#
#     def __call__(self, filepath: str):
#         with torch.no_grad():
#             embeddings = self.vec2wav(filepath)
#         return embeddings
#
#
# def load_doctor_embedding(file_path):
#     with open(file_path, 'r') as f:
#         embedding_data = json.load(f)
#     if isinstance(embedding_data, list):
#         return np.array(embedding_data)
#     elif isinstance(embedding_data, dict):
#         return np.array(list(embedding_data.values()))
#     else:
#         raise ValueError("Unexpected format in doctor embedding file")
#
#
# class speakerSeparator:
#     def __init__(self):
#         self.device = torch.device("cpu")  # Force CPU usage
#         try:
#             self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
#                                                      use_auth_token="hf_")
#             if self.pipeline is None:
#                 raise ValueError("Pipeline.from_pretrained returned None")
#             self.pipeline = self.pipeline.to(self.device)
#         except Exception as e:
#             logger.error(f"Error initializing pipeline: {e}")
#             self.pipeline = None
#
#     def __call__(self, signal: np.array, sampling_rate=16000, min_speakers=1, max_speakers=4):
#         if self.pipeline is None:
#             logger.error("Pipeline is not initialized. Cannot perform diarization.")
#             return pd.DataFrame(columns=["start", "end", "speaker"])
#
#         tensor = torch.from_numpy(signal).float().unsqueeze(0)
#         file = {"waveform": tensor, "sample_rate": sampling_rate}
#
#         try:
#             diarization = self.pipeline(file,
#                                         min_speakers=min_speakers,
#                                         max_speakers=max_speakers)
#
#             segments = []
#             for turn, _, speaker in diarization.itertracks(yield_label=True):
#                 segments.append({
#                     "start": turn.start,
#                     "end": turn.end,
#                     "speaker": speaker
#                 })
#
#             return pd.DataFrame(segments)
#         except Exception as e:
#             logger.error(f"Error during diarization: {e}")
#             return pd.DataFrame(columns=["start", "end", "speaker"])
#
#
# def calculate_simplified_silhouette(similarities, threshold):
#     labels = similarities > threshold
#     cluster_0 = similarities[~labels]
#     cluster_1 = similarities[labels]
#     if len(cluster_0) == 0 or len(cluster_1) == 0:
#         return -1  # Invalid clustering
#     a = np.mean(np.abs(similarities - np.mean(similarities)))
#     b = min(np.mean(cluster_0), np.mean(cluster_1))
#     return (b - a) / max(a, b)
#
#
# class DynamicThresholder:
#     def __init__(self, window_size=1000, initial_threshold=0.1):
#         self.window_size = window_size
#         self.similarities = []
#         # self.threshold = 0.5  # Initial default threshold
#         self.threshold = initial_threshold
#
#     def update(self, similarity):
#         self.similarities.append(similarity)
#         if len(self.similarities) > self.window_size:
#             self.similarities.pop(0)
#
#         if len(self.similarities) >= 10:  # Minimum samples for reliable statistics
#             # IQR method
#             median = np.median(self.similarities)
#             q1, q3 = np.percentile(self.similarities, [25, 75])
#             iqr = q3 - q1
#             iqr_threshold = median + 1.5 * iqr
#
#             # Silhouette method
#             possible_thresholds = np.percentile(self.similarities, [25, 50, 75])
#             silhouette_scores = [calculate_simplified_silhouette(self.similarities, t) for t in possible_thresholds]
#             silhouette_threshold = possible_thresholds[np.argmax(silhouette_scores)]
#
#             # Combine both methods (adjust the weights as needed)
#             self.threshold = 0.5 * iqr_threshold + 0.7 * silhouette_threshold
#
#     def get_threshold(self):
#         return self.threshold
#
#
# class AdaptiveThresholder:
#     def __init__(self, window_size=100, percentile=75):
#         self.thresholds = {}
#         self.data = defaultdict(list)
#         self.window_size = window_size
#         self.percentile = percentile
#         self.logger = logging.getLogger(__name__)
#
#     def update(self, feature_dict):
#         for feature, value in feature_dict.items():
#             if isinstance(value, (int, float)):
#                 self.data[feature].append(value)
#                 if len(self.data[feature]) > self.window_size:
#                     self.data[feature] = self.data[feature][-self.window_size:]
#                 old_threshold = self.thresholds.get(feature, None)
#                 self.thresholds[feature] = np.percentile(self.data[feature], self.percentile)
#                 self.logger.info(f"Updated threshold for {feature}: {old_threshold} -> {self.thresholds[feature]}")
#             elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
#                 max_value = np.max(value)
#                 self.data[feature].append(max_value)
#                 if len(self.data[feature]) > self.window_size:
#                     self.data[feature] = self.data[feature][-self.window_size:]
#                 old_threshold = self.thresholds.get(feature, None)
#                 self.thresholds[feature] = np.percentile(self.data[feature], self.percentile)
#                 self.logger.info(f"Updated threshold for {feature}: {old_threshold} -> {self.thresholds[feature]}")
#
#     def get_threshold(self, feature, default_value):
#         threshold = self.thresholds.get(feature, default_value)
#         self.logger.info(f"Getting threshold for {feature}: {threshold}")
#         return threshold
#
#     def get_all_thresholds(self):
#         self.logger.info(f"Current thresholds: {self.thresholds}")
#         return self.thresholds
#
#
# def safe_get_threshold(adaptive_thresholder, feature, default_value):
#     try:
#         return adaptive_thresholder.get_threshold(feature, default_value)
#     except Exception as e:
#         logger.error(f"Error getting threshold for {feature}: {e}")
#         return default_value
#
#
# def visualize_real_thresholds(adaptive_thresholder, feature_name):
#     thresholds = []
#     values = []
#
#     # Assuming you have a list of processed segments or features
#     for segment in processed_segments:  # or however you iterate through your data
#         feature_value = segment['features'].get(feature_name)
#         if feature_value is not None:
#             adaptive_thresholder.update({feature_name: feature_value})
#             thresholds.append(adaptive_thresholder.get_threshold(feature_name, 0))
#             values.append(feature_value)
#
#     plt.figure(figsize=(12, 6))
#     plt.plot(range(len(thresholds)), thresholds, label='Adaptive Threshold')
#     plt.scatter(range(len(values)), values, alpha=0.5, label='Feature Values')
#     plt.title(f"Adaptive Threshold for {feature_name}")
#     plt.xlabel("Update Number")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.show()
#
#
# def scale_feature(value, min_val, max_val):
#     return (value - min_val) / (max_val - min_val) if max_val > min_val else 0
#
#
# # handle a wide variety of similarity values to calculate the confidence scale
# # def calculate_confidence(similarity, thresholder):
# # median = np.median(thresholder.similarities)
# # mad = np.median(np.abs(thresholder.similarities - median))
# # z_score = (similarity - median) / (mad * 1.4826)  # 1.4826 is a constant to make MAD comparable to standard deviation
# # confidence = 1 / (1 + np.exp(-abs(z_score)))  # Sigmoid function
# # return confidence
#
# # Incorporate a confidence score for speaker identification
# class SpeakerConfidenceEstimator:
#     def __init__(self):
#         self.gmm = GaussianMixture(n_components=2, random_state=0)
#         self.patient_features = []
#         self.doctor_features = []
#
#     def add_sample(self, features, is_patient):
#         if is_patient:
#             self.patient_features.append(features)
#         else:
#             self.doctor_features.append(features)
#
#     def train(self):
#         if not self.patient_features or not self.doctor_features:
#             logger.warning("Not enough samples to train the SpeakerConfidenceEstimator. "
#                            f"Patient samples: {len(self.patient_features)}, "
#                            f"Doctor samples: {len(self.doctor_features)}")
#             return False
#
#         try:
#             X = np.vstack(self.patient_features + self.doctor_features)
#             y = np.hstack([np.ones(len(self.patient_features)), np.zeros(len(self.doctor_features))])
#             self.gmm.fit(X, y)
#             logger.info(f"SpeakerConfidenceEstimator trained successfully with "
#                         f"{len(self.patient_features)} patient samples and "
#                         f"{len(self.doctor_features)} doctor samples.")
#             return True
#         except Exception as e:
#             logger.error(f"Error training SpeakerConfidenceEstimator: {e}")
#             return False
#
#     def estimate_confidence(self, features):
#         if not hasattr(self.gmm, 'means_'):
#             logger.warning("SpeakerConfidenceEstimator is not trained. Returning default confidence of 0.5.")
#             return 0.5
#         try:
#             proba = self.gmm.predict_proba(features.reshape(1, -1))[0]
#             return proba[1]  # Probability of being the patient
#         except Exception as e:
#             logger.error(f"Error estimating confidence: {e}")
#             return 0.5
#
#
# # keep for feature extraction and similarity comparisons
#
# class PatientFeatureExtractor:
#     def __init__(self, doctor_embedding=None):
#         self.doctor_embedding = doctor_embedding
#         self.thresholder = DynamicThresholder()
#         self.doctor_centroid = None
#         self.doctor_flux = None
#         self.doctor_feature_stats = {}
#         self.patient_feature_stats = {}
#         self.confidence_estimator = SpeakerConfidenceEstimator()
#
#     def update_doctor_spectral_features(self, centroid, flux):
#         alpha = 0.1
#         if self.doctor_centroid is None:
#             self.doctor_centroid = centroid
#             self.doctor_flux = flux
#         else:
#             # alpha = 0.1
#             if centroid.shape != self.doctor_centroid.shape:
#                 logger.warning(f"Centroid shape mismatch. Resizing.")
#                 centroid = np.resize(centroid, self.doctor_centroid.shape)
#             if flux.shape != self.doctor_flux.shape:
#                 logger.warning(f"Flux shape mismatch. Resizing.")
#                 flux = np.resize(flux, self.doctor_flux.shape)
#
#         self.doctor_centroid = (1 - alpha) * self.doctor_centroid + alpha * centroid
#         self.doctor_flux = (1 - alpha) * self.doctor_flux + alpha * flux
#
#         logger.info(f"Updated doctor spectral centroid shape: {self.doctor_centroid.shape}")
#         logger.info(f"Updated doctor spectral flux shape: {self.doctor_flux.shape}")
#
#     def extract_speaker_features(self, segment_audio, sampling_rate):
#         features = {
#             'mfcc': np.mean(librosa.feature.mfcc(y=segment_audio, sr=sampling_rate), axis=1),
#             'chroma': np.mean(librosa.feature.chroma_stft(y=segment_audio, sr=sampling_rate), axis=1),
#             'zcr': np.mean(librosa.feature.zero_crossing_rate(segment_audio))
#         }
#         return features
#
#     def get_segment_features(self, segment_audio, sampling_rate):
#         if len(segment_audio) == 0:
#             logger.warning("Segment audio is empty, returning None")
#             return None  # Skip zero length audio
#
#         try:
#             # Extract basic features
#             features = extract_features(segment_audio, sampling_rate, "segment")
#
#             # Extract advanced features and add them to the features dictionary
#             advanced_features = extract_advanced_features(segment_audio, sampling_rate)
#
#             if advanced_features is not None:
#                 features.update(advanced_features)
#
#                 # Log successfully extracted advanced features
#                 logger.info(f"Successfully extracted advanced features: {list(advanced_features.keys())}")
#
#             # Ensure all expected features are present in the features dictionary
#             expected_features = [
#                 'pitch_variability', 'vtln', 'emotion', 'pauses', 'intervals',
#                 'formant_frequencies', 'formant_freq_ratio', 'spectral_irregularity',
#                 'wavelet_coefficients', 'wavelet_energy', 'formant_bandwidths', 'peak_to_peak',
#                 'mfccs', 'delta_mfccs', 'delta_delta_mfccs', 'spectral_flux', 'spectral_rolloff',
#                 'spectral_flatness', 'harmonics_to_noise_ratio', 'pitches', 'spectral_centroid',
#                 'zero_crossing_rate', 'spectral_entropy', 'spectral_skewness', 'spectral_kurtosis',
#                 'stft_magnitude', 'skewness', 'kurtosis', 'spectral_contrast', 'lsp', 'bark_spectrogram',
#                 'speech_rate', 'speech_intensity', 'harmonicity', 'cpp', 'jitter', 'shimmer', 'pitch',
#                 'energy', 'duration', 'mean', 'variance', 'std_dev', 'median', 'min_val', 'max_val',
#                 'iqr', 'rms', 'crest_factor', 'hurst_exp', 'permutation_entropy', 'sample_entropy',
#             ]
#
#             missing_features = [feature for feature in expected_features if feature not in features]
#             if missing_features:
#                 logger.warning(f"Missing features: {missing_features}")
#             else:
#                 logger.info(f"All expected features are present in the features dictionary.")
#
#             # Log all features in the features dictionary
#             logger.info(f"All features in the features dictionary: {list(features.keys())}")
#
#             return features
#         except Exception as e:
#             logger.error(f"Error in get_segment_features: {str(e)}", exc_info=True)
#             return None
#
#     def update_feature_stats(self, features, is_doctor):
#         stats = self.doctor_feature_stats if is_doctor else self.patient_feature_stats
#         current_time = datetime.now()
#
#         for feature, value in features.items():
#             logger.debug("feature: {feature}, Value Type: {type(value)}, Value: {value}")
#
#             if isinstance(value, (np.ndarray, list)):
#                 if feature == 'wavelet_coeff':
#                     value = np.mean([np.mean(c) for c in value])  # convert list of arrays to single scalar
#                 else:
#                     value = np.mean(value)  # Convert to scalar by taking the mean
#
#             if feature not in stats:
#                 stats[feature] = []
#             stats[feature].append((current_time, value))
#
#     def log_feature_out_of_range(feature_name, feature_value, expected_range, is_doctor):
#         if isinstance(feature_value, np.ndarray):
#             if not ((expected_range[0] <= feature_value).all() and (feature_value <= expected_range[1]).all()):
#                 logger.warning(
#                     f"{'Doctor' if is_doctor else 'Patient'} feature '{feature_name}' with value {feature_value} is outside expected range {expected_range}")
#         else:
#             if not (expected_range[0] <= feature_value <= expected_range[1]):
#                 logger.warning(
#                     f"{'Doctor' if is_doctor else 'Patient'} feature '{feature_name}' with value {feature_value} is outside expected range {expected_range}")
#
#     def validate_features(self, features, is_doctor, sampling_rate):
#         logger.info(
#             f"Validating features for {'doctor' if is_doctor else 'patient'} with sampling_rate: {sampling_rate}")
#         # print(f"validate_features called with sampling_rate: {sampling_rate}")
#         validation_results = {}
#
#         self.doctor_ranges = {
#             'pitch': (80, 180),  # Hz (typically lower for male doctors)
#             'speaking_rate': (2.5, 3.5),  # syllables per second (potentially higher for doctors)
#             'energy': (0.1, 0.8),  # normalized energy
#             'jitter': (0, 0.015),  # relative jitter (potentially lower for trained speakers)
#             'shimmer': (0, 0.08),  # relative shimmer (potentially lower for trained speakers)
#             'spectral_entropy': (5, 8),  # bits
#             'harmonics_to_noise_ratio': (10, 35),  # dB (potentially higher for trained speakers)
#             'spectral_flatness': (0.1, 0.5),
#             'spectral_contrast': (20, 80),  # dB
#             # 'wavelet_energy': (0, 1000)
#         }
#
#         self.patient_ranges = {
#             'pitch': (50, 400),  # Hz (wider range to account for diverse patients)
#             'speaking_rate': (2, 3),  # syllables per second
#             'energy': (0, 1),  # normalized energy
#             'jitter': (0, 0.02),  # relative jitter
#             'shimmer': (0, 0.1),  # relative shimmer
#             'spectral_entropy': (0, 10),  # bits
#             'harmonics_to_noise_ratio': (5, 30),  # dB
#             'spectral_flatness': (0, 1),
#             'spectral_contrast': (0, 100),  # dB
#             # 'wavelet_energy': (0, 1000)
#         }
#
#         expected_ranges = self.doctor_ranges if is_doctor else self.patient_ranges
#
#         # for feature, (min_val, max_val) in expected_ranges.items():
#         # if feature in features:
#         # if isinstance(features[feature], (list, np.ndarray)):
#         # For list-type features, check if all values are within range
#         # values = features[feature]
#         # is_valid = all(min_val <= v <= max_val for v in values)
#         # if not is_valid:
#         # logger.warning(f"{'Doctor' if is_doctor else 'Patient'} feature '{feature}' "
#         # f"with value {value:.2f} is outside expected range [{min_val}, {max_val}]")
#         # else:
#
#         # value = features[feature]
#         # is_valid = min_val <= value <= max_val
#         # if not is_valid:
#         # logger.warning(f"{'Doctor' if is_doctor else 'Patient'} feature '{feature}' "
#         # f"with value {value:.2f} is outside expected range [{min_val}, {max_val}]")
#
#         # validation_results[feature] = is_valid
#
#         # for feature, (min_val, max_val) in expected_ranges.items():
#         # if feature in features:
#         # feature_value = features[feature]
#         # if isinstance(feature_value, (list, np.ndarray)):
#         # For list-type features, check if all values are within range
#         # is_valid = all(min_val <= v <= max_val for v in feature_value)
#         # else:
#         # is_valid = min_val <= feature_value <= max_val
#
#         # if not is_valid:
#         # logger.warning(f"{'Doctor' if is_doctor else 'Patient'} feature '{feature}' "
#         # f"with value {feature_value} is outside expected range [{min_val}, {max_val}]")
#         # validation_results[feature] = is_valid
#
#         for feature, (min_val, max_val) in expected_ranges.items():
#             if feature in features:
#                 feature_value = features[feature]
#                 if isinstance(feature_value, (list, np.ndarray)):
#                     is_valid = (min_val <= np.array(feature_value)).all() and (np.array(feature_value) <= max_val).all()
#                 else:
#                     is_valid = min_val <= feature_value <= max_val
#
#                 if not is_valid:
#                     logger.warning(f"{'Doctor' if is_doctor else 'Patient'} feature '{feature}' "
#                                    f"with value {feature_value} is outside expected range [{min_val}, {max_val}]")
#                 validation_results[feature] = is_valid
#
#         # Special handling for formants
#         if 'formants' in features:
#             formant_freqs = np.roots(features['formants'])
#             formant_freqs = formant_freqs[np.imag(formant_freqs) >= 0]
#             formant_freqs = np.sort(np.abs(formant_freqs)) * (sampling_rate / (2 * np.pi))
#             validation_results['formants'] = 200 <= formant_freqs[0] <= 1000 and 800 <= formant_freqs[1] <= 2500
#             logger.info(f"{'Doctor' if is_doctor else 'Patient'} formant frequencies: {formant_freqs}")
#
#         # Wavelet energy validation ( probably will have  to adjust these ranges)
#         # Special handling for wavelet energy
#         if 'wavelet_energy' in features:
#             wavelet_energy = features['wavelet_energy']
#             # Adjust these thresholds based on distribution
#             is_valid = all(0 < energy < 1000 for energy in wavelet_energy)
#             validation_results['wavelet_energy'] = is_valid
#             if not is_valid:
#                 logger.warning(
#                     f"{'Doctor' if is_doctor else 'Patient'} wavelet energy contains values outside expected range (0, 1000)")
#
#         return validation_results
#
#     async def process_segment(self, segment_audio, sampling_rate, segment_info):
#         try:
#             if len(segment_audio) == 0:
#                 logger.warning("Skipping zero length segment")
#                 return None  # skip zero length segments
#
#             features = await self.get_segment_features(segment_audio, sampling_rate)
#             if features is None:
#                 logger.warning(
#                     f"Skipping segment {segment_info['start']:.2f}-{segment_info['end']:.2f} due to feature extraction failure")
#                 return None  # Skip if features ar None
#
#             is_doctor = segment_info['voice'] == 'doctor'
#             validation_results = self.validate_features(features, is_doctor, sampling_rate)
#
#             # Ensure all features are converted to scalars before updating stats
#             # features = {feature: (np.mean(value) if isinstance(value, (np.ndarray, list)) else value) for feature, value in features.items()}
#             for feature, value in features.items():
#                 logger.debug(f"Feature: {feature}, Value: {value}")
#                 if isinstance(value, (np.ndarray, list)):
#                     if feature == 'wavelet_coeff':
#                         value = np.mean([np.mean(c) for c in value])  ## Convert list of arrays to a single scalar
#                     else:
#                         value = np.mean(value)  # Convert to scalar by taking the mean
#                 features[feature] = value
#
#             self.update_feature_stats(features, is_doctor)
#
#             # wavelet energy visualization
#             # save_path = f"wavelet_energy_{segment_info['start']:.2f}-{segment_info['end']:.2f}.png"
#             # self.visualize_wavelet_energy(features['wavelet_coeffs'], segment_info, save_path)
#
#             # Extract patient_id from segment_info, assuming it's present
#             # patient_id = segment_info.get('patient_id', 'unknown')
#             # Transform patient_id: replace _segment*_xxx with .json
#             # patient_id = re.sub(r'_segment\d+_\w+$', '.json', patient_id)
#
#             # Remove .json extension if present
#             # patient_id = patient_id.replace('.json', '')
#
#             result = {
#                 "patient_id": patient_id,
#                 "start": segment_info['start'],
#                 "end": segment_info['end'],
#                 "voice": segment_info['voice'],
#                 "speaker": segment_info['speaker'],
#                 "text": segment_info['text'],
#                 "features": features,
#                 # "confidence": confidence, #7.25
#                 "threshold": self.thresholder.get_threshold(),  # 7.25
#                 "feature_validation": validation_results
#             }
#
#             # self.confidence_estimator.add_sample(speaker_features, not is_doctor) # 7.25
#
#             logger.info(
#                 f"Processed {'doctor' if is_doctor else 'patient'} segment {segment_info['start']:.2f}-{segment_info['end']:.2f}")
#             for feature, value in features.items():
#                 if isinstance(value, np.ndarray):
#                     logger.info(f"  {feature} shape: {value.shape}, mean: {np.mean(value):.2f}")
#                 elif isinstance(value, (int, float)):
#                     logger.info(f"  {feature}: {value:.2f}")
#                 else:
#                     logger.info(f"  {feature}: {value}")
#
#             return result
#
#         except Exception as e:
#             logger.error(f"Error processing segment {segment_info['start']:.2f}-{segment_info['end']:.2f}: {str(e)}")
#             return None
#
#     async def __call__(self, signal, sampling_rate, diarization_data):
#         processed_segments = []
#
#         for segment in diarization_data:
#             try:
#                 start, end = segment['start'], segment['end']
#                 start_sample = int(start * sampling_rate)
#                 end_sample = int(end * sampling_rate)
#                 segment_audio = signal[start_sample:end_sample]
#
#                 processed_segment = await self.process_segment(segment_audio, sampling_rate, segment)
#                 if processed_segment:
#                     processed_segments.append(processed_segment)
#
#                     # Add sample to confidence estimator
#                     speaker_features = self.extract_speaker_features(segment_audio, sampling_rate)
#                     is_patient = segment['voice'] != 'doctor'
#                     self.confidence_estimator.add_sample(speaker_features, is_patient)
#
#                 # Update doctor features if the segment is labeled as doctor
#                 if segment['voice'] == 'doctor':
#                     features = await self.get_segment_features(segment_audio, sampling_rate)
#                     self.update_doctor_spectral_features(features['spectral_centroid'], features['spectral_flux'])
#             except Exception as e:
#                 logger.error(
#                     f"Error processing segment {segment.get('start', 'unknown')}-{segment.get('end', 'unknown')}: {str(e)}")
#
#         return processed_segments
#
#     def get_feature_ranges(self):
#         doctor_ranges = {
#             'pitch': (80, 180),
#             'speaking_rate': (2.5, 3.5),
#             'energy': (0.1, 0.8),
#             'jitter': (0, 0.015),
#             'shimmer': (0, 0.08),
#             'spectral_entropy': (5, 8),
#             'harmonics_to_noise_ratio': (10, 35),
#             'spectral_flatness': (0.1, 0.5),
#             'spectral_contrast': (20, 80),
#             'wavelet_energy': (0, 1000)
#         }
#
#         patient_ranges = {
#             'pitch': (50, 400),
#             'speaking_rate': (2, 3),
#             'energy': (0, 1),
#             'jitter': (0, 0.02),
#             'shimmer': (0, 0.1),
#             'spectral_entropy': (0, 10),
#             'harmonics_to_noise_ratio': (5, 30),
#             'spectral_flatness': (0, 1),
#             'spectral_contrast': (0, 100),
#             'wavelet_energy': (0, 1000)
#         }
#
#         return {'doctor': doctor_ranges, 'patient': patient_ranges}
#
#
# # def visualize_feature_ranges(feature_ranges, save_path=None):
# # fig, axes = plt.subplots(3, 3, figsize=(20, 20))
# # fig.suptitle("Feature Ranges for Doctor and Patient", fontsize=16)
# # axes = axes.flatten()
#
# # for i, (feature, ranges) in enumerate(feature_ranges['doctor'].items()):
# # ax = axes[i]
# # doctor_range = feature_ranges['doctor'][feature]
# # patient_range = feature_ranges['patient'][feature]
#
# # ax.set_title(feature)
# # ax.set_xlim(min(doctor_range[0], patient_range[0]), max(doctor_range[1], patient_range[1]))
#
# # ax.axvline(doctor_range[0], color='blue', linestyle='--', label='Doctor min')
# # ax.axvline(doctor_range[1], color='blue', linestyle='-', label='Doctor max')
# # ax.axvline(patient_range[0], color='red', linestyle='--', label='Patient min')
# # ax.axvline(patient_range[1], color='red', linestyle='-', label='Patient max')
#
# # ax.legend()
#
# # plt.tight_layout()
#
# # if save_path:
# # plt.savefig(save_path)
# # print(f"Feature ranges visualization saved to {save_path}")
#
# # plt.show()
#
# # def visualize_feature_stats(extractor, save_path=None):
# # feature_ranges = extractor.get_feature_ranges()
#
# # fig, axes = plt.subplots(3, 3, figsize=(20, 20))
# # fig.suptitle("Feature Values Over Time", fontsize=16)
# # axes = axes.flatten()
#
# # for i, feature in enumerate(feature_ranges['doctor'].keys()):
# # ax = axes[i]
# # ax.set_title(feature)
#
# # doctor_data = extractor.doctor_feature_stats.get(feature, [])
# # patient_data = extractor.patient_feature_stats.get(feature, [])
#
# # if doctor_data:
# # doctor_times, doctor_values = zip(*doctor_data)
# # ax.plot(doctor_times, doctor_values, 'b-', label='Doctor')
#
# # if patient_data:
# # patient_times, patient_values = zip(*patient_data)
# # ax.plot(patient_times, patient_values, 'r-', label='Patient')
#
# # ax.axhspan(feature_ranges['doctor'][feature][0], feature_ranges['doctor'][feature][1],
# # alpha=0.2, color='blue', label='Doctor range')
# # ax.axhspan(feature_ranges['patient'][feature][0], feature_ranges['patient'][feature][1],
# # alpha=0.2, color='red', label='Patient range')
#
# # ax.legend()
# # ax.set_xlabel('Time')
# # ax.set_ylabel('Value')
#
# # plt.tight_layout()
#
# # if save_path:
# # plt.savefig(save_path)
# # print(f"Feature stats visualization saved to {save_path}")
#
# # plt.show()
#
#
# # def visualize_wavelet_energy(self, wavelet_coeffs, segment_info, save_path=None):
# # plt.figure(figsize=(12, 6))
# # labels = ['Approximation'] + [f'Detail {i}' for i in range(len(wavelet_coeffs)-1, 0, -1)]
# # energies = [np.sum(np.square(c)) for c in wavelet_coeffs]
#
# # plt.bar(labels, energies)
# # plt.title(f"Wavelet Energy Distribution\nSegment: {segment_info['start']:.2f}-{segment_info['end']:.2f} ({segment_info['voice']})")
# # plt.xlabel("Wavelet Coefficient Levels")
# # plt.ylabel("Energy")
# # plt.yscale('log')  # Use log scale for better visualization
#
# # if save_path:
# # plt.savefig(save_path)
# # print(f"Wavelet energy visualization saved to {save_path}")
#
# # plt.close()
#
#
# # cross validate speaker identification
# def cross_validate_speaker(segment, doctor_embedding):
#     # Compare the segment embedding with the doctor embedding
#     similarity = cosine_similarity(segment['embedding'].reshape(1, -1), doctor_embedding.reshape(1, -1))[0][0]
#
#     # If the similarity is high, it might be the doctor speaking
#     if similarity > 0.8:  # Adjust this threshold as needed
#         segment['needs_review'] = True
#         logger.warning(f"Segment {segment['segment_start']:.2f}-{segment['segment_end']:.2f} "
#                        f"has high similarity to doctor embedding. May need review.")
#
#
# def save_progress(progress_file, processed_chunks):
#     """Save the progress of processed chunks."""
#     with open(progress_file, 'w') as f:
#         json.dump(processed_chunks, f)
#
#
# def load_progress(progress_file):
#     """Load the progress of processed chunks."""
#     if os.path.exists(progress_file):
#         with open(progress_file, 'r') as f:
#             return json.load(f)
#     return []
#
#
# def print_audio_metadata(file_path):
#     try:
#         # Run ffprobe command to get audio metadata
#         output = !ffprobe - i
#         "{file_path}" - hide_banner
#
#         # Print the metadata output
#         for line in output:
#             print(line)
#     except Exception as e:
#         logging.error(f"Error retrieving metadata for {file_path}: {e}")
#
#
# def extract_reference_datetime(file_path):
#     try:
#         # Extract from the filename
#         filename = os.path.basename(file_path)
#         # Remove the '._' prefix if it exists
#         if filename.startswith('._'):
#             filename = filename[2:]
#         date_str, time_str = filename.split()[0], filename.split()[1].split('-')[0]
#         datetime_str = f"{date_str} {time_str}"
#         reference_datetime = datetime.strptime(datetime_str, "%Y%m%d %H%M%S")
#         logger.info(f"Extracted reference datetime from filename for {file_path}: {reference_datetime}")
#         return reference_datetime
#     except Exception as e:
#         logger.warning(f"Failed to extract datetime from filename for {file_path}: {e}")
#
#     # If filename extraction fails, try metadata
#     try:
#         result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path],
#                                 capture_output=True, text=True)
#         metadata = json.loads(result.stdout)
#         creation_time = metadata.get('format', {}).get('tags', {}).get('creation_time')
#
#         if creation_time:
#             reference_datetime = datetime.strptime(creation_time, '%Y-%m-%dT%H:%M:%S.%fZ')
#             logger.info(f"Reference datetime for {file_path}: {reference_datetime}")
#             return reference_datetime
#         else:
#             logger.warning(f"Creation time not found in metadata for {file_path}")
#             return None
#     except Exception as e:
#         logger.error(f"Error extracting reference datetime for {file_path}: {e}")
#         return None
#
#
# # Speaker separator class using pyAudioAnalysis
# # class speakerSeparator:
# # def __init__(self):
# # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # def __call__(self, signal, sampling_rate):
# # Use silence removal with correct parameters
# # segments = aS.silence_removal(signal, sampling_rate, st_win=0.02, st_step=0.02, smooth_window=0.5, weight=0.5, plot=False)
# # dfDiarization = pd.DataFrame(segments, columns=["start", "end"])
# # return dfDiarization
#
#
# def apply_preemphasis(y, pre_emphasis=0.97):
#     return np.append(y[0], y[1:] - pre_emphasis * y[:-1])
#
#
# def apply_vad(y, sr, vad_window_size=0.03, vad_threshold=0.8):
#     try:
#         logger.info("Applying VAD")
#         vad = webrtcvad.Vad(3)  # Aggressive VAD mode
#         frames = librosa.util.frame(y, frame_length=int(vad_window_size * sr), hop_length=int(vad_window_size * sr))
#         vad_labels = [vad.is_speech(frame.astype(np.int16).tobytes(), sample_rate=sr) for frame in frames.T]
#         vad_segments = []
#         start_time = 0
#         for i, label in enumerate(vad_labels):
#             if label:
#                 if len(vad_segments) == 0 or vad_segments[-1][1] != i - 1:
#                     vad_segments.append([i, i])
#                 else:
#                     vad_segments[-1][1] = i
#         vad_segments = [(start * vad_window_size, end * vad_window_size) for start, end in vad_segments]
#         return vad_segments
#     except Exception as e:
#         logger.error(f"Error applying VAD: {e}")
#         raise
#
#         # def load_audio(audio_file_path, target_sr=16000):
#         # try:
#         # Load M4A file
#         # audio = AudioSegment.from_file(audio_file_path, format="m4a")
#
#         # Convert to numpy array
#         samples = np.array(audio.get_array_of_samples())
#
#         # Convert to float32
#         # samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
#
#         # Resample if necessary
#         # if audio.frame_rate != target_sr:
#         # samples = librosa.resample(y=samples, orig_sr=audio.frame_rate, target_sr=target_sr)
#
#         # return samples, target_sr
#     # except Exception as e:
#     # logger.error(f"Error loading audio file {audio_file_path}: {e}")
#     # return None, None
#
#
# def convert_to_wav_cmd(input_path, output_path):
#     try:
#         command = ['ffmpeg', '-y', '-i', input_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_path]
#         subprocess.run(command, check=True)
#         logging.info(f"Converted {input_path} to {output_path}")
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Failed to convert {input_path} to {output_path}: {e}")
#         raise
#
#
# def convert_to_wav(input_bucket_name, output_bucket_name, audio_file, audio_directory_path, output_dir):
#     try:
#         client = storage.Client()
#         input_bucket = client.get_bucket(input_bucket_name)
#         output_bucket = client.get_bucket(output_bucket_name)
#
#         # Remove the "._" prefix from the file name
#         file_name = os.path.basename(audio_file)
#         cleaned_file_name = file_name.replace("._", "", 1)
#         cleaned_file_path = os.path.join(audio_directory_path, cleaned_file_name)
#
#         blob = input_bucket.blob(cleaned_file_path)
#         if not blob.exists():
#             logger.error(f"Failed to find file {cleaned_file_path} in bucket {input_bucket_name}")
#             return None
#
#         logger.info(f"Found file {cleaned_file_path} in bucket {input_bucket_name}")
#
#         # Download to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_m4a:
#             blob.download_to_filename(temp_m4a.name)
#             logger.info(f"Downloaded {cleaned_file_path} to temporary file {temp_m4a.name}")
#
#             # Convert to WAV
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
#                 command = ['ffmpeg', '-y', '-i', temp_m4a.name, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
#                            temp_wav.name]
#                 subprocess.run(command, check=True)
#                 logger.info(f"Converted {temp_m4a.name} to {temp_wav.name}")
#
#                 # Upload the converted file to the output bucket
#                 wav_blob_name = os.path.join(output_dir, cleaned_file_name.replace('.m4a', '.wav'))
#                 wav_blob = output_bucket.blob(wav_blob_name)
#                 wav_blob.upload_from_filename(temp_wav.name)
#
#                 logger.info(f"Uploaded converted file to {output_bucket_name}/{wav_blob_name}")
#
#                 logger.info(f"Converted {cleaned_file_path} and uploaded to {output_bucket_name}/{wav_blob_name}")
#
#                 # Clean up temporary files
#                 os.unlink(temp_m4a.name)
#                 os.unlink(temp_wav.name)
#
#                 return wav_blob_name
#     except Exception as e:
#         logger.error(f"Failed to convert {audio_file} to WAV format: {e}")
#         return None
#
#
# # def load_diarization_json(json_bucket_name, json_path): # added 7.24
# # try:
# # client = storage.Client()
# # bucket = client.get_bucket(json_bucket_name)
# # blob = bucket.blob(json_path)
# # json_bytes = blob.download_as_bytes()
#
# # json_str = json_bytes.decode('utf-8')
# # data = json.loads(json_str)
#
# # processed_segments = []
# # if 'transcription' in data:
# # for segment in data['transcription']:
# # processed_segments.append({
# # 'start': segment['timestamp'][0],
# # 'end': segment['timestamp'][1],
# # 'speaker': segment.get('speaker', ''),
# # 'voice': segment.get('voice', ''),
# # 'text': segment.get('text', '')
# # })
#
# # if not processed_segments:
# # logger.warning(f"No diarized segments found in JSON: {json_path}")
# # return processed_segments
# # except Exception as e:
# # logger.error(f"Error loading diarized segments from {json_path}: {e}")
# # return None
#
# # logger = logging.getLogger(__name__)
#
# def load_audio(audio_file, target_sr):
#     logger.info(f"Attempting to load audio file: {audio_file}")
#
#     bucket_name = "private-management-files"
#     file_path = audio_file.replace(f"{bucket_name}/", "", 1)
#
#     logger.info(f"Bucket name: {bucket_name}")
#     logger.info(f"File path: {file_path}")
#
#     try:
#         client = storage.Client()
#         bucket = client.get_bucket(bucket_name)
#         blob = bucket.blob(file_path)
#
#         if not blob.exists():
#             logger.error(f"Audio file does not exist in bucket: {file_path}")
#             return None, None
#
#         audio_bytes = blob.download_as_bytes()
#         logger.info(f"Successfully downloaded audio file: {file_path}")
#
#         # Use pydub to read the audio data
#         audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="m4a")
#
#         # Convert to numpy array
#         samples = np.array(audio.get_array_of_samples())
#
#         # If stereo, take the mean of both channels
#         if audio.channels == 2:
#             samples = samples.reshape((-1, 2)).mean(axis=1)
#
#         # Get the sample rate
#         sr = audio.frame_rate
#
#         logger.info(f"Successfully read audio data. Shape: {samples.shape}, Sample rate: {sr}")
#
#         # Resample if necessary
#         if sr != target_sr:
#             logger.info(f"Resampling audio from {sr} Hz to {target_sr} Hz")
#             samples = librosa.resample(samples, sr, target_sr)
#             sr = target_sr
#
#         logger.info(f"Loaded audio file. Shape: {samples.shape}, Sample rate: {sr}")
#         return samples, sr
#
#     except Exception as e:
#         logger.error(f"Error loading audio file {audio_file}: {str(e)}")
#         return None, None
#
#
# def apply_spectral_gating(y, sr):
#     try:
#         S = librosa.stft(y)
#         S_magnitude, S_phase = np.abs(S), np.angle(S)
#         S_db = librosa.amplitude_to_db(S_magnitude, ref=np.max)
#         mean_db = np.mean(S_db, axis=1, keepdims=True)
#         threshold_db = mean_db - 20
#         mask = S_db > threshold_db
#         S_denoised = S_magnitude * mask
#         S_denoised = S_denoised * np.exp(1j * S_phase)
#         return librosa.istft(S_denoised)
#     except Exception as e:
#         logger.error(f"Error applying spectral gating: {e}")
#         raise
#
#
# def calculate_snr(original_audio, processed_audio):
#     try:
#         signal_power = np.mean(processed_audio ** 2)
#         noise_power = np.mean((original_audio[:len(processed_audio)] - processed_audio) ** 2)
#         snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
#         return snr_db
#     except Exception as e:
#         logger.error(f"Error calculating SNR: {e}")
#         raise
#
#
# def split_audio(y, sr, chunk_duration=300):
#     """Split the audio data into smaller chunks."""
#     chunk_length = int(chunk_duration * sr)
#     chunks = []
#
#     for i in range(0, len(y), chunk_length):
#         chunk = y[i:i + chunk_length]
#         chunks.append(chunk)
#
#     return chunks
#
#
# def save_progress(progress_file, processed_chunks):
#     """Save the progress of processed chunks."""
#     with open(progress_file, 'w') as f:
#         json.dump(processed_chunks, f)
#
#
# def load_progress(progress_file):
#     """Load the progress of processed chunks."""
#     if os.path.exists(progress_file):
#         with open(progress_file, 'r') as f:
#             return json.load(f)
#     return []
#
#
# def recurrence_matrix(time_series, epsilon):
#     """
#     Compute the recurrence matrix for a given time series.
#     """
#     distance_matrix = squareform(pdist(time_series.reshape(-1, 1), 'euclidean'))
#     recurrence_matrix = distance_matrix < epsilon
#     return recurrence_matrix
#
#
# def recurrence_rate(R):
#     """
#     Compute the recurrence rate (RR) from the recurrence matrix.
#     """
#     N = R.shape[0]
#     return np.sum(R) / (N * N)
#
#
# def determinism(R, l_min):
#     """
#     Compute the determinism (DET) from the recurrence matrix.
#     """
#     N = R.shape[0]
#     diagonal_histogram = np.zeros(N)
#     for i in range(1, N):
#         diagonal_histogram[i] = np.sum(np.diag(R, k=i))
#     det = np.sum(diagonal_histogram[l_min:]) / np.sum(diagonal_histogram)
#     return det
#
#
# def rqa_entropy(R, l_min):
#     """
#     Compute the RQA entropy from the recurrence matrix.
#     """
#     N = R.shape[0]
#     lengths = []
#     for i in range(1, N):
#         diag = np.diag(R, k=i)
#         lengths.extend([sum(1 for _ in group) for k, group in groupby(diag) if k == 1])
#     lengths = [l for l in lengths if l >= l_min]
#     if len(lengths) == 0:
#         return 0
#     hist, _ = np.histogram(lengths, bins=range(min(lengths), max(lengths) + 2))
#     probs = hist / np.sum(hist)
#     return -np.sum(probs * np.log(probs + 1e-10))
#
#
# # complexity O(M log M)
# @jit(nopython=True)
# def _count_neighbors(distances, r):
#     return np.sum(distances <= r, axis=1)
#
#
# def sample_entropy_numba(x, m=2, r=0.2):
#     N = len(x)
#     r *= np.std(x)
#
#     xm = np.array([x[i:i + m] for i in range(N - m + 1)])
#     xmp = np.array([x[i:i + m + 1] for i in range(N - m)])
#
#     tree = cKDTree(xm)
#     tree_mp = cKDTree(xmp)
#
#     distances = tree.query(xm, k=N - m + 1, p=np.inf)[0]
#     distances_mp = tree_mp.query(xmp, k=N - m, p=np.inf)[0]
#
#     B = np.sum(_count_neighbors(distances, r)) - (N - m + 1)
#     A = np.sum(_count_neighbors(distances_mp, r)) - (N - m)
#
#     return -np.log(A / B)
#
#
# def sample_entropy_chunked(x, m=2, r=0.2, chunk_size=1000):
#     N = len(x)
#     r *= np.std(x)
#
#     B = 0
#     A = 0
#
#     for i in range(0, N - m, chunk_size):
#         chunk_end = min(i + chunk_size, N - m)
#
#         xm_chunk = np.array([x[j:j + m] for j in range(i, chunk_end)])
#         xmp_chunk = np.array([x[j:j + m + 1] for j in range(i, chunk_end)])
#
#         tree = cKDTree(xm_chunk)
#         tree_mp = cKDTree(xmp_chunk)
#
#         B += np.sum(tree.count_neighbors(tree, r, p=np.inf)) - len(xm_chunk)
#         A += np.sum(tree_mp.count_neighbors(tree_mp, r, p=np.inf)) - len(xmp_chunk)
#
#     return -np.log(A / B)
#
#
# def approximate_entropy(x, m=2, r=0.2, N=None):
#     if N is None:
#         N = len(x)
#     r *= np.std(x)
#
#     def phi(m):
#         z = np.array([x[i:i + m] for i in range(N - m + 1)])
#         tree = cKDTree(z)
#         count = tree.count_neighbors(tree, r, p=np.inf)
#         return np.sum(np.log(count)) / (N - m + 1)
#
#     return phi(m) - phi(m + 1)
#
#
# def extract_sample_entropy(audio_segment, sr, method='numba', chunk_size=1000):
#     """
#     Extract sample entropy from audio segment using the specified method.
#     """
#     try:
#         # Normalize audio
#         audio_normalized = (audio_segment - np.mean(audio_segment)) / np.std(audio_segment)
#
#         # Calculate sample entropy based on the specified method
#         if method == 'numba':
#             sample_entropy = sample_entropy_numba(audio_normalized, m=2, r=0.2)
#         elif method == 'chunked':
#             sample_entropy = sample_entropy_chunked(audio_normalized, m=2, r=0.2, chunk_size=chunk_size)
#         elif method == 'approximate':
#             sample_entropy = approximate_entropy(audio_normalized, m=2, r=0.2)
#         else:
#             raise ValueError(f"Unknown method: {method}")
#
#         return sample_entropy
#     except Exception as e:
#         logger.error(f"Error extracting sample entropy: {e}")
#         return None
#
#
# def extract_pitch_variability(audio_data, sampling_rate, chunk_size=30):
#     try:
#         total_frames = len(audio_data)
#         chunk_frames = chunk_size * sampling_rate
#         pitches = []
#
#         for start in range(0, total_frames, chunk_frames):
#             end = min(start + chunk_frames, total_frames)
#             chunk = audio_data[start:end]
#             pitch, voiced_flag, voiced_probs = librosa.pyin(chunk, fmin=librosa.note_to_hz('C2'),
#                                                             fmax=librosa.note_to_hz('C7'))
#             pitches.extend(pitch)
#
#         # Remove unvoiced (NaN) entries
#         pitches = np.array(pitches)
#         pitches = pitches[~np.isnan(pitches)]
#         pitch_variability = np.std(pitches)  # Standard deviation as a measure of pitch variability
#         logger.info("Pitch variability extraction successful.")
#         return pitch_variability
#     except Exception as e:
#         logger.error(f"Error extracting pitch variability: {e}", exc_info=True)
#         return None
#
#
# def extract_wavelet_coefficients(audio_data, wavelet='db4', level=5):
#     try:
#         coeffs = pywt.wavedec(audio_data, wavelet, level=level)
#         coeffs_flattened = np.concatenate(coeffs)
#         logger.info("Wavelet coefficients extraction successful.")
#         return coeffs_flattened
#     except Exception as e:
#         logger.error(f"Error extracting wavelet coefficients: {e}", exc_info=True)
#         return None
#
#
# def validate_feature_range(feature_name, feature_value, expected_range):
#     if isinstance(feature_value, (list, np.ndarray)):
#         if np.any(feature_value < expected_range[0]) or np.any(feature_value > expected_range[1]):
#             logger.warning(
#                 f"Doctor feature '{feature_name}' with value {feature_value} is outside expected range {expected_range}")
#     else:
#         if feature_value < expected_range[0] or feature_value > expected_range[1]:
#             logger.warning(
#                 f"Doctor feature '{feature_name}' with value {feature_value} is outside expected range {expected_range}")
#
#
# def extract_advanced_features(audio_segment, sr):
#     features = {}
#     successfully_extracted_features = []
#
#     try:
#         # Pitch Variability
#         pitch_variability = extract_pitch_variability(audio_segment, sr)
#         if pitch_variability is not None:
#             features['pitch_variability'] = pitch_variability
#             successfully_extracted_features.append('pitch_variability')
#             logger.info(f"Extracted pitch variability: {pitch_variability}")
#
#         # VTLN
#         try:
#             features['vtln'] = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr).flatten()
#             successfully_extracted_features.append('vtln')
#             logger.info(f"Extracted VTLN: {vtln}")
#         except Exception as e:
#             logger.error(f"Error extracting VTLN: {e}")
#
#         # Emotion recognition features
#         try:
#             emotion = {
#                 'arousal': np.mean(librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr)),
#                 'valence': np.mean(librosa.feature.spectral_contrast(y=audio_segment, sr=sr))
#             }
#             features['emotion'] = emotion
#             successfully_extracted_features.append('emotion')
#             logger.info(f"Extracted emotion {emotion}")
#         except Exception as e:
#             logger.error(f"Error extracting emotion features: {e}")
#
#         # Pause and silence features
#         try:
#             intervals = librosa.effects.split(audio_segment, top_db=20)
#             pause_durations = [np.mean(np.diff(intervals))]
#             features['pauses'] = pause_durations
#             features['intervals'] = intervals
#             successfully_extracted_features.append('pauses')
#             logger.info(f"Extracted pauses: {pauses}")
#             successfully_extracted_features.append('intervals')
#             logger.info(f"Extracted intervals: {intervals}")
#         except Exception as e:
#             logger.error(f"Error extracting pause features: {e}")
#
#         # Formant frequencies and bandwidths
#         try:
#             lpc_coeffs = librosa.lpc(audio_segment, order=10)
#             roots = np.roots(lpc_coeffs)
#             roots = [r for r in roots if np.imag(r) >= 0]
#             formant_frequencies = sorted(np.abs(np.angle(roots)) * (sr / (2 * np.pi)))
#             features['formant_frequencies'] = formant_frequencies
#             formant_freq_ratio = formant_frequencies[0] / formant_frequencies[1] if len(formant_frequencies) > 1 else 0
#             features['formant_freq_ratio'] = formant_freq_ratio
#             successfully_extracted_features.append('formant_frequencies')
#             logger.info(f"Extracted formant frequencies: {formant_frequencies}")
#             successfully_extracted_features.append('formant_freq_ratio')
#             logger.info(f"Extracted formant frequency ratio: {formant_freq_ratio}")
#         except Exception as e:
#             logger.error(f"Error extracting formant frequencies and bandwidths: {e}")
#
#         # Spectral entropy and flatness combination
#         try:
#             spectral_entropy = entropy(np.abs(librosa.stft(audio_segment)))
#             spectral_flatness = librosa.feature.spectral_flatness(y=audio_segment)
#             spectral_irregularity = np.mean(spectral_entropy) + np.mean(spectral_flatness)
#             features['spectral_irregularity'] = spectral_irregularity
#             successfully_extracted_features.append('spectral_irregularity')
#             logger.info(f"Extracted spectral irregularity: {spectral_irregularity}")
#         except Exception as e:
#             logger.error(f"Error extracting spectral irregularity: {e}")
#
#         # Wavelet Coefficients
#         try:
#             wavelet_coeffs = extract_wavelet_coefficients(audio_segment)
#             if wavelet_coeffs is not None:
#                 features['wavelet_coefficients'] = wavelet_coeffs.tolist()  # Convert to list for JSON compatibility
#                 successfully_extracted_features.append('wavelet_coefficients')
#                 logger.info(f"Extracted wavelet coefficients: {wavelet_coefficients}")
#         except Exception as e:
#             logger.error(f"Error extracting wavelet coefficients: {e}", exc_info=True)
#
#         # Print successfully extracted features
#         logger.info(f"Successfully extracted advanced features: {successfully_extracted_features}")
#
#         # Validate feature ranges
#         validate_feature_range('energy', features.get('energy', 0), [0.1, 0.8])
#         validate_feature_range('jitter', features.get('jitter', 0), [0, 0.015])
#         validate_feature_range('shimmer', features.get('shimmer', 0), [0, 0.08])
#         validate_feature_range('spectral_entropy', features.get('spectral_entropy', np.array([])), [5, 8])
#         validate_feature_range('harmonics_to_noise_ratio', features.get('harmonics_to_noise_ratio', np.array([])),
#                                [10, 35])
#
#         return features
#     except Exception as e:
#         logger.error(f"Error in extract_advanced_features: {str(e)}", exc_info=True)
#         return {}
#
#
# def extract_features(audio_segment, sr, patient_id, segment_size=5 * 16000):  # 5seconds
#     # features = {}
#     features = {'patient_id': patient_id}
#     feature_times = {}
#
#     try:
#         logger.info(f"Extracting features for segment {patient_id}")
#
#         # Check if the segment is too short
#         minimum_samples = sr * 0.1  # Minimum 100 ms
#         if len(audio_segment) < minimum_samples:
#             logger.warning(
#                 f"Segment too short for feature extraction: {len(audio_segment)} samples, minimum {minimum_samples} samples required")
#             return None
#
#         # Process audio in smaller segments with 50% overlap
#         hop_length = segment_size // 2
#         for i in range(0, len(audio_segment) - segment_size + 1, hop_length):
#             segment = audio_segment[i:i + segment_size]
#
#         # Wavelet energy and coefficients
#         logger.info("Extracting wavelet energy and coefficients")
#         start_time = time.time()
#         wavelet = 'db4'
#         coefficients = pywt.wavedec(audio_segment, wavelet)
#         wavelet_energy = [np.sum(np.square(c)) for c in coefficients]
#         wavelet_coeffs_flat = np.concatenate(coefficients)
#         feature_times['wavelet_energy'] = time.time() - start_time
#
#         # Formant frequencies
#         logger.info("Extracting formant frequencies")
#         start_time = time.time()
#         lpc_coeffs = librosa.lpc(audio_segment, order=10)
#         roots = np.roots(lpc_coeffs)
#         roots = [r for r in roots if np.imag(r) >= 0]
#         formant_frequencies = np.abs(np.angle(roots)) * (sr / (2 * np.pi))
#         feature_times['formant_frequencies'] = time.time() - start_time
#
#         # Formant Bandwidths
#         logger.info("Extracting formant bandwidths")
#         start_time = time.time()
#         formant_bandwidths = -np.log(np.abs(roots)) * (sr / (2 * np.pi))
#         feature_times['formant_bandwidths'] = time.time() - start_time
#
#         # Peak-to-peak
#         logger.info("Extracting peak-to-peak")
#         start_time = time.time()
#         peak_to_peak = np.ptp(audio_segment)
#         feature_times['peak_to_peak'] = time.time() - start_time
#
#         # MFCCs
#         logger.info("Extracting MFCCs")
#         start_time = time.time()
#         n_mels = min(128, sr // 2)  # Adjust n_mels based on sr
#         mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13, n_mels=n_mels)
#         feature_times['mfccs'] = time.time() - start_time
#
#         # Delta MFCCs
#         logger.info("Extracting delta MFCCs")
#         start_time = time.time()
#         if mfccs.shape[1] >= 9:
#             delta_mfccs = librosa.feature.delta(mfccs)
#         else:
#             delta_mfccs = np.zeros_like(mfccs)
#         feature_times['delta_mfccs'] = time.time() - start_time
#
#         # Delta-Delta MFCCs
#         logger.info("Extracting delta-delta MFCCs")
#         start_time = time.time()
#         if mfccs.shape[1] >= 9:
#             delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
#         else:
#             delta_delta_mfccs = np.zeros_like(mfccs)
#         feature_times['delta_delta_mfccs'] = time.time() - start_time
#
#         # Spectral flux
#         logger.info("Extracting spectral flux")
#         start_time = time.time()
#         spectral_flux = librosa.onset.onset_strength(y=audio_segment, sr=sr)
#         feature_times['spectral_flux'] = time.time() - start_time
#
#         # Spectral rolloff
#         logger.info("Extracting spectral rolloff")
#         start_time = time.time()
#         spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr)
#         feature_times['spectral_rolloff'] = time.time() - start_time
#
#         # Spectral flatness
#         logger.info("Extracting spectral flatness")
#         start_time = time.time()
#         spectral_flatness = librosa.feature.spectral_flatness(y=audio_segment)
#         feature_times['spectral_flatness'] = time.time() - start_time
#
#         # Harmonics-to-noise ratio
#         logger.info("Extracting harmonics-to-noise ratio")
#         start_time = time.time()
#         hnr = librosa.effects.harmonic(y=audio_segment)
#         feature_times['hnr'] = time.time() - start_time
#
#         # Pitches
#         logger.info("Extracting pitches")
#         start_time = time.time()
#         pitches, _ = librosa.piptrack(y=audio_segment, sr=sr)
#         feature_times['pitches'] = time.time() - start_time
#
#         # Spectral centroid
#         logger.info("Extracting spectral centroid")
#         start_time = time.time()
#         spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
#         feature_times['spectral_centroid'] = time.time() - start_time
#
#         # Zero crossing rate
#         logger.info("Extracting zero crossing rate")
#         start_time = time.time()
#         zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_segment)
#         feature_times['zero_crossing_rate'] = time.time() - start_time
#
#         # Spectral entropy
#         logger.info("Extracting spectral entropy")
#         start_time = time.time()
#         spectral_entropy_value = entropy(np.abs(librosa.stft(audio_segment)))
#         feature_times['spectral_entropy'] = time.time() - start_time
#
#         # Spectral skewness
#         logger.info("Extracting spectral skewness")
#         start_time = time.time()
#         spectral_skewness = skew(np.abs(librosa.stft(audio_segment)).flatten())
#         feature_times['spectral_skewness'] = time.time() - start_time
#
#         # Spectral kurtosis
#         logger.info("Extracting spectral kurtosis")
#         start_time = time.time()
#         spectral_kurtosis = kurtosis(np.abs(librosa.stft(audio_segment)).flatten())
#         feature_times['spectral_kurtosis'] = time.time() - start_time
#
#         # STFT magnitude
#         logger.info("Extracting STFT magnitude")
#         start_time = time.time()
#         n_fft = min(2048, len(audio_segment))  # Adjust n_fft based on signal length
#         stft_data = librosa.stft(audio_segment, n_fft=n_fft)
#         stft_magnitude = np.abs(stft_data)
#         feature_times['stft_magnitude'] = time.time() - start_time
#
#         # Skewness
#         logger.info("Extracting skewness")
#         start_time = time.time()
#         skewness = skew(stft_magnitude.flatten())
#         feature_times['skewness'] = time.time() - start_time
#
#         # Kurtosis
#         logger.info("Extracting kurtosis")
#         start_time = time.time()
#         kurt = kurtosis(stft_magnitude.flatten())
#         feature_times['kurt'] = time.time() - start_time
#
#         # Spectral contrast
#         logger.info("Extracting spectral contrast")
#         start_time = time.time()
#         spectral_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sr)
#         feature_times['spectral_contrast'] = time.time() - start_time
#
#         # Line Spectral Pairs (LSP)
#         logger.info("Extracting line spectral pairs")
#         start_time = time.time()
#         lsp_coeffs = librosa.lpc(audio_segment, order=10)
#         feature_times['lsp'] = time.time() - start_time
#
#         # Bark Spectrogram
#         logger.info("Extracting bark spectrogram")
#         start_time = time.time()
#         n_mels = min(24, sr // 2)  # Adjust n_mels based on sr
#         bark_spec = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=n_mels, fmin=0, fmax=sr // 2)
#         feature_times['bark_spectrogram'] = time.time() - start_time
#
#         # Speech Rate
#         logger.info("Extracting speech rate")
#         start_time = time.time()
#         speech_rate = np.sum(librosa.effects.split(audio_segment, top_db=20)) / sr
#         feature_times['speech_rate'] = time.time() - start_time
#
#         # Speech Intensity
#         logger.info("Extracting speech intensity")
#         start_time = time.time()
#         intensity = np.mean(librosa.feature.rms(y=audio_segment))
#         feature_times['intensity'] = time.time() - start_time
#
#         # Speech Pauses
#         logger.info("Extracting speech pauses")
#         start_time = time.time()
#         intervals = librosa.effects.split(audio_segment, top_db=20)
#         pauses = [(end - start) / sr for start, end in intervals]
#         feature_times['pauses'] = time.time() - start_time
#
#         # Harmonicity
#         logger.info("Extracting harmonicity")
#         start_time = time.time()
#         harmonicity = librosa.effects.harmonic(y=audio_segment)
#         feature_times['harmonicity'] = time.time() - start_time
#
#         # Cepstral Peak Prominence (CPP)
#         logger.info("Extracting cepstral peak prominence")
#         start_time = time.time()
#         ceps = librosa.feature.mfcc(y=audio_segment, sr=sr)
#         ceps_db = librosa.amplitude_to_db(np.abs(librosa.stft(ceps)))
#         cpp = np.max(ceps_db, axis=1)
#         feature_times['cpp'] = time.time() - start_time
#
#         # Jitter
#         logger.info("Extracting jitter")
#         start_time = time.time()
#         pitches, magnitudes = librosa.core.piptrack(y=audio_segment, sr=sr)
#         pitches = pitches[magnitudes > np.median(magnitudes)]
#         jitter = np.mean(np.abs(np.diff(pitches)))
#         feature_times['jitter'] = time.time() - start_time
#
#         # Shimmer
#         logger.info("Extracting shimmer")
#         start_time = time.time()
#         y_harmonic, y_percussive = librosa.effects.hpss(audio_segment)
#         shimmer = np.mean(np.abs(np.diff(librosa.amplitude_to_db(y_harmonic))))
#         feature_times['shimmer'] = time.time() - start_time
#
#         # Pitch, energy, duration
#         logger.info("Extracting pitch, energy, and duration")
#         start_time = time.time()
#         pitch = librosa.yin(audio_segment, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
#         energy = np.sum(audio_segment ** 2) / len(audio_segment)
#         duration = len(audio_segment) / sr
#         feature_times['pitch_energy_duration'] = time.time() - start_time
#
#         # Mean
#         logger.info("Extracting mean")
#         start_time = time.time()
#         mean = np.mean(audio_segment)
#         feature_times['mean'] = time.time() - start_time
#
#         # Variance
#         logger.info("Extracting variance")
#         start_time = time.time()
#         variance = np.var(audio_segment)
#         feature_times['variance'] = time.time() - start_time
#
#         # Standard deviation
#         logger.info("Extracting standard deviation")
#         start_time = time.time()
#         std_dev = np.std(audio_segment)
#         feature_times['std_dev'] = time.time() - start_time
#
#         # Median
#         logger.info("Extracting median")
#         start_time = time.time()
#         median = np.median(audio_segment)
#         feature_times['median'] = time.time() - start_time
#
#         # Minimum value
#         logger.info("Extracting minimum value")
#         start_time = time.time()
#         min_val = np.min(audio_segment)
#         feature_times['min'] = time.time() - start_time
#
#         # Maximum value
#         logger.info("Extracting maximum value")
#         start_time = time.time()
#         max_val = np.max(audio_segment)
#         feature_times['max'] = time.time() - start_time
#
#         # Interquartile range
#         logger.info("Extracting interquartile range")
#         start_time = time.time()
#         iqr = np.percentile(audio_segment, 75) - np.percentile(audio_segment, 25)
#         feature_times['iqr'] = time.time() - start_time
#
#         # RMS
#         logger.info("Extracting RMS")
#         start_time = time.time()
#         rms = np.sqrt(np.mean(audio_segment ** 2))
#         feature_times['rms'] = time.time() - start_time
#
#         # Crest factor
#         logger.info("Extracting crest factor")
#         start_time = time.time()
#         crest_factor = np.max(np.abs(audio_segment)) / rms
#         feature_times['crest_factor'] = time.time() - start_time
#
#         # VTLN
#         # logger.info("Extracting VTLN")
#         # start_time = time.time()
#         # vtln = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr)
#         # feature_times['vtln'] = time.time() - start_time
#
#         # Hurst Exponent
#         logger.info("Extracting Hurst exponent")
#         start_time = time.time()
#         hurst_exp = nolds.hurst_rs(audio_segment)
#         features['hurst_exp'] = hurst_exp
#         feature_times['hurst_exp'] = time.time() - start_time
#
#         # Sample Entropy
#         logger.info("Extracting sample entropy")
#         sample_entropy = extract_sample_entropy(audio_segment, sr, method='chunked', chunk_size=1000)
#         features['sample_entropy'] = sample_entropy
#         feature_times['sample_entropy'] = time.time() - start_time
#
#         # Permutation Entropy
#         logger.info("Extracting permutation entropy")
#         start_time = time.time()
#         permutation_entropy = ant.perm_entropy(audio_segment)
#         features['permutation_entropy'] = permutation_entropy
#         feature_times['permutation_entropy'] = time.time() - start_time
#
#         logger.info(f"Finished extracting features for segment {patient_id}")
#
#         # Compile all features into a dictionary
#         features.update({
#             "patient_id": patient_id,
#             "wavelet_energy": wavelet_energy,
#             "wavelet_coefficients": wavelet_coeffs_flat,
#             "formant_frequencies": formant_frequencies,
#             "spectral_entropy": spectral_entropy_value,
#             "spectral_flux": spectral_flux,
#             "mfccs": mfccs,
#             "delta_mfccs": delta_mfccs,
#             "delta_delta_mfccs": delta_delta_mfccs,
#             "spectral_flatness": spectral_flatness,
#             "spectral_centroid": spectral_centroid,
#             "stft_magnitude": stft_magnitude,
#             "harmonics_to_noise_ratio": hnr,
#             "spectral_rolloff": spectral_rolloff,
#             "pitches": pitches,
#             "energy": energy,
#             "lsp": lsp_coeffs,
#             "bark_spectrogram": bark_spec,
#             "speech_rate": speech_rate,
#             "formant_bandwidths": formant_bandwidths,
#             # "intensity": intensity,
#             # "pauses": pause_durations,
#             "harmonicity": harmonicity,
#             "cpp": cpp,
#             "duration": duration,
#             "skewness": skewness,
#             "kurt": kurt,
#             "spectral_contrast": spectral_contrast,
#             "jitter": jitter,
#             "shimmer": shimmer,
#             "peak_to_peak": peak_to_peak,
#             "zero_crossing_rate": zero_crossing_rate,
#             "mean": mean,
#             "variance": variance,
#             "std_dev": std_dev,
#             "median": median,
#             "min_val": min_val,
#             "max_val": max_val,
#             "iqr": iqr,
#             "rms": rms,
#             "crest_factor": crest_factor,
#             # "vtln": vtln,
#             "hurst_exp": hurst_exp,
#             "sample_entropy": sample_entropy,
#             "permutation_entropy": permutation_entropy,
#             # "feature_times": feature_times
#         })
#
#         logger.info(f"Extracted features for segment {patient_id}: {len(features)} features")
#         return features
#     except Exception as e:
#         logger.error(f"Error extracting features: {e}", exc_info=True)
#         return None
#
#
# # logger = logging.getLogger(__name__)
#
#
# adaptive_thresholder = AdaptiveThresholder()  # call before classification
#
#
# def classify_anxiety(features, adaptive_thresholder):
#     logger.info("Entering classify_anxiety")
#     try:
#         score = 0
#         total_features = 8  # Update this if add or remove features
#
#         # Update thresholds with current data
#         adaptive_thresholder.update(features)
#
#         pitch_variability = features.get('pitch_variability', None)
#         jitter = features.get('jitter', None)
#         shimmer = features.get('shimmer', None)
#         spectral_entropy = features.get('spectral_entropy', [None])[0]
#         hnr = features.get('hnr', None)
#         emotion_arousal = features.get('emotion', {}).get('arousal', [None])[0]
#         hurst_exp = features.get('hurst_exp', None)
#         permutation_entropy = features.get('permutation_entropy', None)
#
#         if pitch_variability is not None:
#             score += scale_feature(pitch_variability, 0, 100) > safe_get_threshold(adaptive_thresholder,
#                                                                                    'pitch_variability', 0.5)
#         if jitter is not None:
#             score += scale_feature(jitter, 0, 2) > safe_get_threshold(adaptive_thresholder, 'jitter', 0.5)
#         if shimmer is not None:
#             score += scale_feature(shimmer, 0, 10) > safe_get_threshold(adaptive_thresholder, 'shimmer', 0.5)
#         if spectral_entropy is not None:
#             score += scale_feature(np.max(np.array([spectral_entropy])), 0, 1) > safe_get_threshold(
#                 adaptive_thresholder, 'spectral_entropy', 0.5)
#         if hnr is not None:
#             score += (1 - scale_feature(hnr, 0, 25)) > safe_get_threshold(adaptive_thresholder, 'hnr',
#                                                                           0.5)  # Lower values indicate anxiety
#         if emotion_arousal is not None:
#             score += scale_feature(np.max(np.array([emotion_arousal])), 0, 1) > safe_get_threshold(adaptive_thresholder,
#                                                                                                    'emotion_arousal',
#                                                                                                    0.5)
#         if hurst_exp is not None:
#             score += scale_feature(hurst_exp, 0, 1) > safe_get_threshold(adaptive_thresholder, 'hurst_exp', 0.5)
#         if permutation_entropy is not None:
#             score += (1 - scale_feature(permutation_entropy, 0, 1)) > safe_get_threshold(adaptive_thresholder,
#                                                                                          'permutation_entropy',
#                                                                                          0.5)  # Lower values indicate anxiety
#
#         score_ratio = score / total_features
#         anxiety_threshold = safe_get_threshold(adaptive_thresholder, 'anxiety',
#                                                0.6)  # Default to 60th percentile if not set
#         result = score_ratio > anxiety_threshold
#
#         logger.info(f"Anxiety classification result: {result}, Score: {score_ratio}, Threshold: {anxiety_threshold}")
#         return result
#
#     except Exception as e:
#         logger.error(f"Error in classify_anxiety: {e}")
#         result = False
#     logger.info("Exiting classify_anxiety")
#     return result
#
#
# def classify_bipolar_state(features, adaptive_thresholder):
#     logger.info("Entering classify_bipolar_state")
#     try:
#         manic_score = 0
#         depressive_score = 0
#         mixed_score = 0
#         total_features = 0  # Increment this for each feature used
#
#         # Update thresholds with current data
#         adaptive_thresholder.update(features)
#
#         # Sample Entropy
#         sample_entropy = features.get('sample_entropy')
#         if sample_entropy is not None:
#             total_features += 1
#             if sample_entropy > safe_get_threshold(adaptive_thresholder, 'sample_entropy_high', 0.75):
#                 manic_score += 1
#             elif sample_entropy < safe_get_threshold(adaptive_thresholder, 'sample_entropy_low', 0.25):
#                 depressive_score += 1
#             else:
#                 mixed_score += 0.5
#
#         # Speech Rate
#         speech_rate = features.get('speech_rate')
#         if speech_rate is not None:
#             total_features += 1
#             if speech_rate > safe_get_threshold(adaptive_thresholder, 'speech_rate_high', 0.75):
#                 manic_score += 1
#             elif speech_rate < safe_get_threshold(adaptive_thresholder, 'speech_rate_low', 0.25):
#                 depressive_score += 1
#             else:
#                 mixed_score += 0.5
#
#         # Pitch Variability
#         pitch_var = features.get('pitch_variability')
#         if pitch_var is not None:
#             total_features += 1
#             if pitch_var > safe_get_threshold(adaptive_thresholder, 'pitch_var_high', 0.75):
#                 manic_score += 1
#             elif pitch_var < safe_get_threshold(adaptive_thresholder, 'pitch_var_low', 0.25):
#                 depressive_score += 1
#             else:
#                 mixed_score += 0.5
#
#         # Spectral Entropy
#         spectral_entropy = features.get('spectral_entropy', [None])[0]
#         if spectral_entropy is not None:
#             total_features += 1
#             if spectral_entropy > safe_get_threshold(adaptive_thresholder, 'spectral_entropy_high', 0.75):
#                 manic_score += 1
#             elif spectral_entropy < safe_get_threshold(adaptive_thresholder, 'spectral_entropy_low', 0.25):
#                 depressive_score += 1
#             else:
#                 mixed_score += 0.5
#
#         # Energy
#         energy = features.get('energy')
#         if energy is not None:
#             total_features += 1
#             if energy > safe_get_threshold(adaptive_thresholder, 'energy_high', 0.75):
#                 manic_score += 1
#             elif energy < safe_get_threshold(adaptive_thresholder, 'energy_low', 0.25):
#                 depressive_score += 1
#             else:
#                 mixed_score += 0.5
#
#         # Hurst Exponent
#         hurst_exp = features.get('hurst_exp')
#         if hurst_exp is not None:
#             total_features += 1
#             if hurst_exp > safe_get_threshold(adaptive_thresholder, 'hurst_exp_high', 0.75):
#                 manic_score += 1
#             elif hurst_exp < safe_get_threshold(adaptive_thresholder, 'hurst_exp_low', 0.25):
#                 depressive_score += 1
#             else:
#                 mixed_score += 0.5
#
#         # Jitter
#         jitter = features.get('jitter')
#         if jitter is not None:
#             total_features += 1
#             if jitter > safe_get_threshold(adaptive_thresholder, 'jitter_high', 0.75):
#                 manic_score += 1
#             elif jitter < safe_get_threshold(adaptive_thresholder, 'jitter_low', 0.25):
#                 depressive_score += 1
#             else:
#                 mixed_score += 0.5
#
#         # Shimmer
#         shimmer = features.get('shimmer')
#         if shimmer is not None:
#             total_features += 1
#             if shimmer > safe_get_threshold(adaptive_thresholder, 'shimmer_high', 0.75):
#                 manic_score += 1
#             elif shimmer < safe_get_threshold(adaptive_thresholder, 'shimmer_low', 0.25):
#                 depressive_score += 1
#             else:
#                 mixed_score += 0.5
#
#         # Permutation Entropy
#         perm_entropy = features.get('permutation_entropy')
#         if perm_entropy is not None:
#             total_features += 1
#             if perm_entropy > safe_get_threshold(adaptive_thresholder, 'perm_entropy_high', 0.75):
#                 manic_score += 1
#             elif perm_entropy < safe_get_threshold(adaptive_thresholder, 'perm_entropy_low', 0.25):
#                 depressive_score += 1
#             else:
#                 mixed_score += 0.5
#
#         # Determine the state
#         manic_ratio = manic_score / total_features if total_features > 0 else 0
#         depressive_ratio = depressive_score / total_features if total_features > 0 else 0
#         mixed_ratio = mixed_score / total_features if total_features > 0 else 0
#
#         # Thresholds for different states
#         state_threshold = 0.6
#         mixed_threshold = 0.4
#         euthymic_threshold = 0.3
#
#         if manic_ratio > max(depressive_ratio, mixed_ratio, state_threshold):
#             result = "Manic"
#         elif depressive_ratio > max(manic_ratio, mixed_ratio, state_threshold):
#             result = "Depressive"
#         elif mixed_ratio > max(manic_ratio, depressive_ratio, mixed_threshold):
#             result = "Mixed"
#         elif max(manic_ratio, depressive_ratio, mixed_ratio) < euthymic_threshold:
#             result = "Euthymic"
#         else:
#             result = "Uncertain"
#
#         logger.info(
#             f"Bipolar state classification result: {result}, Manic: {manic_ratio:.2f}, Depressive: {depressive_ratio:.2f}, Mixed: {mixed_ratio:.2f}")
#         return result
#
#     except Exception as e:
#         logger.error(f"Error in classify_bipolar_state: {e}")
#         return "Error"
#
#     finally:
#         logger.info("Exiting classify_bipolar_state")
#
#
# def classify_schizophrenia_positive(features, adaptive_thresholder):
#     logger.info("Entering classify_schizophrenia_positive")
#     try:
#         score = 0
#         total_features = 6  # Updated total features count
#
#         # Update thresholds with current data
#         adaptive_thresholder.update(features)  # Changed this line
#
#         pitch = features.get('pitch', [None])[0]
#         formant_frequencies = features.get('formant_frequencies', [None])[0]
#         energy = features.get('energy', None)
#         hurst_exp = features.get('hurst_exp', None)
#         sample_entropy = features.get('sample_entropy', None)
#         permutation_entropy = features.get('permutation_entropy', None)
#
#         if pitch is not None:
#             score += scale_feature(np.max(np.array([pitch])), 0, 300) > safe_get_threshold(adaptive_thresholder,
#                                                                                            'pitch', 0.90)
#         if formant_frequencies is not None:
#             score += scale_feature(np.max(np.array([formant_frequencies])), 0, 1) > safe_get_threshold(
#                 adaptive_thresholder, 'formant_frequencies', 0.90)
#         if energy is not None:
#             score += scale_feature(energy, 0, 1) > safe_get_threshold(adaptive_thresholder, 'energy', 0.90)
#         if hurst_exp is not None:
#             score += scale_feature(hurst_exp, 0, 1) > safe_get_threshold(adaptive_thresholder, 'hurst_exp', 0.90)
#         if sample_entropy is not None:
#             score += (1 - scale_feature(sample_entropy, 0, 2)) > safe_get_threshold(adaptive_thresholder,
#                                                                                     'sample_entropy', 0.10)
#         if permutation_entropy is not None:
#             score += (1 - scale_feature(permutation_entropy, 0, 1)) > safe_get_threshold(adaptive_thresholder,
#                                                                                          'permutation_entropy', 0.10)
#
#         score_ratio = score / total_features
#         schizophrenia_positive_threshold = safe_get_threshold(adaptive_thresholder, 'schizophrenia_positive', 0.60)
#         result = score_ratio > schizophrenia_positive_threshold
#
#         logger.info(
#             f"Schizophrenia (positive) classification result: {result}, Score: {score_ratio}, Threshold: {schizophrenia_positive_threshold}")
#         return result
#
#     except Exception as e:
#         logger.error(f"Error in classify_schizophrenia_positive: {e}")
#         result = False
#     logger.info("Exiting classify_schizophrenia_positive")
#     return result
#
#
# def classify_schizophrenia_negative(features, adaptive_thresholder):
#     logger.info("Entering classify_schizophrenia_negative")
#     try:
#         score = 0
#         total_features = 5
#
#         # Update thresholds with current data
#         adaptive_thresholder.update(features)
#
#         formant_frequencies = features.get('formant_frequencies', [None])[0]
#         shimmer = features.get('shimmer', None)
#         hurst_exp = features.get('hurst_exp', None)
#         sample_entropy = features.get('sample_entropy', None)
#         permutation_entropy = features.get('permutation_entropy', None)
#
#         if formant_frequencies is not None:
#             score += (1 - scale_feature(np.min(np.array([formant_frequencies])), 0, 1)) > safe_get_threshold(
#                 adaptive_thresholder, 'formant_frequencies', 0.10)
#         if shimmer is not None:
#             score += scale_feature(shimmer, 0, 10) > safe_get_threshold(adaptive_thresholder, 'shimmer', 0.90)
#         if hurst_exp is not None:
#             score += (1 - scale_feature(hurst_exp, 1, 0)) > safe_get_threshold(adaptive_thresholder, 'hurst_exp', 0.10)
#         if sample_entropy is not None:
#             score += scale_feature(sample_entropy, 0, 2) > safe_get_threshold(adaptive_thresholder, 'sample_entropy',
#                                                                               0.90)
#         if permutation_entropy is not None:
#             score += scale_feature(permutation_entropy, 0, 1) > safe_get_threshold(adaptive_thresholder,
#                                                                                    'permutation_entropy', 0.90)
#
#         score_ratio = score / total_features
#         schizophrenia_negative_threshold = safe_get_threshold(adaptive_thresholder, 'schizophrenia_negative', 0.60)
#         result = score_ratio > schizophrenia_negative_threshold
#
#         logger.info(
#             f"Schizophrenia (negative) classification result: {result}, Score: {score_ratio}, Threshold: {schizophrenia_negative_threshold}")
#         return result
#
#     except Exception as e:
#         logger.error(f"Error in classify_schizophrenia_negative: {e}")
#         result = False
#     logger.info("Exiting classify_schizophrenia_negative")
#     return result
#
#
# def classify_ocd(features, adaptive_thresholder):
#     logger.info("Entering classify_ocd")
#     try:
#         score = 0
#         total_features = 7
#
#         # Update thresholds with current data
#         adaptive_thresholder.update(features)
#
#         pitch_variability = features.get('pitch_variability', None)
#         jitter = features.get('jitter', None)
#         shimmer = features.get('shimmer', None)
#         hnr = features.get('hnr', None)
#         hurst_exp = features.get('hurst_exp', None)
#         sample_entropy = features.get('sample_entropy', None)
#         permutation_entropy = features.get('permutation_entropy', None)
#
#         if pitch_variability is not None:
#             score += scale_feature(pitch_variability, 0, 100) > safe_get_threshold(adaptive_thresholder,
#                                                                                    'pitch_variability', 0.75)
#         if jitter is not None:
#             score += scale_feature(jitter, 0, 2) > safe_get_threshold(adaptive_thresholder, 'jitter', 0.75)
#         if shimmer is not None:
#             score += scale_feature(shimmer, 0, 10) > safe_get_threshold(adaptive_thresholder, 'shimmer', 0.75)
#         if hnr is not None:
#             score += (1 - scale_feature(hnr, 0, 25)) > safe_get_threshold(adaptive_thresholder, 'hnr', 0.25)
#         if hurst_exp is not None:
#             score += scale_feature(hurst_exp, 0, 1) > safe_get_threshold(adaptive_thresholder, 'hurst_exp', 0.75)
#         if sample_entropy is not None:
#             score += (1 - scale_feature(sample_entropy, 0, 2)) > safe_get_threshold(adaptive_thresholder,
#                                                                                     'sample_entropy', 0.25)
#         if permutation_entropy is not None:
#             score += (1 - scale_feature(permutation_entropy, 0, 1)) > safe_get_threshold(adaptive_thresholder,
#                                                                                          'permutation_entropy', 0.25)
#
#         score_ratio = score / total_features
#         ocd_threshold = safe_get_threshold(adaptive_thresholder, 'ocd', 0.60)
#         result = score_ratio > ocd_threshold
#
#         logger.info(f"OCD classification result: {result}, Score: {score_ratio}, Threshold: {ocd_threshold}")
#         return result
#
#     except Exception as e:
#         logger.error(f"Error in classify_ocd: {e}")
#         result = False
#     logger.info("Exiting classify_ocd")
#     return result
#
#
# def classify_ptsd(features, adaptive_thresholder):
#     logger.info("Entering classify_ptsd")
#     try:
#         score = 0
#         total_features = 5
#
#         # Update thresholds with current data
#         adaptive_thresholder.update(features)
#
#         spectral_entropy = features.get('spectral_entropy', [None])[0]
#         mfccs = features.get('mfccs', [None])[0]
#         hurst_exp = features.get('hurst_exp', None)
#         sample_entropy = features.get('sample_entropy', None)
#         permutation_entropy = features.get('permutation_entropy', None)
#
#         if spectral_entropy is not None:
#             score += scale_feature(np.max(np.array([spectral_entropy])), 0, 1) > safe_get_threshold(
#                 adaptive_thresholder, 'spectral_entropy', 0.90)
#         if mfccs is not None:
#             score += scale_feature(np.max(np.array([mfccs])), 0, 1) > safe_get_threshold(adaptive_thresholder, 'mfccs',
#                                                                                          0.90)
#         if hurst_exp is not None:
#             score += scale_feature(hurst_exp, 0, 1) > safe_get_threshold(adaptive_thresholder, 'hurst_exp', 0.90)
#         if sample_entropy is not None:
#             score += (1 - scale_feature(sample_entropy, 0, 2)) > safe_get_threshold(adaptive_thresholder,
#                                                                                     'sample_entropy', 0.10)
#         if permutation_entropy is not None:
#             score += (1 - scale_feature(permutation_entropy, 0, 1)) > safe_get_threshold(adaptive_thresholder,
#                                                                                          'permutation_entropy', 0.10)
#
#         score_ratio = score / total_features
#         ptsd_threshold = safe_get_threshold(adaptive_thresholder, 'ptsd', 0.60)
#         result = score_ratio > ptsd_threshold
#
#         logger.info(f"PTSD classification result: {result}, Score: {score_ratio}, Threshold: {ptsd_threshold}")
#         return result
#
#     except Exception as e:
#         logger.error(f"Error in classify_ptsd: {e}")
#         result = False
#     logger.info("Exiting classify_ptsd")
#     return result
#
#
# def classify_adhd(features, adaptive_thresholder):
#     logger.info("Entering classify_adhd")
#     try:
#         score = 0
#         total_features = 7
#
#         # Update thresholds with current data
#         adaptive_thresholder.update(features)
#
#         speech_rate = features.get('speech_rate', None)
#         wavelet_energy = features.get('wavelet_energy', [None])[0]
#         formant_frequencies = features.get('formant_frequencies', [None])[0]
#         spectral_flux = features.get('spectral_flux', [None])[0]
#         hurst_exp = features.get('hurst_exp', None)
#         sample_entropy = features.get('sample_entropy', None)
#         permutation_entropy = features.get('permutation_entropy', None)
#
#         if speech_rate is not None:
#             score += scale_feature(speech_rate, 0, 1) > safe_get_threshold(adaptive_thresholder, 'speech_rate', 0.90)
#         if wavelet_energy is not None:
#             score += scale_feature(np.max(np.array([wavelet_energy])), 0, 1) > safe_get_threshold(adaptive_thresholder,
#                                                                                                   'wavelet_energy',
#                                                                                                   0.90)
#         if formant_frequencies is not None:
#             score += scale_feature(np.max(np.array([formant_frequencies])), 0, 1) > safe_get_threshold(
#                 adaptive_thresholder, 'formant_frequencies', 0.90)
#         if spectral_flux is not None:
#             score += scale_feature(np.max(np.array([spectral_flux])), 0, 1) > safe_get_threshold(adaptive_thresholder,
#                                                                                                  'spectral_flux', 0.90)
#         if hurst_exp is not None:
#             score += scale_feature(hurst_exp, 0, 1) > safe_get_threshold(adaptive_thresholder, 'hurst_exp', 0.90)
#         if sample_entropy is not None:
#             score += (1 - scale_feature(sample_entropy, 0, 2)) > safe_get_threshold(adaptive_thresholder,
#                                                                                     'sample_entropy', 0.10)
#         if permutation_entropy is not None:
#             score += (1 - scale_feature(permutation_entropy, 0, 1)) > safe_get_threshold(adaptive_thresholder,
#                                                                                          'permutation_entropy', 0.10)
#
#         score_ratio = score / total_features
#         adhd_threshold = safe_get_threshold(adaptive_thresholder, 'adhd', 0.60)
#         result = score_ratio > adhd_threshold
#
#         logger.info(f"ADHD classification result: {result}, Score: {score_ratio}, Threshold: {adhd_threshold}")
#         return result
#
#     except Exception as e:
#         logger.error(f"Error in classify_adhd: {e}")
#         result = False
#     logger.info("Exiting classify_adhd")
#     return result
#
#
# def classify_depression(features, adaptive_thresholder):
#     logger.info("Entering classify_depression")
#     try:
#         score = 0
#         total_features = 9
#
#         # Update thresholds with current data
#         adaptive_thresholder.update(features)
#
#         speech_rate = features.get('speech_rate', None)
#         pitch_variability = features.get('pitch_variability', None)
#         spectral_entropy = features.get('spectral_entropy', [None])[0]
#         spectral_flatness = features.get('spectral_flatness', [None])[0]
#         jitter = features.get('jitter', None)
#         shimmer = features.get('shimmer', None)
#         hurst_exp = features.get('hurst_exp', None)
#         sample_entropy = features.get('sample_entropy', None)
#         permutation_entropy = features.get('permutation_entropy', None)
#
#         if speech_rate is not None:
#             score += (1 - scale_feature(speech_rate, 0, 1)) > safe_get_threshold(adaptive_thresholder, 'speech_rate',
#                                                                                  0.10)
#         if pitch_variability is not None:
#             score += (1 - scale_feature(pitch_variability, 0, 100)) > safe_get_threshold(adaptive_thresholder,
#                                                                                          'pitch_variability', 0.10)
#         if spectral_entropy is not None:
#             score += (1 - scale_feature(np.max(np.array([spectral_entropy])), 0, 1)) > safe_get_threshold(
#                 adaptive_thresholder, 'spectral_entropy', 0.10)
#         if spectral_flatness is not None:
#             score += (1 - scale_feature(np.max(np.array([spectral_flatness])), 0, 1)) > safe_get_threshold(
#                 adaptive_thresholder, 'spectral_flatness', 0.10)
#         if jitter is not None:
#             score += scale_feature(jitter, 0, 2) > safe_get_threshold(adaptive_thresholder, 'jitter', 0.90)
#         if shimmer is not None:
#             score += scale_feature(shimmer, 0, 10) > safe_get_threshold(adaptive_thresholder, 'shimmer', 0.90)
#         if hurst_exp is not None:
#             score += (1 - scale_feature(hurst_exp, 1, 0)) > safe_get_threshold(adaptive_thresholder, 'hurst_exp', 0.10)
#         if sample_entropy is not None:
#             score += scale_feature(sample_entropy, 0, 2) > safe_get_threshold(adaptive_thresholder, 'sample_entropy',
#                                                                               0.90)
#         if permutation_entropy is not None:
#             score += scale_feature(permutation_entropy, 0, 1) > safe_get_threshold(adaptive_thresholder,
#                                                                                    'permutation_entropy', 0.90)
#
#         score_ratio = score / total_features
#         depression_threshold = safe_get_threshold(adaptive_thresholder, 'depression', 0.60)
#         result = score_ratio > depression_threshold
#
#         logger.info(
#             f"Depression classification result: {result}, Score: {score_ratio}, Threshold: {depression_threshold}")
#         return result
#
#     except Exception as e:
#         logger.error(f"Error in classify_depression: {e}")
#         result = False
#     logger.info("Exiting classify_depression")
#     return result
#
#
# def assign_label_and_score(features, adaptive_thresholder):
#     logger.info("Starting classification for segment")
#
#     is_depression = classify_depression(features, adaptive_thresholder)
#     is_anxiety = classify_anxiety(features, adaptive_thresholder)
#     bipolar_state = classify_bipolar_state(features, adaptive_thresholder)
#     is_bipolar = bipolar_state != "Euthymic" and bipolar_state != "Uncertain"
#     is_schizophrenia_positive = classify_schizophrenia_positive(features, adaptive_thresholder)
#     is_schizophrenia_negative = classify_schizophrenia_negative(features, adaptive_thresholder)
#     is_ocd = classify_ocd(features, adaptive_thresholder)
#     is_ptsd = classify_ptsd(features, adaptive_thresholder)
#     is_adhd = classify_adhd(features, adaptive_thresholder)
#
#     diagnosis = assign_final_label(is_bipolar, is_depression, is_anxiety,
#                                    is_schizophrenia_positive, is_schizophrenia_negative,
#                                    is_ocd, is_ptsd, is_adhd)
#
#     logger.info(f"Classified as {diagnosis}")
#
#     raw_score, scaled_score = compute_weighted_clinical_score(features)
#
#     features['diagnosis'] = diagnosis
#     features['raw_clinical_score'] = raw_score
#     features['scaled_clinical_score'] = scaled_score
#
#     return features
#
#
# def assign_final_label(bipolar_state, is_depression, is_anxiety, is_schizophrenia_positive, is_schizophrenia_negative,
#                        is_ocd, is_ptsd, is_adhd):
#     conditions = [
#         (bipolar_state not in ["Euthymic", "Uncertain"], f"Bipolar Disorder ({bipolar_state})"),
#         (is_depression, "Depression"),
#         (is_anxiety, "Anxiety"),
#         (is_schizophrenia_positive, "Schizophrenia (Positive)"),
#         (is_schizophrenia_negative, "Schizophrenia (Negative)"),
#         (is_ocd, "OCD"),
#         (is_ptsd, "PTSD"),
#         (is_adhd, "ADHD")
#     ]
#
#     detected_conditions = [label for condition, label in conditions if condition]
#
#     if not detected_conditions:
#         return "Normative Baseline"
#     elif len(detected_conditions) == 1:
#         return detected_conditions[0]
#     else:
#         return f"Multiple Conditions: {', '.join(detected_conditions)}"
#
#     detected_conditions = [label for condition, label in conditions if condition]
#
#     if not detected_conditions:
#         return "Normative Baseline"
#     elif len(detected_conditions) == 1:
#         return detected_conditions[0]
#     else:
#         return f"Multiple Conditions: {', '.join(detected_conditions)}"
#
#
# def process_segments(segments):
#     adaptive_thresholder = AdaptiveThresholder()
#
#     for segment in segments:
#         features = extract_features(segment)  # feature extraction function
#
#         # Update thresholds
#         update_thresholds(adaptive_thresholder, features)
#
#         # Classify and assign label
#         labeled_segment = assign_label_and_score(features, adaptive_thresholder)
#
#         # Log or store results
#         log_results(segment, labeled_segment['diagnosis'], features)
#
#         # You might want to store the labeled_segment for further processing or analysis
#         segment.update(labeled_segment)
#
#     # Optionally, log final threshold state
#     logger.info("Final Adaptive Thresholds:")
#     for feature, threshold in adaptive_thresholder.get_all_thresholds().items():
#         logger.info(f"  {feature}: {threshold}")
#
#     return segments  # Return the processed segments if needed
#
#
# def log_results(segment, final_label, features):
#     logger.info(f"Segment {segment['start']:.2f}-{segment['end']:.2f}: {final_label}")
#     logger.debug(f"Features: {features}")
#
#
# def classify_healthy(features, adaptive_thresholder):
#     logger.info("Entering classify_healthy")
#     diagnosis = assign_label_and_score(features, adaptive_thresholder)['diagnosis']
#     result = diagnosis == "Normative Baseline"
#     logger.info("Exiting classify_healthy")
#     return result
#
#
# def compute_weighted_clinical_score(features):
#     weights = {
#         'pitch_mean': 1.2,
#         'jitter': 1.3,
#         'shimmer': 1.2,
#         'spectral_entropy': 1.1,
#         'spectral_flux': 1.1,
#         'spectral_centroid': 1.0,
#         'spectral_rolloff': 1.0,
#         'mfccs_mean': 0.9,
#         'delta_mfccs_mean': 0.9,
#         'delta_delta_mfccs_mean': 0.9,
#         'spectral_flatness': 1.0,
#         'formant_frequencies': 1.0,
#         'formant_bandwidths': 1.0,
#         'hnr': 1.0,
#         'speech_rate': 1.3,
#         'pauses': 1.2,
#         'harmonicity': 1.0,
#         'cpp': 1.1,
#         'duration': 1.0,
#         'skewness': 1.0,
#         'kurt': 1.0,
#         'spectral_contrast': 1.1,
#         'peak_to_peak': 1.0,
#         'zero_crossing_rate': 1.0,
#         'wavelet_energy': 1.0,
#         'wavelet_coefficients': 1.1,
#         'rms': 1.0,
#         'crest_factor': 1.0,
#         'pitch_variability': 1.1,
#         'vtln': 1.0,
#         'emotion': 1.1,
#         'hurst_exp': 1.2,
#         'sample_entropy': 1.2,
#         'permutation_entropy': 1.2
#     }
#
#     total_weight = 0
#     weighted_sum = 0
#
#     for feature, weight in weights.items():
#         if feature in features:
#             value = features[feature]
#             if isinstance(value, (int, float)):
#                 feature_value = value
#             elif isinstance(value, np.ndarray):
#                 feature_value = np.mean(value)
#             elif isinstance(value, dict):
#                 feature_value = np.mean(list(value.values()))
#             else:
#                 continue  # Skip unknown types
#
#             weighted_sum += feature_value * weight
#             total_weight += weight
#
#     if total_weight == 0:
#         return 0, 0  # Avoid division by zero
#
#     raw_score = weighted_sum / total_weight if total_weight > 0 else 0
#
#     # Scale the score between 0 and 10
#     min_possible_score = 0  # Adjust if necessary
#     max_possible_score = 1000  # Adjust based on expected maximum score
#     scaled_score = 10 * (raw_score - min_possible_score) / (max_possible_score - min_possible_score)
#     scaled_score = max(0, min(scaled_score, 10))  # Ensure score is between 0 and 10
#
#     # return weighted_sum / total_weight
#     return raw_score, scaled_score
#
#
# # def assign_label_and_score(row):
# # diagnosis = assign_diagnosis(row)
# # clinical_score = compute_weighted_clinical_score(row)
#
# # row['diagnosis'] = diagnosis
# # row['clinical_score'] = clinical_score
#
# # return row
#
#
# def adjust_features_based_on_medication(features, medication_info):
#     medication = medication_info.get('medication')
#     dosage = medication_info.get('dosage')
#     duration = medication_info.get('duration')
#
#     if medication == 'Vraylar':
#         # Example adjustment factors (these should be based on empirical data)
#         adjustment_factors = {
#             'pitch_variability': 0.9,  # Reduce by 10%
#             'speech_rate': 0.95,  # Reduce by 5%
#             'jitter': 0.85,  # Reduce by 15%
#             'shimmer': 0.9  # Reduce by 10%
#         }
#
#         for feature, factor in adjustment_factors.items():
#             if feature in features:
#                 features[feature] *= factor
#
#     # Add more medications if needed
#     return features
#
#     score = sum(weights[feature] * row[feature] for feature in weights if feature in row)  # climicla score only
#     return score
#
#     clinical_score = compute_weighted_clinical_score(row)
#     return pd.Series([diagnosis, clinical_score])
#
#
# # async def process_audio_file(audio_file, target_sr, bucket_name, chunk_duration=300, medication_info=None, reference_datetime=None, force_reprocess=False, doctor_embedding=None, separator=None):
# # try:
# # Load audio
# # y, sr = load_audio(audio_file, target_sr)
# # if y is None or sr is None:
# # logger.error(f"Failed to load audio file: {audio_file}")
# # return
#
# # Split audio into chunks
# # chunk_length = int(chunk_duration * sr)
# # chunks = [y[i:i+chunk_length] for i in range(0, len(y), chunk_length)]
#
# # logger.info(f"Audio file split into {len(chunks)} chunks")
#
# # Create a progress file
# # progress_file = f"{audio_file}_progress.json"
#
# # if force_reprocess:
# # logger.info("Force reprocessing all chunks.")
# # processed_chunks = []
# # else:
# # processed_chunks = load_progress(progress_file)
# # logger.info(f"Loaded progress file. Processed chunks: {processed_chunks}")
#
# # Process chunks
# # for i, chunk in enumerate(chunks):
# # if i in processed_chunks and not force_reprocess:
# # logger.info(f"Chunk {i} already processed. Skipping.")
# # continue
#
# # logger.info(f"Processing chunk {i}")
#
# # processed_segments = await process_chunk(chunk, sr, i, bucket_name,
# # medication_info, reference_datetime,
# # progress_file, doctor_embedding, separator)
#
# # if processed_segments:
# # processed_chunks.append(i)
# ##save_progress(progress_file, processed_chunks)
# # logger.info(f"Saved progress. Processed chunks: {processed_chunks}")
# # else:
# # logger.warning(f"No segments processed for chunk {i}")
#
# # logger.info(f"Finished processing chunk {i}")
#
# # logger.info(f"Processing {len(chunks)} chunks")
# # logger.info(f"Finished processing audio file: {audio_file}")
#
# # except Exception as e:
# # logger.error(f"Error processing audio file: {e}", exc_info=True)
#
#
# # updated 7.22 -to include diarized json files and performs the following:
# # It initializes a SpeakerConfidenceEstimator object at the beginning of the function.
# # For each processed segment, it extracts speaker features and adds them to the confidence estimator.
# # It cross-validates each segment against the doctor embedding using the cross_validate_speaker function.
# # After processing all chunks, it trains the confidence estimator.
# # It then estimates the speaker confidence for all segments and adds this information to each segment.
#
#
# def list_audio_files(bucket_name, prefix):
#     client = storage.Client()
#     bucket = client.get_bucket(bucket_name)
#     blobs = bucket.list_blobs(prefix=prefix)
#     return [blob.name for blob in blobs if blob.name.endswith(('.m4a', '.wav'))]
#
#
# def list_json_files(bucket_name, prefix):
#     client = storage.Client()
#     bucket = client.get_bucket(bucket_name)
#     blobs = bucket.list_blobs(prefix=prefix)
#     return [blob.name for blob in blobs if blob.name.endswith('.json')]
#
#
# def match_audio_json_files(audio_files, json_files):
#     matched_files = []
#     for audio_file in audio_files:
#         audio_id = os.path.splitext(os.path.basename(audio_file))[0]
#         matching_json = next((j for j in json_files if audio_id in j), None)
#         if matching_json:
#             matched_files.append((audio_file, matching_json))
#     return matched_files
#
#
# def list_matching_files_subset(json_bucket_name, json_directory_path, audio_bucket_name, audio_directory_path,
#                                subset_size=100):
#     client = storage.Client()
#     json_bucket = client.get_bucket(json_bucket_name)
#     audio_bucket = client.get_bucket(audio_bucket_name)
#
#     json_blobs = list(json_bucket.list_blobs(prefix=json_directory_path))
#     audio_blobs = list(audio_bucket.list_blobs(prefix=audio_directory_path))
#
#     json_files = [blob.name for blob in json_blobs if blob.name.endswith(".json")]
#     audio_files = [blob.name for blob in audio_blobs if
#                    blob.name.lower().endswith(".m4a") and not blob.name.startswith("._")]
#
#     logger.info(f"Found {len(json_files)} JSON files and {len(audio_files)} audio files.")
#
#     matching_files = []
#     for json_file in json_files:
#         json_id = extract_id(json_file)
#         if not json_id:
#             continue
#         for audio_file in audio_files:
#             audio_id = extract_id(audio_file)
#             if json_id == audio_id:
#                 matching_files.append((json_file, audio_file))
#                 break
#
#     matching_files = matching_files[:subset_size]
#
#     json_files_subset = [match[0] for match in matching_files]
#     audio_files_subset = [match[1] for match in matching_files]
#
#     return json_files_subset, audio_files_subset
#
#
# def extract_id(filename):
#     """Extract the ID between the dash and the extension in the filename."""
#     base = os.path.basename(filename)
#     try:
#         start_idx = base.index('-') + 1
#         end_idx = base.rindex('.')
#         return base[start_idx:end_idx]
#     except ValueError:
#         return None
#
#
# async def process_audio_file(y, sr, audio_bucket_name, json_file, reference_datetime, force_reprocess, doctor_embedding,
#                              separator, chunk_duration, adaptive_thresholder, checkpoint, result_bucket_path):
#     file_id = os.path.splitext(os.path.basename(json_file))[0]
#     checkpoint_blob_name = f"{result_bucket_path}/{file_id}_checkpoint.json"
#
#     try:
#         logger.info(f"Processing file: {json_file}")
#
#         # Load checkpoint if exists
#         client = storage.Client()
#         bucket = client.bucket(audio_bucket_name)
#         checkpoint_blob = bucket.blob(checkpoint_blob_name)
#
#         if checkpoint_blob.exists() and not force_reprocess:
#             checkpoint_data = json.loads(checkpoint_blob.download_as_text())
#             processed_segments = checkpoint_data['processed_segments']
#             start_index = checkpoint_data['next_segment_index']
#             logger.info(f"Resuming from checkpoint at segment {start_index}")
#         else:
#             processed_segments = []
#             start_index = 0
#
#         if not force_reprocess and check_if_processed(file_id):
#             logger.info(f"Skipping {json_file} as it has already been processed.")
#             return None
#
#         # Load diarization data
#         json_bucket_name = "processed-json-files-v2"
#         diarized_segments = load_diarization_json(json_bucket_name, json_file)
#
#         if not diarized_segments:
#             logger.error(f"No diarized segments found for file: {json_file}")
#             return None
#
#         total_segments = len(diarized_segments)
#         # Apply preemphasis to the entire audio
#         logger.info("Applying preemphasis to the entire audio file")
#         y_preemph = apply_preemphasis(y)
#
#         # Apply VAD to the entire audio
#         logger.info("Applying Voice Activity Detection (VAD) to the entire audio file")
#         vad_segments = apply_vad(y_preemph, sr)
#         logger.info(f"VAD detected {len(vad_segments)} speech segments")
#
#         if not vad_segments:
#             logger.warning("No speech detected by VAD. Processing will continue with diarized segments.")
#
#         for i, segment in enumerate(diarized_segments[start_index:], start=start_index):
#             try:
#                 logger.info(f"Processing segment {i + 1}/{total_segments}")
#
#                 # Skip if not a patient segment
#                 if segment['voice'] != 'patient':
#                     logger.info(f"Skipping non-patient segment {i + 1}")
#                     continue
#
#                 segment_start = int(segment['start'] * sr)
#                 segment_end = int(segment['end'] * sr)
#
#                 # Extract the full segment audio as indicated by the JSON file
#                 full_segment_audio = y_preemph[segment_start:segment_end]
#
#                 # Use VAD segments if available, otherwise use full segment
#                 if vad_segments:
#                     vad_segment_audio = []
#                     for vs_start, vs_end in vad_segments:
#                         vs_start_sample = int(vs_start * sr)
#                         vs_end_sample = int(vs_end * sr)
#                         if segment_start <= vs_start_sample < segment_end or segment_start < vs_end_sample <= segment_end:
#                             overlap_start = max(segment_start, vs_start_sample)
#                             overlap_end = min(segment_end, vs_end_sample)
#                             vad_segment_audio.extend(y_preemph[overlap_start:overlap_end])
#
#                     if vad_segment_audio:
#                         segment_audio = np.array(vad_segment_audio)
#                     else:
#                         segment_audio = full_segment_audio
#                 else:
#                     segment_audio = full_segment_audio
#
#                 # Apply spectral gating for noise reduction
#                 segment_audio_denoised = apply_spectral_gating(segment_audio, sr)
#
#                 # Extract features for the full segment
#                 features = extract_features(segment_audio_denoised, sr,
#                                             f"{os.path.splitext(json_file)[0]}_segment{i}_{segment['voice']}")
#
#                 if features:
#                     # Update adaptive thresholder with new features
#                     adaptive_thresholder.update(features)
#
#                     # Add segment information
#                     features['segment_start'] = segment['start']
#                     features['segment_end'] = segment['end']
#                     features['voice'] = segment['voice']
#
#                     # Assign label and score
#                     features = assign_label_and_score(features, adaptive_thresholder)
#
#                     logger.info(f"Patient Segment {i + 1}:")
#                     logger.info(f"  Diagnosis: {features.get('diagnosis', 'N/A')}")
#                     logger.info(f"  Raw Clinical Score: {features.get('raw_clinical_score', 'N/A')}")
#                     logger.info(f"  Scaled Clinical Score: {features.get('scaled_clinical_score', 'N/A')}")
#
#                     # Save features
#                     save_features_to_parquet(features, os.path.splitext(json_file)[0], f"segment{i}_{segment['voice']}",
#                                              audio_bucket_name, csv_file_path)
#
#                 processed_segments.append({**segment, 'features': features})
#
#                 # Save checkpoint after each segment
#                 checkpoint_data = {
#                     'processed_segments': [
#                         {k: numpy_to_python(v) for k, v in seg.items()}
#                         for seg in processed_segments
#                     ],
#                     'next_segment_index': i + 1
#                 }
#                 checkpoint_blob.upload_from_string(json.dumps(checkpoint_data, default=numpy_to_python))
#                 logger.info(f"Checkpoint saved at segment {i + 1}/{total_segments}")
#
#                 # Monitor system resources
#                 if i % 50 == 0:
#                     cpu_usage = monitor_cpu_usage()
#                     monitor_disk_io()
#                     monitor_network()
#                     logger.info(f"Resource check at segment {i + 1}/{total_segments}: CPU usage {cpu_usage}%")
#
#                     # Check if CPU usage is too high
#                     if cpu_usage > 90:
#                         logger.warning("CPU usage is very high. Pausing for 60 seconds.")
#                         await asyncio.sleep(60)
#
#             except Exception as e:
#                 logger.error(f"Error processing segment {i}: {str(e)}")
#
#         patient_segments = [seg for seg in processed_segments if seg['voice'] == 'patient']
#
#         logger.info(f"Processed file: {json_file}")
#         logger.info(f"Total segments: {len(processed_segments)}")
#         logger.info(f"Patient segments: {len(patient_segments)}")
#
#         # Mark as processed at the end of successful processing
#         mark_as_processed(file_id, audio_bucket_name, result_bucket_path, processed_files)
#
#         return [{
#             'filename': file_id,
#             'num_patient_segments': len(patient_segments),
#         }]
#
#     except Exception as e:
#         logger.error(f"Error processing audio file for JSON {json_file}: {e}")
#         return None
#
#     finally:
#         # Process completed, remove checkpoint file
#         checkpoint_blob.delete()
#         logger.info(f"Checkpoint file removed: {checkpoint_blob_name}")
#
#
# async def process_subset(audio_bucket_name, audio_directory_path, json_bucket_name, audio_files, json_files, target_sr,
#                          output_dir, patient_segments_dir, clinician_segments_dir):
#     successful_extractions = []
#     unsuccessful_extractions = []
#
#     for audio_file, json_file in zip(audio_files, json_files):
#         diarized_segments = load_diarized_segments(json_bucket_name, json_file)
#         if diarized_segments:
#             wav_file = os.path.join(output_dir, os.path.basename(audio_file).replace('.m4a', '.wav'))
#             result = await process_single_audio_file(audio_bucket_name, audio_directory_path, audio_file, json_file,
#                                                      diarized_segments, target_sr, output_dir, patient_segments_dir,
#                                                      clinician_segments_dir)
#             if result:
#                 successful_extractions.append(result)
#             else:
#                 unsuccessful_extractions.append((audio_file, json_file))
#         else:
#             logger.warning(f"Skipping {audio_file} due to missing or empty diarized segments")
#             unsuccessful_extractions.append((audio_file, json_file))
#
#     return successful_extractions, unsuccessful_extractions
#
#
# async def process_single_audio_file(y, sr, audio_bucket_name, json_file, medication_info, force_reprocess,
#                                     doctor_embedding, separator, chunk_duration):
#     try:
#         logger.info(f"Processing audio file for JSON: {json_file}")
#
#         # Apply preemphasis and VAD
#         y_preemph = apply_preemphasis(y)
#         vad_segments = apply_vad(y_preemph, sr)
#
#         # Load diarization data
#         diarized_segments = load_diarized_segments(json_bucket_name, json_file)
#         if not diarized_segments:
#             logger.warning(f"No diarized segments found for {json_file}")
#             return None
#
#         # Process segments using PatientFeatureExtractor
#         processed_segments = await separator(y_preemph, sr, diarized_segments, vad_segments)
#
#         # Separate patient and clinician segments
#         patient_segments = [seg for seg in processed_segments if seg['voice'] == 'patient']
#         clinician_segments = [seg for seg in processed_segments if seg['voice'] == 'doctor']
#
#         # Save patient segments and features
#         for i, segment in enumerate(patient_segments):
#             segment_audio = y_preemph[int(segment['start'] * sr):int(segment['end'] * sr)]
#
#             # Apply spectral gating for noise reduction
#             segment_audio_denoised = apply_spectral_gating(segment_audio, sr)
#
#             # Calculate SNR
#             snr = calculate_snr(segment_audio, segment_audio_denoised)
#             logger.info(f"SNR for patient segment {i + 1}: {snr:.2f} dB")
#
#             # Extract features
#             features = extract_features(segment_audio_denoised, sr, f"{os.path.splitext(json_file)[0]}_patient_{i + 1}")
#
#             if features:
#                 # Adjust features based on medication info if provided
#                 if medication_info:
#                     features = adjust_features_based_on_medication(features, medication_info)
#
#                 # Assign label and score
#                 features = assign_label_and_score(features)
#
#                 # Save features
#                 features_df = pd.DataFrame([features])
#                 features_path = f"analysis_results/diarized_patient_feature_analysis/patient/chunk_{i + 1}_patient_features.parquet"
#                 save_features_to_parquet(features, i, f"patient_{i + 1}", audio_bucket_name, csv_file_path)
#
#         # Save clinician segments (optional)
#         for i, segment in enumerate(clinician_segments):
#             segment_audio = y_preemph[int(segment['start'] * sr):int(segment['end'] * sr)]
#             segment_path = f"analysis_results/diarized_patient_feature_analysis/clinician/chunk_{i + 1}_clinician_segment.wav"
#             save_audio_to_bucket(audio_bucket_name, segment_audio, segment_path, sr)
#
#         return {
#             'filename': os.path.splitext(os.path.basename(json_file))[0],
#             'num_patient_segments': len(patient_segments),
#             'num_clinician_segments': len(clinician_segments)
#         }
#
#     except Exception as e:
#         logger.error(f"Error processing audio file for JSON {json_file}: {e}")
#         return None
#
#
# def extract_speaker_features(audio_segment, sr):
#     mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
#     return np.mean(mfccs, axis=1)
#
#
# def cross_validate_speaker(segment, doctor_embedding):
#     # Compare the segment embedding with the doctor embedding
#     similarity = cosine_similarity(segment['embedding'].reshape(1, -1), doctor_embedding.reshape(1, -1))[0][0]
#
#     # If the similarity is high, it might be the doctor speaking
#     if similarity > 0.8:  # Adjust this threshold as needed
#         segment['needs_review'] = True
#         logger.warning(f"Segment {segment['segment_start']:.2f}-{segment['segment_end']:.2f} "
#                        f"has high similarity to doctor embedding. May need review.")
#
#
# def review_classifications(results, bucket_name):
#     doctor_segments = [r for r in results if r['voice'] == 'doctor']
#     patient_segments = [r for r in results if r['voice'] == 'patient']
#     uncertain_segments = [r for r in results if r['voice'] == 'uncertain']
#
#     logger.info(f"Total segments: {len(results)}")
#     logger.info(f"Doctor segments: {len(doctor_segments)}")
#     logger.info(f"Patient segments: {len(patient_segments)}")
#     logger.info(f"Uncertain segments: {len(uncertain_segments)}")
#
#     # Review low confidence classifications
#     low_confidence = [r for r in results if r['confidence'] < 0.6]
#     logger.info(f"Low confidence classifications: {len(low_confidence)}")
#
#     for i, segment in enumerate(low_confidence[:5]):  # Review first 5 low confidence segments
#         logger.info(f"Low confidence segment {i + 1}: {segment}")
#         blob_path = f"analysis_results/marianna/{segment['voice']}/chunk_{segment['chunk_index']}_segment_{segment['segment_index']}.wav"
#         local_filename = f"low_confidence_segment_{i + 1}.wav"
#
#         user_input = input(f"Do you want to listen to low confidence segment {i + 1}? (y/n): ")
#         if user_input.lower() == 'y':
#             download_and_play_segment(bucket_name, blob_path, local_filename)
#
#         user_classification = input("Enter correct classification (doctor/patient/uncertain): ")
#         if user_classification in ['doctor', 'patient', 'uncertain']:
#             segment['voice'] = user_classification
#             logger.info(f"Updated classification for segment {i + 1} to {user_classification}")
#
#     # After manual review, update the counts
#     doctor_segments = [r for r in results if r['voice'] == 'doctor']
#     patient_segments = [r for r in results if r['voice'] == 'patient']
#     uncertain_segments = [r for r in results if r['voice'] == 'uncertain']
#
#     logger.info("Updated counts after manual review:")
#     logger.info(f"Doctor segments: {len(doctor_segments)}")
#     logger.info(f"Patient segments: {len(patient_segments)}")
#     logger.info(f"Uncertain segments: {len(uncertain_segments)}")
#
#     return results  # Return the potentially updated results
#
#
# def save_updated_results(updated_results, audio_file):
#     output_file = f"{audio_file}_updated_results.json"
#     with open(output_file, 'w') as f:
#         json.dump(updated_results, f)
#     logger.info(f"Saved updated results to {output_file}")
#
#
# def play_audio_segment(audio_data, sample_rate):
#     audio_data = (audio_data * 32767).astype(np.int16)
#     play_obj = sa.play_buffer(audio_data, 1, 2, sample_rate)
#     play_obj.wait_done()
#
#
# def validate_patient_segments(processed_segments, audio_file, sample_rate, num_samples=5):
#     y, sr = librosa.load(audio_file, sr=sample_rate)
#
#     samples = random.sample(processed_segments, min(num_samples, len(processed_segments)))
#
#     for segment in samples:
#         start_sample = int(segment['segment_start'] * sr)
#         end_sample = int(segment['segment_end'] * sr)
#         segment_audio = y[start_sample:end_sample]
#
#         print(f"Playing segment: {segment['segment_start']:.2f} - {segment['segment_end']:.2f}")
#         print(f"Transcription: {segment['text']}")
#         play_audio_segment(segment_audio, sr)
#
#         is_patient = input("Is this the patient speaking? (y/n): ").lower() == 'y'
#         if not is_patient:
#             print("Marking segment for review.")
#             segment['needs_review'] = True
#
#
# async def process_chunk(chunk, sr, chunk_index, bucket_name, medication_info, reference_datetime, progress_file,
#                         doctor_embedding, feature_extractor, chunk_diarization, chunk_start_time, confidence_estimator):
#     processed_segments = []
#     try:
#         logger.info(f"Starting to process chunk {chunk_index}")
#
#         if feature_extractor is None:
#             raise ValueError(
#                 "Feature extractor is None. Make sure it's properly initialized and passed to the function.")
#
#         # Process all segments
#         logger.info(f"Type of feature_extractor: {type(feature_extractor)}")
#         logger.info(f"Is feature_extractor callable: {callable(feature_extractor)}")
#
#         # all_segments = await feature_extractor(chunk, sr, chunk_diarization)
#         all_segments = feature_extractor(chunk, sr, chunk_diarization)
#
#         logger.info(f"Feature extraction completed for chunk {chunk_index}. Number of segments: {len(all_segments)}")
#
#         for j, segment in enumerate(all_segments):
#             try:
#                 logger.info(f"Processing segment {j} in chunk {chunk_index}")
#
#                 start = segment['start']
#                 end = segment['end']
#                 start_sample = int(start * sr)
#                 end_sample = int(end * sr)
#
#                 # Check if the segment has a valid length
#                 if end_sample <= start_sample or (
#                         end_sample - start_sample) < sr * 0.1:  # Skip segments shorter than 100ms
#                     logger.warning(f"Skipping segment {j} in chunk {chunk_index} due to invalid or short length")
#                     continue
#
#                 segment_audio = chunk[start_sample:end_sample]
#
#                 # Extract speaker features for confidence estimation
#                 speaker_features = extract_speaker_features(segment_audio, sr)
#
#                 # Estimate confidence
#                 logger.info(f"Estimating confidence for segment {j}")
#                 confidence = confidence_estimator.estimate_confidence(speaker_features)
#
#                 # Determine segment type
#                 logger.info(f"Determining segment type for segment {j}")
#                 voice = segment.get('voice', 'unknown')  # Use 'unknown' if 'voice' key is missing
#                 if voice == "patient" and confidence >= 0.7:
#                     segment_type = "patient"
#                 elif voice == "doctor" and confidence < 0.3:
#                     segment_type = "doctor"
#                 else:
#                     # If voice is unknown or confidence is ambiguous, use a more sophisticated method
#                     segment_type = determine_segment_type(segment, confidence, doctor_embedding)
#
#                 # Apply spectral gating for noise reduction
#                 logger.info(f"Applying spectral gating for segment {j}")
#                 processed_audio = apply_spectral_gating(segment_audio, sr)
#
#                 # Calculate SNR
#                 logger.info(f"Calculating SNR for segment {j}")
#                 snr = calculate_snr(segment_audio, processed_audio)
#                 logger.info(f"SNR for segment {j} in chunk {chunk_index}: {snr:.2f} dB")
#
#                 # Add common information
#                 segment.update({
#                     'segment_number': j + 1,
#                     'segment_start': start + chunk_start_time,
#                     'segment_end': end + chunk_start_time,
#                     'segment_duration': end - start,
#                     'snr': snr,
#                     'segment_type': segment_type,
#                     'confidence_score': confidence,
#                     'voice': voice  # Include the voice field, even if it's 'unknown'
#                 })
#
#                 if reference_datetime is not None:
#                     segment_datetime = reference_datetime + timedelta(seconds=start + chunk_start_time)
#                     segment['segment_datetime'] = segment_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')
#                 else:
#                     segment['segment_datetime'] = None
#
#                 if medication_info:
#                     segment = adjust_features_based_on_medication(segment, medication_info)
#
#                 # Save audio segment to GCP bucket
#                 segment_path = f"analysis_results/marianna/{segment_type}/chunk_{chunk_index}_segment_{j + 1}.wav"
#                 upload_to_bucket(bucket_name, processed_audio, segment_path, content_type='audio/wav')
#                 logger.info(f"Saved segment {j} in chunk {chunk_index} to {segment_path} in bucket {bucket_name}")
#
#                 # Assign label and score
#                 segment = assign_label_and_score(segment)
#                 logger.info(f"Assigned label and score for segment {j} in chunk {chunk_index}: "
#                             f"{segment.get('diagnosis', 'N/A')}, Raw: {segment.get('raw_clinical_score', 'N/A')}, "
#                             f"Scaled: {segment.get('scaled_clinical_score', 'N/A')}")
#
#                 # Save features to parquet
#                 save_features_to_parquet(segment, chunk_index, j, bucket_name, csv_file_path)
#
#                 processed_segments.append(segment)
#                 logger.info(f"Processed segment {j} in chunk {chunk_index}: "
#                             f"Type: {segment_type}, Voice: {voice}, Confidence: {confidence:.2f}, "
#                             f"Diagnosis: {segment.get('diagnosis', 'N/A')}, "
#                             f"Raw Score: {segment.get('raw_clinical_score', 'N/A')}, "
#                             f"Scaled Score: {segment.get('scaled_clinical_score', 'N/A')}")
#
#                 # Add sample to confidence estimator for future training
#                 confidence_estimator.add_sample(speaker_features, segment_type == 'patient')
#
#             except Exception as e:
#                 logger.error(f"Error processing segment {j} in chunk {chunk_index}: {e}", exc_info=True)
#
#         logger.info(f"Finished processing chunk {chunk_index}. Total segments processed: {len(processed_segments)}")
#
#     except Exception as e:
#         logger.error(f"Error processing chunk {chunk_index}: {e}", exc_info=True)
#
#     return processed_segments
#
#
# def determine_segment_type(segment, confidence, doctor_embedding):
#     # implement a more sophisticated
#     # method of determining the segment type when the voice is unknown or ambiguous.
#     # This could involve comparing the segment's embedding with the doctor_embedding,
#     # analyzing the content of the speech, or using other available features.
#
#     if confidence > 0.5:
#         return "patient"
#     elif 'embedding' in segment:
#         similarity = cosine_similarity(segment['embedding'].reshape(1, -1), doctor_embedding.reshape(1, -1))[0][0]
#         if similarity > 0.8:
#             return "doctor"
#
#     return "needs_review"
#
#
# def verify_timestamp_alignment(audio_file, json_data, sample_rate):
#     audio_duration = librosa.get_duration(filename=audio_file)
#     json_duration = max(segment['end'] for segment in json_data)
#
#     tolerance = 1.0  # 1 second tolerance
#
#     if abs(audio_duration - json_duration) > tolerance:
#         logger.warning(f"Potential misalignment detected: "
#                        f"Audio duration: {audio_duration:.2f}s, "
#                        f"JSON duration: {json_duration:.2f}s")
#         return False
#
#     # Verify a few random segments
#     for _ in range(5):
#         segment = random.choice(json_data)
#         start_sample = int(segment['start'] * sample_rate)
#         end_sample = int(segment['end'] * sample_rate)
#         segment_audio = \
#         librosa.load(audio_file, sr=sample_rate, offset=segment['start'], duration=segment['end'] - segment['start'])[0]
#
#         if len(segment_audio) != (end_sample - start_sample):
#             logger.warning(f"Segment length mismatch for {segment['start']:.2f}-{segment['end']:.2f}")
#             return False
#
#     return True
#
#
# # async def process_audio_file(audio_file, target_sr, bucket_name, chunk_duration=300, medication_info=None, reference_datetime=None, force_reprocess=False, doctor_embedding=None):
# # try:
# # Load audio
# # y, sr = load_audio(audio_file, target_sr)
# # if y is None or sr is None:
# # logger.error(f"Failed to load audio file: {audio_file}")
# # return
#
# # Split audio into chunks
# # chunks = split_audio(y, sr, chunk_duration)
#
# # logger.info(f"Audio file split into {len(chunks)} chunks")
#
# # Create a progress file
# # progress_file = f"{audio_file}_progress.json"
#
# # if force_reprocess:
# # logger.info("Force reprocessing all chunks.")
# # processed_chunks = []
# # else:
# # processed_chunks = load_progress(progress_file)
# # logger.info(f"Loaded progress file. Processed chunks: {processed_chunks}")
#
# # Initialize ProviderPatientSeparator
# # separator = ProviderPatientSeparator(doctor_embedding)
#
# # Process chunks
# # for i, chunk in enumerate(chunks):
# # if i in processed_chunks and not force_reprocess:
# # logger.info(f"Chunk {i} already processed. Skipping.")
# # continue
#
# # logger.info(f"Processing chunk {i}")
#
# # processed_segments = await process_chunk(chunk, sr, i, bucket_name, medication_info, reference_datetime, progress_file, separator)
#
# # if processed_segments:
# # processed_chunks.append(i)
# # save_progress(progress_file, processed_chunks)
# # logger.info(f"Saved progress. Processed chunks: {processed_chunks}")
# # else:
# # logger.warning(f"No segments processed for chunk {i}")
#
# # logger.info(f"Finished processing chunk {i}")
#
# # logger.info(f"Finished processing audio file: {audio_file}")
#
# # except Exception as e:
# # logger.error(f"Error processing audio file: {e}", exc_info=True)
#
# # def review_classifications(results):
# # doctor_segments = [r for r in results if r['voice'] == 'doctor']
# # patient_segments = [r for r in results if r['voice'] == 'patient']
# # uncertain_segments = [r for r in results if r['voice'] == 'uncertain']
# ##
# # logger.info(f"Total segments: {len(results)}")
# ###logger.info(f"Doctor segments: {len(doctor_segments)}")
# # logger.info(f"Patient segments: {len(patient_segments)}")
# # logger.info(f"Uncertain segments: {len(uncertain_segments)}")
#
# ## Review low confidence classifications
# # low_confidence = [r for r in results if r['confidence'] < 0.6]
# ##logger.info(f"Low confidence classifications: {len(low_confidence)}")
# # for segment in low_confidence[:10]:  # Review first 10 low confidence segments
# # logger.info(f"Low confidence segment: {segment}")
#
#
# def upload_to_bucket(bucket_name, data, destination_blob_name, content_type='application/octet-stream'):
#     """Uploads data to the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     # Convert data to bytes if it's not already
#     if isinstance(data, str):
#         data = data.encode('utf-8')
#     elif not isinstance(data, bytes):
#         data = bytes(data)
#
#     blob.upload_from_string(data, content_type=content_type)
#
#     print(f"Data uploaded to {destination_blob_name}.")
#
#
# # def save_features_to_parquet(patient_features, clinician_features, uncertain_features, chunk_index, file_count, bucket_name):
# # blob_names = []
#
# # def save_features(features, file_path, blob_path, speaker_type):
# # df = pd.DataFrame([features])  # Convert single feature dictionary to a DataFrame
# # df.to_parquet(file_path, index=False)
# # upload_to_bucket(bucket_name, file_path, blob_path)
# # logger.info(f"Saved {speaker_type} segment to {blob_path} in bucket {bucket_name}")
# # logger.info(f"{speaker_type.capitalize()} segment includes segment number: {features['segment_number']}")
# # logger.info(f"{speaker_type.capitalize()} segment includes datetime: {features['segment_datetime']}")
# # os.remove(file_path)  # Remove the local file after uploading
#
# # if patient_features:
# # patient_parquet_file = f"chunk_{chunk_index}_patient_features_{file_count}.parquet"
# # patient_parquet_blob_path = f"analysis_results/marianna/patient/{patient_parquet_file}"
# # patient_df = pd.DataFrame(patient_features)
# # patient_df.to_parquet(patient_parquet_file, index=False)
# # logger.info(f"Added {len(patient_features)} patient segments to parquet file {patient_parquet_file}")
# # upload_to_bucket(bucket_name, patient_parquet_file, patient_parquet_blob_path)
# # logger.info(f"Uploaded complete parquet file with {len(patient_features)} patient segments to {patient_parquet_blob_path} in bucket {bucket_name}")
# # os.remove(patient_parquet_file)  # Remove the local file after uploading
# # blob_names.append(patient_parquet_blob_path)
# # else:
# # logger.info("No patient features to save.")
#
# # if clinician_features:
# # clinician_parquet_file = f"chunk_{chunk_index}_clinician_features_{file_count}.parquet"
# # clinician_parquet_blob_path = f"analysis_results/marianna/clinician/{clinician_parquet_file}"
# # clinician_df = pd.DataFrame(clinician_features)
# # clinician_df.to_parquet(clinician_parquet_file, index=False)
# # logger.info(f"Added {len(clinician_features)} clinician segments to parquet file {clinician_parquet_file}")
# # upload_to_bucket(bucket_name, clinician_parquet_file, clinician_parquet_blob_path)
# # logger.info(f"Uploaded complete parquet file with {len(clinician_features)} clinician segments to {clinician_parquet_blob_path} in bucket {bucket_name}")
# # os.remove(clinician_parquet_file)  # Remove the local file after uploading
# # blob_names.append(clinician_parquet_blob_path)
# # else:
# # logger.info("No clinician features to save.")
#
# # if uncertain_features:
# # uncertain_parquet_file = f"chunk_{chunk_index}_uncertain_features_{file_count}.parquet"
# # uncertain_parquet_blob_path = f"analysis_results/marianna/uncertain/{uncertain_parquet_file}"
# # uncertain_df = pd.DataFrame(uncertain_features)
# # uncertain_df.to_parquet(uncertain_parquet_file, index=False)
# # logger.info(f"Added {len(uncertain_features)} uncertain segments to parquet file {uncertain_parquet_file}")
# # upload_to_bucket(bucket_name, uncertain_parquet_file, uncertain_parquet_blob_path)
# # logger.info(f"Uploaded complete parquet file with {len(uncertain_features)} uncertain segments to {uncertain_parquet_blob_path} in bucket {bucket_name}")
# ##os.remove(uncertain_parquet_file)  # Remove the local file after uploading
# # blob_names.append(uncertain_parquet_blob_path)
# # else:
# # logger.info("No uncertain features to save.")
#
# # return blob_names
#
#
# # flatten nested dictionaries
# def flatten_dict(d, parent_key='', sep='_'):
#     items = []
#     for k, v in d.items():
#         new_key = f"{parent_key}{sep}{k}" if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         elif isinstance(v, (list, np.ndarray)):
#             if len(v) > 0:
#                 if isinstance(v[0], (list, np.ndarray)):
#                     # For 2D arrays, convert to string representation
#                     items.append((new_key, str(v)))
#                 else:
#                     # For 1D arrays, keep as is
#                     items.append((new_key, v))
#         else:
#             items.append((new_key, v))
#     return dict(items)
#
#
# def transform_patient_id(patient_id):
#     """Transform patient_id by replacing _segment*_xxx with .json"""
#     return re.sub(r'_segment\d+_\w+$', '.json', patient_id)
#
#
# # join patient disorder families and diagnoses to dataframe , then to parquet file
#
#
# def save_features_to_parquet(features, chunk_index, segment_index, bucket_name, csv_file_path):
#     blob_names = []
#
#     segment_type = features.get('segment_type', features.get('voice', 'unknown'))
#     parquet_file = f"chunk_{chunk_index}_segment_{segment_index}_{segment_type}_features.parquet"
#     # parquet_blob_path = f"analysis_results/marianna/{segment_type}/{parquet_file}"
#     parquet_blob_path = f"analysis_results/diarized_patient_feature_analysis/{segment_type}/{parquet_file}"
#
#     try:
#         # Flatten nested dictionaries and handle multi-dimensional arrays
#         flat_features = flatten_dict(features)
#
#         # Convert all numpy arrays to lists
#         for key, value in flat_features.items():
#             if isinstance(value, np.ndarray):
#                 flat_features[key] = value.tolist()
#
#         # Transform patient_id if it exists
#         if 'patient_id' in flat_features:
#             flat_features['patient_id'] = transform_patient_id(flat_features['patient_id'])
#
#         # Ensure required fields are present without overwriting existing data
#         required_fields = ['raw_clinical_score', 'scaled_clinical_score', 'confidence_score', 'segment_type',
#                            'diagnosis', 'Disorder Families']
#         for field in required_fields:
#             if field not in flat_features:
#                 logger.warning(f"{field} not found for segment {segment_index}")
#                 if field in ['raw_clinical_score', 'scaled_clinical_score', 'confidence_score']:
#                     flat_features[field] = 0.0
#                 elif field == 'segment_type':
#                     flat_features[field] = segment_type
#                 elif field == 'diagnosis':
#                     flat_features[field] = 'Unknown'
#                 else:
#                     flat_features[field] = 'N/A'
#
#         df = pd.DataFrame([flat_features])
#
#         # Check if 'patient_id' is in the features DataFrame
#         if 'patient_id' not in df.columns:
#             logger.error(f"'patient_id' column is missing from the features DataFrame for segment {segment_index}")
#             return blob_names
#
#         # Log the patient_id from the features DataFrame
#         logger.info(f"Patient ID in features DataFrame: {df['patient_id'].values[0]}")
#
#         # Check if DataFrame is empty
#         if df.empty:
#             logger.warning(f"Empty DataFrame for segment {segment_index}. Skipping save.")
#             return blob_names
#
#         # Log DataFrame info
#         logger.info(f"DataFrame for segment {segment_index}:")
#         logger.info(f"  Shape: {df.shape}")
#         logger.info(f"  Columns: {df.columns.tolist()}")
#         logger.info(f"  Non-null counts:\n{df.count()}")
#
#         # Ensure patient_id is the first column
#         if 'patient_id' in df.columns:
#             df = df[['patient_id'] + [col for col in df.columns if col != 'patient_id']]
#
#         # Read the CSV file
#         csv_df = pd.read_csv(csv_file_path)
#
#         # Check if 'patient_id' is in the CSV DataFrame
#         if 'patient_id' not in csv_df.columns:
#             logger.error(f"'patient_id' column is missing from the CSV file {csv_file_path}")
#             return blob_names
#
#         # Log the unique patient_ids from the CSV DataFrame
#         logger.info(f"Unique Patient IDs in CSV DataFrame: {csv_df['patient_id'].unique()}")
#
#         # Check for null values in patient_id
#         if df['patient_id'].isnull().any():
#             logger.warning(f"Null values found in patient_id for segment {segment_index}")
#         if csv_df['patient_id'].isnull().any():
#             logger.warning(f"Null values found in patient_id in CSV file")
#
#         # Join the DataFrame with the CSV file
#         merged_df = pd.merge(df, csv_df, on='patient_id', how='left', suffixes=('', '_csv'))
#
#         # Check if the merge resulted in any rows
#         if merged_df.empty:
#             logger.warning(
#                 f"Merge resulted in empty DataFrame for segment {segment_index}. Check if patient_id matches between features and CSV.")
#
#         # Log DataFrame info
#         logger.info(f"Merged DataFrame for segment {segment_index}:")
#         logger.info(f"  Shape: {merged_df.shape}")
#         logger.info(f"  Columns: {merged_df.columns.tolist()}")
#         logger.info(f"  Non-null counts:\n{merged_df.count()}")
#
#         # Log important information
#         logger.info(f"Saving features for segment {segment_index}:")
#         logger.info(f"  Patient ID: {flat_features.get('patient_id', 'N/A')}")
#         logger.info(f"  Diagnosis: {flat_features.get('diagnosis', 'N/A')}")
#         logger.info(f"  Raw Clinical Score: {flat_features.get('raw_clinical_score', 'N/A')}")
#         logger.info(f"  Scaled Clinical Score: {flat_features.get('scaled_clinical_score', 'N/A')}")
#
#         # Convert DataFrame to Parquet
#         table = pa.Table.from_pandas(df)
#
#         # Check if table is empty
#         if table.num_rows == 0:
#             logger.warning(f"Empty Parquet table for segment {segment_index}. Skipping save.")
#             return blob_names
#
#         # Log Parquet table info
#         logger.info(f"Parquet table for segment {segment_index}:")
#         logger.info(f"  Number of rows: {table.num_rows}")
#         logger.info(f"  Number of columns: {table.num_columns}")
#         logger.info(f"  Schema: {table.schema}")
#
#         buf = io.BytesIO()
#         pq.write_table(table, buf)
#
#         # Check if buffer is empty
#         if buf.getbuffer().nbytes == 0:
#             logger.warning(f"Empty buffer for segment {segment_index}. Skipping save.")
#             return blob_names
#
#         # Upload to bucket
#         upload_to_bucket(bucket_name, buf.getvalue(), parquet_blob_path, content_type='application/octet-stream')
#
#         logger.info(f"Saved {segment_type} segment {segment_index} to {parquet_blob_path} in bucket {bucket_name}")
#         logger.info(f"Segment {segment_index} includes {len(flat_features)} features")
#         logger.info(f"Key features for segment {segment_index}:")
#         for field in required_fields:
#             if field in flat_features:
#                 logger.info(f"  - {field}: {flat_features[field]}")
#
#         blob_names.append(parquet_blob_path)
#     except Exception as e:
#         logger.error(f"Error saving features for segment {segment_index}: {e}")
#         logger.exception("Exception details:")  # Full traceback
#
#     return blob_names
#
#
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
#
#
# async def write_results_to_files(successful_extractions, unsuccessful_extractions, audio_bucket_name,
#                                  result_bucket_path):
#     # Write successful extractions
#     successful_extractions_path = "successful_extractions.txt"
#     async with aiofiles.open(successful_extractions_path, "w") as f:
#         for item in successful_extractions:
#             await f.write(
#                 f"{item['filename']}.json: {item['num_patient_segments']} patient segments, {item['num_clinician_segments']} clinician segments\n")
#
#     # Upload the updated successful_extractions.txt to GCS
#     client = storage.Client()
#     bucket = client.bucket(audio_bucket_name)
#     blob = bucket.blob(f"{result_bucket_path}/successful_extractions.txt")
#     await asyncio.to_thread(blob.upload_from_filename, successful_extractions_path)
#
#     logger.info(f"Updated successful_extractions.txt with {len(successful_extractions)} entries")
#     await upload_to_gcs(successful_extractions_path, audio_bucket_name,
#                         f"{result_bucket_path}/successful_extractions.txt")
#
#     # Write unsuccessful extractions
#     unsuccessful_extractions_path = "unsuccessful_extractions.txt"
#     async with aiofiles.open(unsuccessful_extractions_path, "w") as f:
#         for item in unsuccessful_extractions:
#             await f.write(f"{item}\n")
#     await upload_to_gcs(unsuccessful_extractions_path, audio_bucket_name,
#                         f"{result_bucket_path}/unsuccessful_extractions.txt")
#
#     logger.info(f"Processing complete. Total files processed: {len(successful_extractions)}")
#     logger.info(f"Total files skipped or failed: {len(unsuccessful_extractions)}")
#
#
# # Path to the local file
# local_file_path = "processed_files.json"
# local_checkpoint_path = "checkpoint.pickle"
#
# # Initialize processed_files and checkpoint
# processed_files = None
# checkpoint = None
#
# # Try to load existing processed files
# try:
#     with open(local_file_path, 'r') as f:
#         processed_files = json.load(f)
#     logger.info(f"Loaded existing processed files: {len(processed_files)} files")
# except FileNotFoundError:
#     logger.info("No existing processed files found. Initializing empty list.")
#     processed_files = []
#
# # Try to load existing checkpoint
# try:
#     with open(local_checkpoint_path, 'rb') as f:
#         checkpoint = pickle.load(f)
#     logger.info(
#         f"Loaded existing checkpoint: last_processed_index={checkpoint['last_processed_index']}, processed_count={checkpoint['processed_count']}")
# except FileNotFoundError:
#     logger.info("No existing checkpoint found. Initializing new checkpoint.")
#     checkpoint = Checkpoint(last_processed_index=0, processed_count=0)
#
# # Only create new files if they don't exist
# if not os.path.exists(local_file_path):
#     with open(local_file_path, 'w') as f:
#         json.dump(processed_files, f)
#     logger.info(f"{local_file_path} created with initial content: {processed_files}")
#
# if not os.path.exists(local_checkpoint_path):
#     with open(local_checkpoint_path, 'wb') as f:
#         pickle.dump(checkpoint, f)
#     logger.info(f"{local_checkpoint_path} created locally.")
#
#
# async def upload_to_gcs(local_file_path, bucket_name, gcs_path):
#     if os.path.exists(local_file_path):
#         client = storage.Client()
#         bucket = client.bucket(bucket_name)
#         blob = bucket.blob(gcs_path)
#
#         # Run the upload in a separate thread
#         await asyncio.to_thread(blob.upload_from_filename, local_file_path)
#
#         logger.info(f"Uploaded {local_file_path} to {gcs_path}")
#     else:
#         logger.error(f"Local file does not exist: {local_file_path}")
#
#
# # Upload files to GCS
# await upload_to_gcs(local_file_path, "private-management-files",
#                     "analysis_results/diarized_patient_feature_analysis/processed_files.json")
#
# bucket_name = "private-management-files"
# gcs_checkpoint_path = "analysis_results/diarized_patient_feature_analysis/checkpoint.pickle"
# await upload_to_gcs(local_checkpoint_path, bucket_name, gcs_checkpoint_path)
#
# # Initialize logger
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
#
#
# async def main(start_index, num_files, batch_size):
#     input_bucket_name = "private-management-files"
#     audio_bucket_name = input_bucket_name
#     result_bucket_name = "private-management-files"
#     result_bucket_path = "analysis_results/diarized_patient_feature_analysis"  # Directory path within the results bucket
#     audio_directory_path = "Voice Memos 14965 FILES"
#     json_bucket_name = "processed-json-files-v2"
#     parquet_base_path = 'analysis_results/diarized_patient_feature_analysis/patient'
#     json_directory_path = ""
#     output_dir = "converted_wavs"
#     target_sr = 16000
#     csv_file_path = 'extraction_6k_07_30.csv'
#     output_bucket_name = "converted-wav-files"
#     patient_segments_dir = "patient/"
#     clinician_segments_dir = "clinician/"
#     needs_review = "needs_review/"
#     uncertain = "uncertain/"
#     subset_size = 100
#     chunk_duration = 600  # 10 minutes
#     force_reprocess = False  # Set to true if reprocessing of all files is desired
#
#     # Constants for batch processing
#     # TARGET_FILE_COUNT = NUM_FILES_TO_PROCESS
#     BATCH_SIZE = batch_size
#     # MAX_WORKERS = 4  # Adjust based on CPU core count
#
#     doctor_embedding = [0.027388021, -0.07954174, -0.1880659, -0.07362926, 0.017095255, 0.06614484, -0.060658358,
#                         -0.1520089,
#                         0.05050724, 0.13823983, -0.195343, -0.021990541, -0.012354583, 0.13133956, 0.046236396,
#                         -0.047753766,
#                         -0.07486497, -0.006071479, -0.012003269, 0.10758188, -0.13095702, -0.032014452, 0.063345745,
#                         0.010770617,
#                         0.075977154, -0.09341858, 0.40589684, 0.10753021, 0.07130297, 0.049874492, -0.10928312,
#                         -0.23408347,
#                         0.031938203, 0.100357085, 0.012041524, 0.067951955, -0.28349358, 0.14764488, 0.09192346,
#                         -0.1970588,
#                         -0.11602559, -0.08523904, 0.28757828, -0.007402247, 0.15888563, -0.2731386, 0.13024445,
#                         -0.35028216,
#                         -0.091732174, -0.057032816, -0.070172, -0.1896452, -0.23176798, 0.0023431256, -0.07311069,
#                         0.06198946,
#                         -0.11586146, -0.07282369, 0.038819317, 0.19677408, 0.1981905, -0.080469996, -0.035803348,
#                         -0.05332913,
#                         0.0932007, 0.12431268, -0.18882939, 0.022097435, 0.15987822, -0.01827943, 0.031037426,
#                         -0.21608266,
#                         0.035995483, -0.23686396, 0.038641535, -0.012032904, 0.07147251, 0.18760602, 0.033584677,
#                         0.07263459,
#                         -0.010202788, 0.06862273, -0.072724774, 0.0027499315, -0.05819865, -0.15924524, -0.10606319,
#                         -0.14328869,
#                         0.09313385, 0.047067262, -0.011805432, 0.03301046, 0.07456483, 0.038653612, -0.13466889,
#                         -0.080597125,
#                         0.08855612, 0.009737631, -0.0064263474, -0.14640735, -0.08189522, 0.1337342, 0.21723405,
#                         -0.022292873,
#                         -0.053229712, -0.14825343, -0.12637818, -0.057791345, 0.046465997, -0.1491514, -0.13027054,
#                         -0.120630905,
#                         0.021518994, 0.090340726, 0.05713073, 0.11808799, 0.05239176, -0.08953367, -0.08599762,
#                         0.032347158,
#                         -0.26080036, -0.091449924, 0.21309778, 0.13188404, 0.28140226, 0.07158558, -0.070624776,
#                         0.011989512,
#                         0.036408383, 0.1111061, -0.04961122, 0.056732737, -0.086238176, 0.022235006, -0.05333481,
#                         0.13974461,
#                         0.10195785, -0.20220064, -0.23005918, 0.098049305, -0.052117, 0.08899307, 0.024334833,
#                         0.07546252,
#                         -0.19669111, -0.054963145, -0.19585925, 0.028986478, 0.18582132, -0.034755882, -0.06631662,
#                         -0.046198606,
#                         -0.13444185, -0.20085803, 0.1119844, 0.20185867, 0.11537046, 0.04083378, 0.019831663,
#                         -0.061341245,
#                         -0.22542395, -0.11161387, -0.1631795, -0.3688563, -0.12441651, -0.04530985, 0.0018225629,
#                         -0.06620886,
#                         -0.13069871, -0.07591105, 0.080656834, 0.16797425, -0.12757833, -0.013322804, -0.03947594,
#                         -0.0023866054,
#                         -0.023134485, 0.0066579916, 0.26853716, -0.19113475, -0.037268497, -0.07751894, -0.21992223,
#                         0.059405293,
#                         0.18945974, 0.052219722, 0.024879504, 0.15811552, -0.003171729, 0.03698936, -0.05968223,
#                         0.074354604,
#                         -0.23650698, -0.10495075, -0.11417772, 0.110579714, -0.0090435, 0.12677349, 0.19851904,
#                         0.09120062,
#                         0.05464324, 0.07442424, -0.12977292, 0.04894126, 0.17627418, 0.014972787, -0.09029392,
#                         -0.009723386,
#                         0.061440635, -0.17538126, 0.06401659, -0.14217643, 0.063380286, -0.19541802, 0.17453186,
#                         -0.11952966,
#                         -0.0629101, -0.07532615, 0.14984012, 0.063968405, -0.20957893, -0.10937524, -0.08522758,
#                         0.13371451,
#                         -0.0008033216, 0.17591108, 0.059045143, 0.09383244, 0.08384961, -0.06593692, -0.13012241,
#                         -0.08402608,
#                         -0.05414322, 0.18448748, -0.120289005, 0.12521201, 0.048590824, 0.1144361, 0.102006495,
#                         -0.10667032,
#                         -0.13393456]
#
#     # Ensure output directories exist
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, patient_segments_dir), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, clinician_segments_dir), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, needs_review), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, uncertain), exist_ok=True)
#
#     adaptive_thresholder = AdaptiveThresholder()
#     separator = PatientFeatureExtractor(doctor_embedding)
#     filtering_criteria = load_filtering_criteria('aggregated-speaker-embeddings-with-docs-voice-and-cluster.gzip')
#
#     # Load the list of already processed files
#     processed_files = load_processed_files(audio_bucket_name, result_bucket_path)
#     logger.info(f"Processed files loaded: {processed_files}")
#     if processed_files is None:
#         logger.warning("No processed files found. Initializing empty set.")
#         processed_files = set()
#     else:
#         logger.info(f"Loaded {len(processed_files)} processed files.")
#
#     # Load the checkpoint
#     checkpoint = await load_checkpoint(result_bucket_name, result_bucket_path)
#     logger.info(
#         f"Loaded checkpoint: last_processed_index={checkpoint['last_processed_index']}, processed_count={checkpoint['processed_count']}")
#
#     # Load the checkpoint
#     checkpoint = await load_checkpoint(result_bucket_name, result_bucket_path)
#     logger.info(f"Main function: Loaded checkpoint: {checkpoint}")
#
#     if not hasattr(checkpoint, 'current_file_progress'):
#         logger.warning("Checkpoint does not have current_file_progress. Adding it.")
#         checkpoint.current_file_progress = {}
#
#     # if not isinstance(checkpoint, Checkpoint):
#     # logger.error("Loaded checkpoint is not a valid Checkpoint object.")
#     # checkpoint = Checkpoint()
#
#     # logger.info(f"Loaded {len(processed_files)} already processed files.")
#
#     # After processing files
#     await save_processed_files(audio_bucket_name, result_bucket_path, processed_files)
#
#     # Retrieve matching file subsets
#     json_files_subset, audio_files_subset = list_matching_files_subset(
#         json_bucket_name, json_directory_path, input_bucket_name, audio_directory_path, subset_size
#     )
#
#     file_pairs = list(zip(json_files_subset, audio_files_subset))
#     total_files = len(file_pairs)
#
#     # Shuffle the file pairs to ensure randomness
#     random.shuffle(file_pairs)
#
#     logger.info(f"Total files to process: {total_files}")
#     logger.info(f"JSON files subset: {json_files_subset}")
#     logger.info(f"Audio files subset: {audio_files_subset}")
#
#     # Process files in batches
#     end_index = min(start_index + num_files, total_files)
#     files_to_process = file_pairs[start_index:end_index]
#
#     successful_extractions = []
#     unsuccessful_extractions = []
#
#     for batch_start in range(0, len(files_to_process), BATCH_SIZE):
#         batch_end = min(batch_start + BATCH_SIZE, len(files_to_process))
#         batch = files_to_process[batch_start:batch_end]
#
#         logger.info(
#             f"Processing batch {batch_start // BATCH_SIZE + 1}, files {batch_start + start_index} to {batch_end + start_index - 1}")
#
#         results = await process_files(batch)
#
#         for result in results:
#             if result and result['status'] == 'success':
#                 successful_extractions.append(result)
#                 processed_files.add(result['filename'])
#                 checkpoint['processed_count'] += 1
#             elif result and result['status'] != 'already_processed':
#                 unsuccessful_extractions.append(result['filename'])
#
#         checkpoint['last_processed_index'] = start_index + batch_end
#
#         if checkpoint['processed_count'] % 10 == 0:
#             await save_processed_files(audio_bucket_name, result_bucket_path, processed_files)
#             await save_checkpoint(audio_bucket_name, result_bucket_path, checkpoint)
#
#         logger.info(f"Processed {checkpoint['processed_count']} files")
#
#         await asyncio.to_thread(monitor_cpu_usage)
#         await asyncio.to_thread(monitor_disk_io)
#         await asyncio.to_thread(monitor_network)
#
#     logger.info(f"Completed processing files from index {start_index} to {end_index}")
#
#     # Process with medication information
#     medication_info = {
#         'medication': 'Vraylar',
#         'dosage': '1.5 mg',
#         'duration': '7-8 days'
#     }
#
#     start_time = time.time()
#
#     async def process_file_pair(json_file, audio_file, result_bucket_path, checkpoint, adaptive_thresholder):
#         logger.info(f"Checkpoint received in process_file_pair: {checkpoint}")
#         if not isinstance(checkpoint, Checkpoint):
#             logger.error(f"Invalid checkpoint object: {type(checkpoint)}")
#             checkpoint = Checkpoint()  # Create a new Checkpoint object if the passed one is invalid
#
#         if not hasattr(checkpoint, 'current_file_progress'):
#             logger.warning("Checkpoint does not have current_file_progress. Adding it.")
#             checkpoint.current_file_progress = {}
#
#         file_id = os.path.splitext(os.path.basename(json_file))[0]
#         logger.info(f"Processing file pair: Audio: {audio_file}, JSON: {json_file}")
#
#         if not force_reprocess and check_if_processed(file_id):
#             logger.info(f"Skipping {json_file} as it has already been processed.")
#             return None
#
#         try:
#             # Load any existing progress for this file
#             file_progress = checkpoint.current_file_progress.get(file_id, {})
#             start_segment = file_progress.get('last_processed_segment', 0)
#         except AttributeError as e:
#             logger.error(f"AttributeError accessing checkpoint.current_file_progress: {e}")
#             logger.info(f"Checkpoint type: {type(checkpoint)}, dir(checkpoint): {dir(checkpoint)}")
#             file_progress = {}
#             start_segment = 0
#
#         # Apply filtering criteria
#         file_criteria = filtering_criteria[filtering_criteria['original-file'] == file_id]
#         if not file_criteria.empty:
#             try:
#                 num_speakers = file_criteria['num-unique-speakers'].iloc[0]
#                 duration = file_criteria['full-audio-duration(seconds)'].iloc[0]
#                 cluster = file_criteria['cluster'].iloc[0]
#
#                 if num_speakers > 2 or (duration <= 900 and num_speakers == 1) or duration <= 900 or cluster == -1:
#                     logger.info(f"File {file_id} does not meet processing criteria. Skipping.")
#                     return {'filename': file_id, 'status': 'skipped'}
#             except Exception as e:
#                 logger.error(
#                     f"Error extracting filtering criteria for file {file_id}: {str(e)}. Proceeding with processing.")
#
#         # Convert the audio file to WAV format
#         converted_wav_gcs_path = convert_to_wav(input_bucket_name, output_bucket_name, audio_file, audio_directory_path,
#                                                 output_dir)
#         if converted_wav_gcs_path is None:
#             logger.error(f"Failed to convert {audio_file} to WAV format. Skipping this file.")
#             return {'filename': file_id, 'status': 'conversion_failed'}
#
#         # Ensure the converted file exists in GCS
#         client = storage.Client()
#         bucket = client.get_bucket(output_bucket_name)
#         blob = bucket.blob(converted_wav_gcs_path)
#
#         if not blob.exists():
#             logger.error(
#                 f"Converted file not found in GCS: gs://{output_bucket_name}/{converted_wav_gcs_path}. Skipping this file.")
#             return {'filename': file_id, 'status': 'file_not_found'}
#
#         local_temp_path = f"/tmp/{os.path.basename(converted_wav_gcs_path)}"
#         try:
#             reference_datetime = extract_reference_datetime(converted_wav_gcs_path)
#             if reference_datetime is None:
#                 reference_datetime = datetime.now()
#
#             blob.download_to_filename(local_temp_path)
#             y, sr = librosa.load(local_temp_path, sr=target_sr)
#
#             json_segments = load_diarization_json(json_bucket_name, json_file)
#             if not json_segments:
#                 return {'filename': file_id, 'status': 'no_json_segments'}
#
#             segments = await process_audio_file(
#                 y, sr, audio_bucket_name, json_file,
#                 reference_datetime=reference_datetime,
#                 force_reprocess=force_reprocess,
#                 doctor_embedding=doctor_embedding,
#                 separator=separator,
#                 chunk_duration=chunk_duration,
#                 adaptive_thresholder=adaptive_thresholder,
#                 checkpoint=checkpoint,
#                 result_bucket_path=result_bucket_path
#             )
#
#             if segments:
#                 processed_segments = process_segments(segments)
#                 diagnoses = [segment['diagnosis'] for segment in processed_segments]
#                 diagnosis_counts = Counter(diagnoses)
#
#                 return {
#                     'filename': file_id,
#                     'status': 'success',
#                     'num_patient_segments': len([s for s in processed_segments if s['voice'] == 'patient']),
#                     'num_clinician_segments': len([s for s in processed_segments if s['voice'] == 'doctor']),
#                     'diagnoses': dict(diagnosis_counts)
#                 }
#             else:
#                 return {'filename': file_id, 'status': 'processing_failed'}
#
#         except Exception as e:
#             logger.error(f"Error processing file gs://{output_bucket_name}/{converted_wav_gcs_path}: {str(e)}")
#             return {'filename': file_id, 'status': 'error', 'error_message': str(e)}
#
#         finally:
#             # Remove the temporary file to free up space
#             if os.path.exists(local_temp_path):
#                 os.remove(local_temp_path)
#                 logger.info(f"Deleted temporary file: {local_temp_path}")
#
#     async def process_files(batch):
#         return await asyncio.gather(
#             *[process_file_pair(json_file, audio_file, result_bucket_path, checkpoint, adaptive_thresholder) for
#               json_file, audio_file in batch])
#
#     with tqdm(total=min(num_files, total_files)) as pbar:
#         while checkpoint['processed_count'] < num_files and checkpoint['last_processed_index'] < total_files:
#             batch = file_pairs[checkpoint['last_processed_index']:checkpoint['last_processed_index'] + BATCH_SIZE]
#
#             results = await process_files(batch)
#
#             for result in results:
#                 if result and result['status'] == 'success':
#                     successful_extractions.append(result)
#                     processed_files.add(result['filename'])
#                     checkpoint['processed_count'] += 1
#                     pbar.update(1)
#                 elif result and result['status'] != 'already_processed':
#                     unsuccessful_extractions.append(result['filename'])
#
#             checkpoint['last_processed_index'] += 1
#
#             if checkpoint['processed_count'] % 10 == 0:
#                 await save_processed_files(audio_bucket_name, result_bucket_path, processed_files)
#                 await save_checkpoint(audio_bucket_name, result_bucket_path, checkpoint)
#
#             logger.info(f"Processed {checkpoint['processed_count']} files")
#
#             await asyncio.to_thread(monitor_cpu_usage)
#             await asyncio.to_thread(monitor_disk_io)
#             await asyncio.to_thread(monitor_network)
#
#             pbar.set_description(f"Processed {checkpoint['processed_count']}/{total_files} files")
#
#             if checkpoint['processed_count'] >= num_files:
#                 break
#
#     # Summary of processed files
#     logger.info(f"Total files processed: {len(processed_files)}")
#     if len(processed_files) == num_files:
#         logger.info(f"All {num_files} files have been successfully processed.")
#     else:
#         logger.warning(f"Processed {len(processed_files)} out of {num_files} files.")
#
#     end_time = time.time()
#     total_time = end_time - start_time
#     logger.info(f"Completed processing {checkpoint['processed_count']} files in {total_time:.2f} seconds")
#     logger.info(f"Average processing speed: {checkpoint['processed_count'] / total_time:.2f} files per second")
#
#     # Write results to files
#     await write_results_to_files(successful_extractions, unsuccessful_extractions, audio_bucket_name,
#                                  result_bucket_path)
#
#
# # For use in Jupyter notebook
# if __name__ == "__main__":
#     import sys
#
#     # Get the start_index from command line arguments
#     if len(sys.argv) > 1:
#         start_index = int(sys.argv[1])
#     else:
#         raise ValueError("Start index not provided")
#
#     # Set other parameters
#     num_files = 100  # Total number of files to process
#     batch_size = 10  # Size of each batch
#
#     # Run the main function
#     asyncio.run(main(start_index, num_files, batch_size))
