#!/usr/bin/env python3
import io
import json
import os

import librosa
import numpy as np
import webrtcvad
from google.cloud import storage
from pydub import AudioSegment


def convert_to_wav(input_bucket_name, audio_file, audio_directory_path):
    client = storage.Client()
    bucket = client.get_bucket(input_bucket_name)

    file_name = os.path.basename(audio_file)
    cleaned_file_name = file_name.replace("._", "", 1)
    cleaned_file_path = os.path.join(audio_directory_path, cleaned_file_name)

    blob = bucket.blob(cleaned_file_path)
    if not blob.exists():
        return None

    audio_bytes = blob.download_as_bytes()
    return audio_bytes


def load_diarization_json(json_bucket_name, json_file):
    client = storage.Client()
    bucket = client.get_bucket(json_bucket_name)
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


def apply_preemphasis(y, pre_emphasis=0.97):
    return np.append(y[0], y[1:] - pre_emphasis * y[:-1])


def apply_vad(y, sr, vad_window_size=0.03, vad_threshold=0.8):
    vad = webrtcvad.Vad(3)  # Aggressive VAD mode
    frames = librosa.util.frame(y, frame_length=int(vad_window_size * sr), hop_length=int(vad_window_size * sr))
    vad_labels = [vad.is_speech(frame.astype(np.int16).tobytes(), sample_rate=sr) for frame in frames.T]
    vad_segments = []
    start_time = 0
    for i, label in enumerate(vad_labels):
        if label:
            if len(vad_segments) == 0 or vad_segments[-1][1] != i - 1:
                vad_segments.append([i, i])
            else:
                vad_segments[-1][1] = i
    vad_segments = [(start * vad_window_size, end * vad_window_size) for start, end in vad_segments]
    return vad_segments


def apply_spectral_gating(y, sr):
    S = librosa.stft(y)
    S_magnitude, S_phase = np.abs(S), np.angle(S)
    S_db = librosa.amplitude_to_db(S_magnitude, ref=np.max)
    mean_db = np.mean(S_db, axis=1, keepdims=True)
    threshold_db = mean_db - 20
    mask = S_db > threshold_db
    S_denoised = S_magnitude * mask
    S_denoised = S_denoised * np.exp(1j * S_phase)
    return librosa.istft(S_denoised)


def extract_features(audio_segment, sr, patient_id):
    features = {'patient_id': patient_id}

    minimum_samples = sr * 0.1  # Minimum 100 ms
    if len(audio_segment) < minimum_samples:
        return None

    # MFCCs
    n_mels = min(128, sr // 2)  # Adjust n_mels based on sr
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13, n_mels=n_mels)
    features['mfccs'] = mfccs

    # Delta MFCCs
    if mfccs.shape[1] >= 9:
        delta_mfccs = librosa.feature.delta(mfccs)
    else:
        delta_mfccs = np.zeros_like(mfccs)
    features['delta_mfccs'] = delta_mfccs

    # Delta-Delta MFCCs
    if mfccs.shape[1] >= 9:
        delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
    else:
        delta_delta_mfccs = np.zeros_like(mfccs)
    features['delta_delta_mfccs'] = delta_delta_mfccs

    return features


def process_audio_file(audio_data, json_data, target_sr=16000):
    try:
        # Convert M4A to WAV
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="m4a")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Load audio data
        y, sr = librosa.load(wav_io, sr=target_sr)

        # Apply preemphasis
        y_preemph = apply_preemphasis(y)

        # Apply VAD
        vad_segments = apply_vad(y_preemph, sr)

        # Apply spectral gating for noise reduction
        y_denoised = apply_spectral_gating(y_preemph, sr)

        patient_features = []
        for segment in json_data:
            if segment['voice'] == 'patient':
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                segment_audio = y_denoised[start_sample:end_sample]

                # Extract features
                features = extract_features(segment_audio, sr, f"{segment['start']}_{segment['end']}")
                if features:
                    features['start'] = segment['start']
                    features['end'] = segment['end']
                    patient_features.append(features)

        return patient_features
    except Exception as e:
        return None


def extract_id(filename):
    """Extract the ID between the dash and the extension in the filename."""
    base = os.path.basename(filename)
    try:
        start_idx = base.index('-') + 1
        end_idx = base.rindex('.')
        return base[start_idx:end_idx]
    except ValueError:
        return None


def list_matching_files_subset(json_bucket_name, json_directory_path, audio_bucket_name, audio_directory_path,
                               subset_size=100):
    client = storage.Client()
    json_bucket = client.get_bucket(json_bucket_name)
    audio_bucket = client.get_bucket(audio_bucket_name)

    json_blobs = list(json_bucket.list_blobs(prefix=json_directory_path))
    audio_blobs = list(audio_bucket.list_blobs(prefix=audio_directory_path))

    json_files = [blob.name for blob in json_blobs if blob.name.endswith(".json")]
    audio_files = [blob.name for blob in audio_blobs if
                   blob.name.lower().endswith(".m4a") and not blob.name.startswith("._")]

    matching_files = []
    for json_file in json_files:
        json_id = extract_id(json_file)
        if not json_id:
            continue
        for audio_file in audio_files:
            audio_id = extract_id(audio_file)
            if json_id == audio_id:
                matching_files.append((json_file, audio_file))
                break

    matching_files = matching_files[:subset_size]

    json_files_subset = [match[0] for match in matching_files]
    audio_files_subset = [match[1] for match in matching_files]

    return json_files_subset, audio_files_subset
