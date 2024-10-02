"""
Audio Processing and Feature Extraction Script

This script processes audio files from a Google Cloud Storage bucket,
extracts MFCC features from patient segments, and handles diarization data.
It's designed to work with M4A audio files and JSON diarization data.
"""

from preprocessing.refactor.manual_features import convert_to_wav, load_diarization_json, process_audio_file, list_matching_files_subset




def main(input_bucket_name, json_bucket_name, audio_directory_path, json_directory_path, num_files_to_process):

    json_files_subset, audio_files_subset = list_matching_files_subset(
        json_bucket_name, json_directory_path, input_bucket_name, audio_directory_path, num_files_to_process, subset=10
    )

    all_patient_features = []

    for json_file, audio_file in zip(json_files_subset, audio_files_subset):
        # Convert audio to WAV
        audio_data = convert_to_wav(input_bucket_name, audio_file, audio_directory_path)
        if audio_data is None:
            print(f"Missing audio data {audio_file}")

        # Load diarization data
        json_data = load_diarization_json(json_bucket_name, json_file)
        if json_data is None:
            print(f"No diarization data bucket: {json_bucket_name}")

        patient_features = process_audio_file(audio_data, json_data)

        if patient_features:
            all_patient_features.extend(patient_features)

    return all_patient_features


if __name__ == "__main__":
    _input_bucket_name = "private-management-files"
    _json_bucket_name = "processed-json-files-v2"
    _audio_directory_path = "Voice Memos 14965 FILES"
    _json_directory_path = ""  # Adjust if needed
    _num_files_to_process = 1  # Adjust this number as needed

    features = main(_input_bucket_name, _json_bucket_name, _audio_directory_path, _json_directory_path,
                    _num_files_to_process)
