"""ARCHIVE BECAUSE BEMMY IS WORKING WITH A TEMP FILE"""
import io

import ffmpeg
import torchaudio




def verify_m4a_file_from_memory(m4a_file_in_memory):
    """verified working from memory"""
    try:
        # Ensure the cursor is at the beginning of the BytesIO object
        m4a_file_in_memory.seek(0)

        # Use ffmpeg to check if the m4a file can be decoded properly
        out, err = (
            ffmpeg
            .input('pipe:0', format='m4a')  # Read from the BytesIO object
            .output('null', f='null')  # Null output means we're just verifying
            .run(input=m4a_file_in_memory.read(), capture_stdout=True, capture_stderr=True)
        )
        print("File is valid and can be processed without errors.")
        return True
    except ffmpeg.Error as e:
        print(f"Error verifying m4a file: {e.stderr.decode('utf-8')}")
        return False

def verify_wav_file_from_memory(wav_file_in_memory):
    """verified working file on memory"""
    try:
        # Ensure the cursor is at the beginning of the BytesIO object
        wav_file_in_memory.seek(0)
        # Load the wav file from memory using torchaudio
        waveform, sample_rate = torchaudio.load(wav_file_in_memory)

        # Print basic properties of the loaded file
        print(f"Sample rate: {sample_rate}")
        print(f"Waveform shape: {waveform.shape}")  # [channels, samples]

        # Verify that the waveform is non-empty
        if waveform.numel() == 0:
            raise ValueError("The waveform is empty.")

        # Optionally, check if sample rate is as expected (e.g., 16000 Hz)
        if sample_rate != 16000:
            print(f"Warning: Unexpected sample rate {sample_rate} Hz")

        # Check if the waveform has a valid number of channels (usually 1 or 2)
        if waveform.size(0) not in [1, 2]:
            raise ValueError(f"Unexpected number of channels: {waveform.size(0)}")

        # Print duration of the audio in seconds
        duration = waveform.size(1) / sample_rate
        print(f"Duration: {duration:.2f} seconds")

        print("WAV file integrity verified.")
        return True
    except Exception as e:
        print(f"Error verifying WAV file: {e}")
        return False

def load_file_from_disk(file_path):
    """verified working for wav and m4a"""
    # Open the wav file in binary mode and read it into memory
    with open(file_path, 'rb') as f:
        file_in_memory = io.BytesIO(f.read())  # Wrap the file data in a BytesIO object
    return file_in_memory




if __name__ == "__main__":
    m4amem = load_file_from_disk("/Users/dm/repo/neurodiag/preprocessing/data/temp.m4a")
    verify_m4a_file_from_memory(m4amem)
    wav_mem = new_function(m4amem)
    verify_wav_file_from_memory(wav_mem)





########################### OLD FROM DISK

def verify_m4a_file(file_path):
    """verified working on disk"""
    try:
        # Use ffmpeg to check if the m4a file can be decoded properly
        # We're not saving the output, just checking if ffmpeg can process it
        out, err = (
            ffmpeg
            .input(file_path)
            .output('null', f='null')  # Null output means we're just verifying
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("File is valid and can be processed without errors.")
        return True
    except ffmpeg.Error as e:
        print(f"Error verifying m4a file: {e.stderr.decode('utf-8')}")
        return False

def convert_m4a_to_wav(input_m4a_path, output_wav_path=None):
    """Verified working on disk"""
    # Convert the m4a file to wav format using ffmpeg in memory
    try:
        # Read input m4a file and convert to wav format using ffmpeg
        out, _ = (
            ffmpeg
            .input(input_m4a_path)  # Read from input file
            .output('pipe:1', format='wav')  # Output as wav to pipe (stdout)
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Load the resulting wav file using torchaudio
        wav_file_in_memory = io.BytesIO(out)
        waveform, sample_rate = torchaudio.load(wav_file_in_memory)

        # Optionally, save the wav file to disk
        if output_wav_path:
            with open(output_wav_path, 'wb') as f:
                f.write(wav_file_in_memory.getvalue())
            print(f"WAV file saved at: {output_wav_path}")

        print("Conversion successful.")
        return waveform, sample_rate

    except ffmpeg.Error as e:
        print(f"Error during conversion: {e.stderr.decode('utf-8')}")
        return None, None