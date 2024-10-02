import io
import tempfile
import os
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
    # Open the file in binary mode and read it into memory
    with open(file_path, 'rb') as f:
        file_in_memory = io.BytesIO(f.read())  # Wrap the file data in a BytesIO object
    # Debugging: Check if the file was fully read
    print(f"Loaded file size: {len(file_in_memory.getvalue())} bytes")
    return file_in_memory


def new_function(m4amem):
    """Convert the in-memory m4a file to wav format using a temporary file."""
    try:
        # Ensure the cursor is at the beginning of the BytesIO object
        m4amem.seek(0)

        # Create a temporary file to write the m4a file to disk
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_m4a_file:
            temp_m4a_file.write(m4amem.read())
            temp_m4a_file_path = temp_m4a_file.name

        # Now convert the temporary m4a file to wav format using ffmpeg
        out, err = (
            ffmpeg
            .input(temp_m4a_file_path)  # Read from the temporary file on disk
            .output('pipe:1', format='wav', ar='16000', ac=1)  # Set sample rate to 16kHz and mono (1 channel)
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Clean up the temporary file
        os.remove(temp_m4a_file_path)

        # Debugging: Print ffmpeg logs
        print("FFmpeg conversion logs:\n", err.decode('utf-8'))

        # Wrap the resulting wav file in a BytesIO object and return it
        wav_file_in_memory = io.BytesIO(out)

        return wav_file_in_memory

    except ffmpeg.Error as e:
        print(f"Error during conversion: {e.stderr.decode('utf-8')}")
        return None


# Example usage with your existing code
if __name__ == "__main__":
    m4amem = load_file_from_disk("/Users/dm/repo/neurodiag/preprocessing/data/temp.m4a")

    # Verify the m4a file from memory
    if verify_m4a_file_from_memory(m4amem):
        print("M4A file verification successful.")
    else:
        print("M4A file verification failed.")

    # Convert the m4a to wav and store in memory
    wav_mem = new_function(m4amem)

    # Verify the converted wav file from memory
    if wav_mem is not None and verify_wav_file_from_memory(wav_mem):
        print("WAV file verification successful.")
    else:
        print("WAV file verification failed.")