from torchaudio import transforms
import torch


class MelFrequency:
    def __init__(self, sample_rate=16000, fft_size=400, window_stride=(400, 200), num_filt=40, num_coeffs=40):
        """
        :param sample_rate:
        :param fft_size:
        :param window_stride:
        :param num_filt:
        :param num_coeffs:
        """
        self.sample_rate = sample_rate
        self.window_stride = window_stride
        self.fft_size = fft_size
        self.num_filt = num_filt
        self.num_coeffs = num_coeffs
        self.transform = transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}, )

    def encode(self, waveform, pad=True, clip=1000):
        """TODO implement pad, clip kwargs"""
        vec = self.transform(waveform)
        if vec.shape[2] > clip:
            vec = vec[:, :, :clip]
        else:
            vec = torch.nn.functional.pad(vec, (1, 999 - vec.shape[2]))
        return vec
