import musdb
import torchaudio
import librosa
import torchaudio.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch


class MUSDBDataset(Dataset):

    def __init__(
        self,
        musdb,
        musdb_dir,
        transformation,
        num_samples,
        device
    ):
        self.musdb = musdb
        self.musdb_dir = musdb_dir
        self.device = device
        self.transformation = transformation
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return len(self.musdb)

    def __get_item__(self, index):
        mix_signal = self.musdb[index].audio.T.astype(np.float32)
        vocal_signal = self.musdb[index].targets["vocals"].audio.T.astype(
            np.float32)

        mix_signal = self._to_mono(mix_signal)
        vocal_signal = self._to_mono(vocal_signal)

        mix_signal = mix_signal.to(self.device)
        vocal_signal = vocal_signal.to(self.device)

        mix_db_spectrogram = self.transformation(mix_signal)
        vocal_db_spectrogram = self.transformation(vocal_signal)

        return mix_db_spectrogram, vocal_db_spectrogram

    def _to_mono(signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    if __name__ == "__main__":
        n_fft = 1024
        hop_length = 512
        spectrogram = T.Spectrogram(
            n_fft=n_fft, power=2, hop_length=hop_length)
        power_to_db = T.AmplitudeToDB(stype='power', top_db=80)
