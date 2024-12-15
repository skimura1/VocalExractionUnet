import musdb
from torch.utils.data import Dataset
import torch
from util import to_mono
from audio_to_dbspec import AudioToDBSpectrogram
import numpy as np


class MUSDBDataset(Dataset):

    def __init__(
        self,
        mdataset,
        transform,
        device
    ):
        self.mdataset = mdataset
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.musdb)

    def __getitem__(self, index):
        # Get the audio from musdb
        mix_signal = self.mdataset[index].audio.T
        vocal_signal = self.mdataset[index].targets["vocals"].audio.T

        # Convert signal into tensor
        mix_signal = torch.from_numpy(mix_signal).float()
        vocal_signal = torch.from_numpy(vocal_signal).float()

        # convert to mono
        mix_signal = to_mono(mix_signal)
        vocal_signal = to_mono(vocal_signal)

        # Use GPU if available
        mix_signal = mix_signal.to(self.device)
        vocal_signal = vocal_signal.to(self.device)

        # Transform into spectrogram
        mix_spectrogram = self.transfrom(mix_signal)
        vocal_spectrogram = self.transform(vocal_signal)

        # Get the magnitude and phase from spectrogram
        # Get number of frequency bands - 1 for even number (n_fft = 1024 -> 513 frequency bands)
        return mix_spectrogram, vocal_spectrogram

    def _spectrogram_mag(self, spec, norm=True):
        """Compute normalized mag spec and phase from spectrogram """
        n_freq_bands = spec.shape[1] - 1
        mag = torch.abs(spec[0, :n_freq_bands, :])
        # mag = mag / torch.max(mag)
        if norm:
            mx = torch.max(mag)
            mn = torch.min(mag)
            mag = ((mag - mn) / (mx - mn))
            norm_param = np.array([mx, mn])
        phase = torch.angle(spec)

        return mag, phase, norm_param


if __name__ == "__main__":
    n_fft = 1024
    hop_length = 512
    train_musdataset = musdb.DB('./musdb', subsets='train',
                                split='train', download=False)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'using device {device}')

    audio_to_dbspec = AudioToDBSpectrogram()
    musdbset = MUSDBDataset(train_musdataset, audio_to_dbspec, device)

    for track in musdbset:
        mix_spec, vocal_spec = track
        print(mix_spec.shape, vocal_spec.shape)
