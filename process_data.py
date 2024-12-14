import musdb
from pathlib import Path
import torchaudio.transforms as T
import numpy as np
import torch


class AudioToDBSpectrogram(torch.nn.Module):
    # TODO: Implement transform pipeline


if __name__ == "__main__":
    # Download MUSDB data (train, valid, test)
    train_musdb = load_data()
    valid_musdb = load_data(split="valid")
    # test_musdb = load_data(subsets="test", split=None)

    spectrogram = T.Spectrogram(n_fft=1024)
    amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)

    for track in train_musdb:
        track_name = track.name
        # Get signals
        mix_signal = track.audio.T.astype(np.float32)
        vocal_signal = track.targets['vocals'].audio.T.astype(np.float32)

        # Convert to Tensor
        mix_signal = torch.from_numpy(mix_signal)
        vocal_signal = torch.from_numpy(vocal_signal)

        # Mix down to single channel
        mix_signal = mix_down_if_necessary(mix_signal)
        vocal_signal = mix_down_if_necessary(vocal_signal)

        # Convert to spectrograms
        mix_spec = spectrogram(mix_signal)
        vocal_spec = spectrogram(vocal_signal)

        # Conver to decibel_spectrogram
        mix_db_spec = amplitude_to_db(mix_spec)
        vocal_db_spec = amplitude_to_db(vocal_spec)

        # Train UNET Model (Input: mix_db_spec, Ouput: vocal_db_spec)
