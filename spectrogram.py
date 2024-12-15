import musdb
import torchaudio.transforms as T
import numpy as np
import torch
import torch.nn as nn
from util import to_mono


class AudioToSpectrogram(nn.Module):
    # TODO: Implement transform pipeline
    def __init__(
        self,
        n_fft=1024,
        hop_length=512,
        top_db=80
    ):
        super().__init__()
        # Computing Complex Spectrogram (For Now)
        self.spectrogram = T.Spectrogram(
            power=None, n_fft=1024, hop_length=512)
        # self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)

    def forward(self, x):
        x = self.spectrogram(x)
        return x


if __name__ == "__main__":
    train_musdb = musdb.DB('./musdb', subsets='train',
                           split='train', download=True)
    # Download MUSDB data (train, valid, test)
    # train_musdb = load_data()
    # valid_musdb = load_data(split="valid")
    # test_musdb = load_data(subsets="test", split=None)

    audio_to_spec = AudioToSpectrogram()

    for track in train_musdb:
        track_name = track.name
        # Get signals
        mix_signal = track.audio.T.astype(np.float32)
        vocal_signal = track.targets['vocals'].audio.T.astype(np.float32)

        # Convert to Tensor
        mix_signal = torch.from_numpy(mix_signal)
        vocal_signal = torch.from_numpy(vocal_signal)

        # Mix down to single channel
        mix_signal = to_mono(mix_signal)
        vocal_signal = to_mono(vocal_signal)

        # Convert to DB spectrogram
        mix_spec = audio_to_spec(mix_signal)
        vocal_spec = audio_to_spec(vocal_signal)

        # Train UNET Model (Input: mix_db_spec, Ouput: vocal_db_spec)
        print('mix_spec:' + str(mix_spec.shape))
        print('vocal_spec:' + str(vocal_spec.shape))
