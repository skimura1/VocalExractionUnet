import musdb
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import torch
import torch.nn as nn

def make_filterbanks(n_fft=1024, hop_length=512, center=False, sample_rate=44100.0, device='cuda'):
    encoder = STFT(n_fft=n_fft, hop_length=hop_length, center=center, device=device)
    decoder = ISTFT(n_fft=n_fft, hop_length=hop_length, device=device)
    return encoder, decoder

class STFT(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=512,
        center = False,
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            center=center,
            onesided = True,
            pad_mode = "reflect",
            power=2
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.spectrogram(x)
        return x

class ISTFT(nn.Module):
    def __init__(self,
        n_fft = 1024,
        hop_length= 512,
        sample_rate = 44100.0,
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2,
            win_length=n_fft
        ).to(self.device)

    def forward(self, y):
        y = y.to(self.device)
        y = self.griffin_lim(y)
        return y


if __name__ == "__main__":
    from dataset import MUSDBDataset

    n_fft = 1024
    hop_length = 512
    train_musdb = MUSDBDataset('./musdb')
    stft, istft = make_filterbanks()

    for track in train_musdb:
       mix_audio, vocal_audio = track
       mix_spec =  stft(mix_audio)
       vocal_spec = stft(vocal_audio)
       mix_reconstructed = istft(mix_spec)
       vocal_reconstructed = istft(vocal_spec)

       print(f"mix_spec: {mix_spec.shape}, vocal_spec:{vocal_spec.shape}")
       print(f"mix_audio: {mix_audio.shape}, vocal_audio:{vocal_audio.shape}")
       print(f"mix_reconstructed: {mix_reconstructed.shape}, vocal_reconstructed:{vocal_reconstructed.shape}")
