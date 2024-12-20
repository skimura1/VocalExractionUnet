import musdb
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import torch
import torch.nn as nn

def make_filterbanks(n_fft=1024, hop_length=512, complex_data=False, center=False, sample_rate=44100.0, device='cuda'):
    if complex_data:
        encoder = TorchSTFT(n_fft=n_fft, n_hop=hop_length, center=center)
        decoder = TorchISTFT(n_fft, n_hop=hop_length, center=center)
    else:
        encoder = TorchASTFT(n_fft=n_fft, hop_length=hop_length, center=center, device=device)
        decoder = TorchAISTFT(n_fft=n_fft)
    return encoder, decoder

class TorchASTFT(nn.Module):
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

class TorchAISTFT(nn.Module):
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

class TorchSTFT(nn.Module):
    def __init__(
        self,
        n_fft = 4096,
        n_hop = 1024,
        center = False,
        window = None,
    ):
        super(TorchSTFT, self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

        if window is None:
            self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        else:
            self.window = window

    def forward(self, x):
        shape = x.size()

        # pack batch
        x = x.view(-1, shape[-1])

        complex_stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )
        stft_f = torch.view_as_real(complex_stft)
        # unpack batch
        stft_f = stft_f.view(shape[:-1] + stft_f.shape[-3:])
        return stft_f

class TorchISTFT(nn.Module):
    def __init__(
            self,
            n_fft= 4096,
            n_hop = 1024,
            center = False,
            sample_rate = 44100.0,
            window = None,
    ) -> None:
        super(TorchISTFT, self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center
        self.sample_rate = sample_rate

        if window is None:
            self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        else:
            self.window = window


    def forward(self, X, length = None):
        shape = X.size()
        X = X.reshape(-1, shape[-3], shape[-2], shape[-1])

        y = torch.istft(
            torch.view_as_complex(X),
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length,
        )

        y = y.reshape(shape[:-3] + y.shape[-1:])

        return y

class ComplexNorm(nn.Module):
    def __init__(self, mono: bool = False):
        super(ComplexNorm, self).__init__()
        self.mono = mono

    def forward(self, spec):
        spec = torch.abs(torch.view_as_complex(spec))

        # downmix in the mag domain to preserve energy
        if self.mono:
            spec = torch.mean(spec, 1, keepdim=True)

        return spec

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
