import musdb
from torch.utils.data import Dataset
import torch

class MUSDBDataset(Dataset):
    def __init__(
        self,
        root = None,
        download = False,
        is_wav = False,
        subsets = "train",
        split = "train",
    ):
        self.is_wav = is_wav
        self.subsets = subsets
        self.split = split
        self.mus = musdb.DB(
            root=root,
            is_wav=is_wav,
            split=split,
            subsets=subsets,
            download=download,
        )
        self.sample_rate = 44100.0  # musdb is fixed sample rate

    def __len__(self):
        return len(self.mus.tracks)

    def __getitem__(self, index):
        # select track
        track = self.mus[index]

        # Load mix and vocal stem as audio waveform (2, 300032)
        mix = torch.as_tensor(track.audio.T, dtype=torch.float32)
        vocal = torch.as_tensor(track.targets["vocals"].audio.T, dtype=torch.float32)

        return mix, vocal


if __name__ == "__main__":
    from transforms import make_filterbanks

    n_fft = 1024
    hop_length = 512
    train_musdb = MUSDBDataset('./musdb')
    stft, _ = make_filterbanks()

    for track in train_musdb:
       mix_audio, vocal_audio = track
       mix_spec =  stft(mix_audio)
       vocal_spec = stft(vocal_audio)
       print(mix_spec.shape, vocal_spec.shape)
