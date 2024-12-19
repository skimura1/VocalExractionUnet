import random
from cmath import acosh

import torch
from torch.utils.data import Dataset

import musdb

def apply_augmentations(audio, gain):
    audio = audio * gain
    if random.random() > 0.5:
        audio = audio.flip(0)
    return audio

class MUSDBDataset(Dataset):
    def __init__(
        self,
        root = None,
        download = False,
        is_wav = False,
        subsets = "train",
        split = "train",
        seq_duration = 6.0,
        samples_per_track = 64,
    ):
        self.is_wav = is_wav
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        self.seq_duration = seq_duration
        self.mus = musdb.DB(
            root=root,
            is_wav=is_wav,
            split=split,
            subsets=subsets,
            download=download,
        )
        self.sample_rate = 44100.0  # musdb is fixed sample rate

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track

    def __getitem__(self, index):
        audio_sources = []
        # random augment gain
        random_gain = 0.25 + torch.rand(1) * (1.25 - 0.25)

        # select track
        target_track = self.mus[index // self.samples_per_track]
        if self.split == "train" and self.seq_duration:
            for source in self.mus.setup["sources"]:
                if source != "vocals":
                    # select a random track
                    accompaniment_track = random.choice(self.mus.tracks)
                    # set the excerpt duration
                    accompaniment_track.chunk_duration = self.seq_duration
                    # set random start position
                    accompaniment_track.chunk_start = random.uniform(0, target_track.duration - self.seq_duration)

                    # load source audio and apply time domain source_augmentations
                    source_audio = torch.as_tensor(accompaniment_track.sources[source].audio.T, dtype=torch.float32)

                    # random augment gain
                    source_audio = apply_augmentations(source_audio, random_gain)
                    audio_sources.append(source_audio)

            target_track.chunk_duration = self.seq_duration
            target_track.chunk_start = random.uniform(0, target_track.duration - self.seq_duration)
            vocal = torch.as_tensor(target_track.targets["vocals"].audio.T, dtype=torch.float32)
            audio_sources.append(vocal)
            stems = torch.stack(audio_sources, dim=0)
            mix = stems.sum(dim=0)
        else:
            mix = torch.as_tensor(target_track.audio.T, dtype=torch.float32)
            vocal = torch.as_tensor(target_track.targets["vocals"].audio.T, dtype=torch.float32)

        return mix, vocal


if __name__ == "__main__":
    from transforms import make_filterbanks
    musdb.DB(
        root='./musdb',
        subsets='train',
        split='train',
        download=True)
    n_fft = 1024
    hop_length = 512
    train_ds = MUSDBDataset(
        root='./musdb',
        subsets='train',
        split='train'
    )
    stft, _ = make_filterbanks(device='cpu')

    for track in train_ds:
       mix_audio, vocal_audio = track
       mix_spec =  stft(mix_audio)
       vocal_spec = stft(vocal_audio)
       print(mix_spec.shape, vocal_spec.shape)
