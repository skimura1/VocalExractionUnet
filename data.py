import musdb
import torchaudio
import librosa
import torchaudio.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


def load_data(subsets="train", split="train"):
    # 1. Download MUSDB data (train, validation) if not exist
    download = False
    if not Path("./musdb").exists():
        download = True

    return musdb.DB("./musdb", download=download, subsets=subsets, split=split)


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
        mix_signal = self.musdb[index].audio.T.astype(np.float64)
        vocal_signal = self.musdb[index].targets["vocals"].audio
        mix_signal = mix_signal.to(self.device)
        vocal_signal = vocal_signal.to(self.device)

        mix_spectrogram = self.transformation(mix_signal)
        vocal_signal = self.transformation(vocal_signal)

        vocal_signal


if __name__ == "__main__":
    train_data = load_data()
    valid_data = load_data(split="valid")
    test_data = load_data(subsets="test", split=None)

    # Train paths
    spec_train_path = Path("./data/train_spec")
    vocal_mask_train_path = Path("./data/train_mask")

    # Validation paths
    spec_val_path = Path("./data/val_spec")
    vocal_mask_val_path = Path("./data/val_mask")

    # Define transform
    spectrogram = T.Spectrogram(n_fft=1024)

    # 3. Convert Vocal Spectrogram into Binary Mask
    # 4. Split into train_image, train_mask, valid_image, valid_mask

    for track in train_data:
        track
