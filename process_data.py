import musdb
from pathlib import Path
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
import skimage.io
import numpy as np
import torch


def load_data(subsets="train", split="train"):
    # 1. Download MUSDB data (train, validation) if not exist
    download = False
    if not Path("./musdb").exists():
        download = True

    return musdb.DB("./musdb", download=download, subsets=subsets, split=split)


def mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def save_specgram_image(specgram, out, title, target_source):
    filename = title + ".png"
    output_path = out / filename
    spec = librosa.power_to_db(specgram)
    spec *= (255.0 / spec.max())
    spec = spec.astype("uint8")
    spec = np.flip(spec, axis=1)
    spec -= 255
    skimage.io.imsave(output_path, spec)


if __name__ == "__main__":
    # Download MUSDB data (train, valid, test)
    train_musdb = load_data()
    valid_musdb = load_data(split="valid")
    # test_musdb = load_data(subsets="test", split=None)

    # train path
    train_images_path = Path("./data/train_images")
    train_masks_path = Path("./data/train_masks")

    # val path
    val_images_path = Path("./data/val_images")
    val_masks_path = Path("./data/val_masks")

    spectrogram = T.Spectrogram(n_fft=1024)

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

        # Save spectrogram in audio path
        save_specgram_image(mix_spec, train_images_path, track_name, "mix")
        save_specgram_image(vocal_spec, train_masks_path, track_name, "vocals")
