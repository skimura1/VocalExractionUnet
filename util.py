import torch
from dataset import MUSDBDataset
from torch.utils.data import DataLoader
import numpy as np
import musdb
from pathlib import Path


def save_checkpoint(state, filename="./models/my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    musdb_dir,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    musdb_path = Path(musdb_dir)
    if not musdb_path.exists():
        musdb_path.mkdir(parents=True)
        _ = musdb.DB(musdb_path, download=True)

    train_ds = MUSDBDataset(
        root=musdb_path,
        subsets='train',
        split='train'
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    valid_ds = MUSDBDataset(
        root=musdb_path,
        subsets='train',
        split='valid'
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, valid_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions(
        loader, model, encoder, folder='saved_spectrograms', device='cuda'
):
    print("=> Saving predictions")
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            x = encoder(x)
            preds = model(x)
        preds = preds.cpu().numpy()

        np.save(f"{folder}/predbatch_{idx}.npy", preds)


if __name__ == "__main__":

    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4
    NUM_EPOCHS = 3
    NUM_WORKERS = 2
    CHANNELS = 2
    FREQUENCY_BIN = 513  # HEIGHT
    FRAMES = 587  # WIDTH
    PIN_MEMORY = False
    LOAD_MODEL = False

    valid_musdataset = musdb.DB('./musdb', subsets='train',
                                split='valid', download=False)

    val_ds = MUSDBDataset(
        mdataset=valid_musdataset,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    mix, vocal = val_ds[0]

    print(f"mix {mix.shape}, vocal {vocal.shape})")

    for mix_batch, vocal_batch in val_loader:
        print(f"mix {mix_batch.shape}, vocal {vocal_batch.shape})")
        print(f'mix {mix_batch}, vocal {vocal_batch})')
