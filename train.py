import torch.optim as optim
import torch.nn as nn
import torch
from model import UNET
from tqdm import tqdm
from spectrogram import AudioToSpectrogram
from util import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
CHANNELS = 2
FREQUENCY_BIN = 513  # HEIGHT
FRAMES = 587  # WIDTH
PIN_MEMORY = True
LOAD_MODEL = False

""" 
Train code has been inspired by:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/558557c7989f0b10fee6e8d8f953d7269ae43d4f/ML/Pytorch/image_segmentation/semantic_segmentation_unet/train.py
"""


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    transform = AudioToSpectrogram()
    model = UNET()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        musdb_dir='./musdb',
        batch_size=BATCH_SIZE,
        transform=transform,
        device=DEVICE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # TODO: Maybe save vocal spectrogram and predicted spectrograms
        save_predictions(
            val_loader, model, folder="saved_spec/", devce=DEVICE
        )


if __name__ == "__main__":
    main()
