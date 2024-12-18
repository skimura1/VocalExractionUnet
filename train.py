import torch.optim as optim
import torch.nn as nn
import torch

import transforms
from model import UNET
from tqdm import tqdm

from util import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    save_predictions
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 10
NUM_WORKERS = 4
CHANNELS = 2
N_FFT = 1024  # HEIGHT
N_HOPS = 512  # WIDTH
PIN_MEMORY = False
LOAD_MODEL = False

""" 
Train code has been inspired by:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/558557c7989f0b10fee6e8d8f953d7269ae43d4f/ML/Pytorch/image_segmentation/semantic_segmentation_unet/train.py
"""


def train_fn(loader, encoder, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)


        # forward
        with torch.amp.autocast(DEVICE):
            data = encoder(data)
            predictions = model(data)
            targets = encoder(targets)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    model = UNET().to(DEVICE)
    loss_fn = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        musdb_dir='./musdb',
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    stft, istft = transforms.make_filterbanks(
        n_fft=N_FFT, hop_length=N_HOPS
    )
    encoder = stft

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.amp.GradScaler(DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_fn(loader=train_loader,
                 model=model,
                 encoder=encoder,
                 optimizer=optimizer,
                 loss_fn=loss_fn,
                 scaler=scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        save_checkpoint(checkpoint)
        save_predictions(val_loader, model, encoder=encoder, folder='saved_spectrograms', device='cuda')


if __name__ == "__main__":
    main()
