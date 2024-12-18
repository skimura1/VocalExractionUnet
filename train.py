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
    save_predictions,
    batch_normalized
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100
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

def validate_model(loader, encoder, model, loss_fn):
    val_bar = tqdm(loader, desc="Validating", leave=False)
    model.eval()
    val_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(val_bar)):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            data = encoder(data)
            data_normalized, _, _ = batch_normalized(data)
            predictions = model(data_normalized)
            targets = encoder(targets)
            targets_normalized, _, _ = batch_normalized(targets)
            loss = loss_fn(predictions, targets_normalized)

            batch_size = data.size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size

            val_bar.set_postfix(loss=val_loss)
    average_loss = val_loss / total_samples
    return average_loss

def train_fn(loader, encoder, model, optimizer, loss_fn, scaler):
    train_bar = tqdm(loader, desc="Training")

    for batch_idx, (data, targets) in enumerate(train_bar):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)


        # forward
        with torch.amp.autocast(DEVICE):
            data = encoder(data)
            data_normalized, _, _ = batch_normalized(data)
            predictions = model(data_normalized)
            targets = encoder(targets)
            targets_normalized, _, _ = batch_normalized(targets)
            loss = loss_fn(predictions, targets_normalized)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        train_bar.set_postfix(loss=loss.item())


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
    best_loss = float('inf')
    patience = 5
    count = 0

    for epoch in range(NUM_EPOCHS):
        train_fn(loader=train_loader,
                 model=model,
                 encoder=encoder,
                 optimizer=optimizer,
                 loss_fn=loss_fn,
                 scaler=scaler)

        val_loss = validate_model(loader=val_loader,
                                  encoder=encoder,
                                  model=model,
                                  loss_fn=loss_fn)
        print(f"Average Val Loss:{val_loss}")

        if val_loss < best_loss:
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            save_checkpoint(checkpoint)
            save_predictions(val_loader, model, encoder=encoder, folder='saved_spectrograms', device='cuda')
        else:
            count += 1
            if count > patience:
                break

if __name__ == "__main__":
    main()
