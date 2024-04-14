import json
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
import albumentations as A
import segmentation_models_pytorch as smp
from class_dataset import DolphinDataset
from class_IoUBCELoss import IoUBCELoss
import random
import numpy as np

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = A.Compose([A.PadIfNeeded(min_height=1024, min_width=1024),
                       A.RandomCrop(1024,1024),
                       A.Resize(256, 256),
                       A.RGBShift(25,25,25),
                       A.RandomBrightnessContrast(0.3,0.3),
                       A.Normalize(mean, std),
                       ToTensorV2(),])

dataset = DolphinDataset(image_path, masks, masks_id,  transforms=transform)

train_images_filenames, val_images_filenames = random_split(dataset, [0.8, 0.2])

BATCH = 16
train_loader = DataLoader(
                train_images_filenames,
                batch_size=BATCH,
                shuffle=True,
                )

val_loader = DataLoader(
                val_images_filenames,
                batch_size=BATCH,
                shuffle=False,
                )

def iou_loss(pred, target, smooth=1):
    pred    = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

LR = 0.001
model = model.to(device)
criterion = IoUBCELoss()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def accuracy(y_pred, y):
    return 1-iou_loss(torch.sigmoid(y_pred), y, smooth=1)

def train(model, dataloader, optimizer, criterion, metrics, device):
    epoch_loss = 0
    epoch_acc  = 0
    model.train()
    for (x, y) in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model.forward(x) 
        loss = criterion(y_pred, y)
        acc  = metrics( y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc  += acc.item()
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def evaluate(model, dataloader, criterion,metrics, device):
    epoch_loss = 0
    epoch_acc  = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in tqdm(dataloader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)
            y_pred = model.forward(x)
            loss = criterion(y_pred, y)
            acc  = metrics( y_pred, y)
            epoch_loss += loss.item()
            epoch_acc  += acc.item()
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

EPOCHS = 20
train_loss = torch.zeros(EPOCHS)
valid_loss = torch.zeros(EPOCHS)
train_acc  = torch.zeros(EPOCHS)
valid_acc  = torch.zeros(EPOCHS)

best_valid_loss = float('inf')
best_epoch = 0

for epoch in trange(EPOCHS, desc="Epochs"):
    start_time = time.monotonic()

    train_loss[epoch], train_acc[epoch] = train(model,
                                                train_loader,
                                                optimizer,
                                                criterion,
                                                accuracy,
                                                device)

    valid_loss[epoch], valid_acc[epoch] = evaluate(model,
                                                   val_loader,
                                                   criterion,
                                                   accuracy,
                                                   device)

    if valid_loss[epoch] < best_valid_loss:
        best_valid_loss = valid_loss[epoch]
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pt')

    epoch_mins, epoch_secs = epoch_time(start_time, time.monotonic())
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss[epoch]:.3f} | Train Acc: {train_acc[epoch]*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss[epoch]:.3f} |  Val. Acc: {valid_acc[epoch]*100:.2f}%')
