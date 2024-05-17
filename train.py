import numpy as np
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm, trange
from albumentations.pytorch import ToTensorV2
import albumentations as A
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import StepLR
from tensorboard.plugins.hparams import api as hp
from torch.utils.tensorboard import SummaryWriter
from class_dataset import DolphinDataset
from train_config import ModelConfig

model = smp.create_model(
    train_config.model,
    encoder_name=train_config.encoder_name,
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

train_config = ModelConfig()

train_directory = train_config.train_directory
val_directory = train_config.val_directory

mean = train_config.normallize_mean
std  = train_config.normallize_std

transform = A.Compose([
                       A.PadIfNeeded(min_height=train_config.pad_if_needed_min_height, min_width=train_config.pad_if_needed_min_width),
                       A.RandomCrop(train_config.random_crop_height,train_config.random_crop_width),
                       A.Resize(train_config.shape[0], train_config.shape[1]),
                       A.RGBShift(train_config.rgb_shift_r_shift_limit,train_config.rgb_shift_g_shift_limit,train_config.rgb_shift_b_shift_limit, p=train_config.rgb_shift_p),
                       A.RandomBrightnessContrast(train_config.random_brightness_contrast_brightness_limit,train_config.random_brightness_contrast_brightness_limit, p=train_config.random_brightness_contrast_p),
                       A.Normalize(train_config.normallize_mean, train_config.normallize_std),
                       ToTensorV2(),
                       ])

test_transform = A.Compose([
                        A.Resize(train_config.shape[0], train_config.shape[1]),
                        A.Normalize(train_config.normallize_mean, train_config.normallize_std),
                        ToTensorV2(),
                        ])

train_dataset = DolphinDataset(image_path, train_directory,  transforms=transform)
val_dataset = DolphinDataset(image_path, val_directory,  transforms=test_transform)

BATCH = train_config.batch_size

train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH,
                shuffle=True,
                )

val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH,
                shuffle=False,
                )

LR = train_config.learning_rate
model = model.to(device)
criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

coef_gamma = 0.7     # Коэффициент уменьшения lr
milestones = [20, 40, 60, 80]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=coef_gamma)

output_directory = train_config.tb_log_path
os.makedirs(output_directory, exist_ok=True)
writer = SummaryWriter(output_directory)

def iou_loss(pred, target, smooth=1, threshold=0.5):
    pred    = pred.view(-1)
    pred[pred>=threshold]=1
    pred[pred<threshold]=0
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou
#-----------------------------
def accuracy(y_pred, y):
    return 1-iou_loss(torch.sigmoid(y_pred), y, smooth=1)
#-----------------------------
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
#--------------------------
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
#-------------------
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

EPOCHS = train_config.num_epochs
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
    scheduler.step()

    if valid_loss[epoch] < best_valid_loss:
        best_valid_loss = valid_loss[epoch]
        best_epoch = epoch
        torch.save(model.state_dict(), train_config.model_save_path)

    epoch_mins, epoch_secs = epoch_time(start_time, time.monotonic())
    writer.add_scalar('training_loss', train_loss[epoch], epoch)
    writer.add_scalar('training_acc', train_acc[epoch], epoch)
    writer.add_scalar('val_loss', valid_loss[epoch], epoch)
    writer.add_scalar('val_acc', valid_acc[epoch], epoch)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss[epoch]:.3f} | Train Acc: {train_acc[epoch]*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss[epoch]:.3f} |  Val. Acc: {valid_acc[epoch]*100:.2f}%')