# %%
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms.v2 as v2
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from torch.utils.data.dataloader import DataLoader
from sklearn.utils import shuffle
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# %%
TRAIN_PATH = "data/train"
VAL_PATH   = "data/val"
TEST_PATH  = "data/test"

SIZE = 224
BATCH_SIZE = 64
EPOCHS = 3
PATIENCE = 5
LR = 1e-5

# %%
def show_all_images_counts(all_path):
    all_images_counts = {}

    for path in all_path:
        if not os.path.exists(path):
            print("Missing:", path)
            continue

        img_counts = {}
        for label in os.listdir(path):
            label_path = os.path.join(path, label)

            if os.path.isdir(label_path):
                img_counts[label] = len(os.listdir(label_path))

        all_images_counts[os.path.basename(path)] = img_counts

    return all_images_counts


all_counts = show_all_images_counts([TRAIN_PATH, VAL_PATH, TEST_PATH])

print(all_counts["train"])
print(all_counts["val"])
print(all_counts["test"])

# %%
def get_transforms(train=False):
    base = [
        v2.Resize((SIZE, SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    aug = [
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(20),
    ]

    # ✅ FIX: 3-channel for ResNet
    tail = [
        v2.Grayscale(num_output_channels=3),
    ]

    if train:
        return v2.Compose(base + aug + tail)
    return v2.Compose(base + tail)


# %%
train_ds = ImageFolder(TRAIN_PATH, transform=get_transforms(train=True))
val_ds   = ImageFolder(VAL_PATH, transform=get_transforms())
test_ds  = ImageFolder(TEST_PATH, transform=get_transforms())

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(len(train_ds), len(val_ds), len(test_ds))

# %%
def build_model():
    model = resnet50(weights="DEFAULT")

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )

    return model.to(device)


# %%
def fit():
    model = build_model()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0
    counter = 0
    history = []
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()

        train_loss = []
        train_preds, train_labels = [], []

        for imgs, labels in tqdm(train_dl):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            use_cuda = torch.cuda.is_available()
            with torch.autocast(device_type='cuda', enabled=use_cuda):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            probs = torch.softmax(outputs, dim=1)

            train_loss.append(loss.item())
            train_preds.append(probs[:, 1].detach().cpu())
            train_labels.append(labels.detach().cpu())

        train_auc = roc_auc_score(
            torch.cat(train_labels),
            torch.cat(train_preds)
        )

        # VAL
        model.eval()
        val_loss = []
        val_preds, val_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)

                use_cuda = torch.cuda.is_available()
                with torch.autocast(device_type='cuda', enabled=use_cuda):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)

                val_loss.append(loss.item())
                val_preds.append(probs[:, 1].cpu())
                val_labels.append(labels.cpu())

        val_auc = roc_auc_score(
            torch.cat(val_labels),
            torch.cat(val_preds)
        )

        history.append({
            "train_loss": np.mean(train_loss),
            "val_loss": np.mean(val_loss),
            "train_auc": train_auc,
            "val_auc": val_auc
        })

        print(f"Epoch {epoch+1} | Train AUC {train_auc:.4f} | Val AUC {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping")
                break

    return model, history, best_epoch


# %%
model, history, best_epoch = fit()

print("Best epoch:", best_epoch)

# %%
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()


# %%
def predict(model, loader):
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    return y_true, y_pred


labels, preds = predict(model, test_dl)

# %%
# Sample visualization
test_imgs, test_labels = next(iter(test_dl))

plt.figure(figsize=(10, 10))
grid = make_grid(test_imgs[:16], nrow=4).permute(1, 2, 0)

plt.imshow(grid)
plt.axis("off")
plt.show()
