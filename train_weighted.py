from pathlib import Path
import copy

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.dataset import RetinalDataset
from src.resnet_model import RetinalResNet18

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "train.csv"
IMAGE_DIR = ROOT / "data" / "train_images"
OUTPUT_PATH = ROOT / "models" / "retinal_resnet18_weighted_experiment.pth"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001

torch.manual_seed(SEED)
device = torch.device("cpu")
print("Training device:", device)

labels_df = pd.read_csv(CSV_PATH)
labels_df["image_name"] = labels_df["id_code"].astype(str) + ".png"

indices = list(range(len(labels_df)))
grades = labels_df["diagnosis"].astype(int).tolist()

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=grades,
)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# This dataset class expects a CSV path, not a Python list.
train_dataset = RetinalDataset(
    image_dir=IMAGE_DIR,
    csv_file=CSV_PATH,
    transform=train_transform,
)
val_dataset = RetinalDataset(
    image_dir=IMAGE_DIR,
    csv_file=CSV_PATH,
    transform=val_transform,
)

train_loader = DataLoader(
    Subset(train_dataset, train_indices),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    Subset(val_dataset, val_indices),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

counts = torch.bincount(
    torch.tensor(grades)[train_indices], minlength=5
).float()

weights = counts.sum() / (5 * counts)
weights = (weights / weights.mean()).to(device)

print("Training grade counts:", counts.int().tolist())
print("Class weights:", [round(v, 3) for v in weights.tolist()])

model = RetinalResNet18(num_classes=5, pretrained=False).to(device)
original_checkpoint = ROOT / "models" / "retinal_resnet18_finetuned.pth"
model.load_state_dict(torch.load(original_checkpoint, map_location=device))

for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in model.network.layer4.parameters():
    parameter.requires_grad = True

for parameter in model.network.fc.parameters():
    parameter.requires_grad = True

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
)

best_val_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, target in train_loader:
        images, target = images.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * target.size(0)
        train_correct += (outputs.argmax(1) == target).sum().item()
        train_total += target.size(0)

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.to(device), target.to(device)
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, target)
            val_loss_sum += loss.item() * target.size(0)
            val_correct += (outputs.argmax(1) == target).sum().item()
            val_total += target.size(0)

    val_loss = val_loss_sum / val_total
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train accuracy: {100 * train_correct / train_total:.2f}% | "
        f"Validation accuracy: {100 * val_correct / val_total:.2f}% | "
        f"Validation loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

if best_state is not None:
    torch.save(best_state, OUTPUT_PATH)
    print("Saved experimental model:", OUTPUT_PATH)
    print("Original model was not overwritten.")
