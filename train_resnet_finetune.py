import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms

from src.dataset import RetinalDataset
from src.resnet_model import RetinalResNet18


random.seed(42)
torch.manual_seed(42)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_base = RetinalDataset(
    image_dir="data/train_images",
    csv_file="data/train.csv",
    transform=train_transform,
)

val_base = RetinalDataset(
    image_dir="data/train_images",
    csv_file="data/train.csv",
    transform=val_transform,
)

# Recreate the same stratified 80/20 split.
grade_indices = defaultdict(list)

for index, grade in enumerate(train_base.labels_df["diagnosis"]):
    grade_indices[int(grade)].append(index)

train_indices = []
val_indices = []

for grade, indices in grade_indices.items():
    random.shuffle(indices)
    split_point = int(len(indices) * 0.8)
    train_indices.extend(indices[:split_point])
    val_indices.extend(indices[split_point:])

random.shuffle(train_indices)
random.shuffle(val_indices)

train_dataset = Subset(train_base, train_indices)
val_dataset = Subset(val_base, val_indices)

# Balance grade frequencies during training.
train_labels = train_base.labels_df.iloc[train_indices][
    "diagnosis"
].to_numpy()
train_labels = torch.tensor(train_labels, dtype=torch.long)

class_counts = torch.bincount(train_labels, minlength=5).float()
class_weights = 1.0 / class_counts.clamp(min=1)
sample_weights = class_weights[train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True,
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=sampler,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

# Load the already-trained ResNet18 model.
model = RetinalResNet18(
    num_classes=5,
    pretrained=False,
)

checkpoint = torch.load(
    "models/retinal_resnet18_transfer.pth",
    map_location="cpu",
)
model.load_state_dict(checkpoint)

# Unfreeze only the final convolutional block and classifier.
for param in model.parameters():
    param.requires_grad = False

for param in model.network.layer4.parameters():
    param.requires_grad = True

for param in model.network.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([
    {
        "params": model.network.layer4.parameters(),
        "lr": 0.0001,
    },
    {
        "params": model.network.fc.parameters(),
        "lr": 0.0005,
    },
])

epochs = 5
save_path = Path("models/retinal_resnet18_finetuned.pth")
save_path.parent.mkdir(parents=True, exist_ok=True)
best_val_loss = float("inf")

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Class counts:", class_counts.tolist())
print("Fine-tuning epochs:", epochs)
print("Trainable parameters:", sum(
    p.numel() for p in model.parameters() if p.requires_grad
))
print("Experiment: ResNet18 layer4 + classifier fine-tuning")

for epoch in range(epochs):
    model.train()

    # Keep BatchNorm statistics in frozen blocks unchanged.
    model.network.eval()
    model.network.layer4.train()
    model.network.fc.train()

    train_loss_total = 0.0
    train_correct = 0
    train_count = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        train_loss_total += loss.item() * batch_size
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_count += batch_size

    train_loss = train_loss_total / train_count
    train_accuracy = train_correct / train_count

    model.eval()
    val_loss_total = 0.0
    val_correct = 0
    val_count = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            val_loss_total += loss.item() * batch_size
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            val_count += batch_size

    val_loss = val_loss_total / val_count
    val_accuracy = val_correct / val_count

    print(f"\nEpoch {epoch + 1}/{epochs}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Train accuracy: {train_accuracy:.2%}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.2%}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print("Best fine-tuned checkpoint saved.")

print("\nFine-tuning complete!")
print("Best model path:", save_path)
