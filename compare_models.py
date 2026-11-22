import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.dataset import RetinalDataset
from src.resnet_model import RetinalResNet18

random.seed(42)
torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = RetinalDataset(
    image_dir="data/train_images",
    csv_file="data/train.csv",
    transform=transform,
)

grade_indices = defaultdict(list)
for index, grade in enumerate(dataset.labels_df["diagnosis"]):
    grade_indices[int(grade)].append(index)

train_indices, val_indices = [], []
for grade, indices in grade_indices.items():
    random.shuffle(indices)
    split_point = int(len(indices) * 0.8)
    train_indices.extend(indices[:split_point])
    val_indices.extend(indices[split_point:])

random.shuffle(train_indices)
random.shuffle(val_indices)

loader = DataLoader(
    Subset(dataset, val_indices),
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for checkpoint_path in [
    "models/retinal_resnet18_finetuned.pth",
    "models/retinal_resnet18_weighted_experiment.pth",
]:
    model = RetinalResNet18(num_classes=5, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    confusion = torch.zeros(5, 5, dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()
            for actual, predicted in zip(labels, predictions):
                confusion[int(actual), int(predicted)] += 1

    accuracy = confusion.diag().sum().item() / confusion.sum().item()
    print(f"\nModel: {checkpoint_path}")
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion)

    for grade in range(5):
        actual_count = confusion[grade].sum().item()
        predicted_count = confusion[:, grade].sum().item()
        recall = confusion[grade, grade].item() / actual_count if actual_count else 0
        precision = confusion[grade, grade].item() / predicted_count if predicted_count else 0
        print(
            f"Grade {grade}: recall={recall * 100:.2f}%, "
            f"precision={precision * 100:.2f}%"
        )

print("\nComparison complete. This is a research experiment, not clinical validation.")
