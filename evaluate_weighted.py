
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.dataset import RetinalDataset
from src.resnet_model import RetinalResNet18


# 1. Recreate the same validation split used during training
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

train_indices = []
val_indices = []

for grade, indices in grade_indices.items():
    random.shuffle(indices)
    split_point = int(len(indices) * 0.8)

    train_indices.extend(indices[:split_point])
    val_indices.extend(indices[split_point:])

random.shuffle(train_indices)
random.shuffle(val_indices)

val_dataset = Subset(dataset, val_indices)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

print(f"Validation images: {len(val_dataset)}")


# 2. Load the fine-tuned ResNet18 checkpoint
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = RetinalResNet18(
    num_classes=5,
    pretrained=False,
)

checkpoint_path = "models/retinal_resnet18_weighted_experiment.pth"

model.load_state_dict(
    torch.load(checkpoint_path, map_location=device)
)

model.to(device)
model.eval()

print(f"Loaded checkpoint: {checkpoint_path}")


# 3. Evaluate predictions and build the confusion matrix
num_classes = 5

confusion = torch.zeros(
    num_classes,
    num_classes,
    dtype=torch.long,
)

all_referral_probs = []
all_referral_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        for actual, predicted in zip(labels, predictions):
            confusion[actual.item(), predicted.item()] += 1

        # Probability that the image belongs to grades 2, 3, or 4
        referral_probs = probabilities[:, 2:].sum(dim=1)

        all_referral_probs.extend(
            referral_probs.cpu().tolist()
        )
        all_referral_labels.extend(
            (labels >= 2).cpu().tolist()
        )


# 4. Overall accuracy and per-grade precision/recall
print("\nConfusion matrix:")
print("Rows = actual grades; columns = predicted grades")
print(confusion)

correct = confusion.diag().sum().item()
total = confusion.sum().item()

print(f"\nOverall validation accuracy: {correct / total:.2%}")

print("\nPer-grade results:")

for grade in range(num_classes):
    true_positive = confusion[grade, grade].item()
    actual_count = confusion[grade, :].sum().item()
    predicted_count = confusion[:, grade].sum().item()

    recall = (
        true_positive / actual_count
        if actual_count else 0.0
    )

    precision = (
        true_positive / predicted_count
        if predicted_count else 0.0
    )

    print(
        f"Grade {grade}: "
        f"Recall = {recall:.2%}, "
        f"Precision = {precision:.2%}, "
        f"Actual images = {actual_count}"
    )


# 5. Preliminary referral metrics using the argmax prediction
# Grades 0-1 = referral-negative; grades 2-4 = referral-positive.
tn = confusion[:2, :2].sum().item()
fp = confusion[:2, 2:].sum().item()
fn = confusion[2:, :2].sum().item()
tp = confusion[2:, 2:].sum().item()

sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
specificity = tn / (tn + fp) if (tn + fp) else 0.0

print("\nPreliminary referral screening metrics (argmax):")
print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)
print(f"Sensitivity: {sensitivity:.2%}")
print(f"Specificity: {specificity:.2%}")


# 6. Explore probability thresholds for referral screening
referral_probs = torch.tensor(all_referral_probs)
referral_labels = torch.tensor(
    all_referral_labels,
    dtype=torch.bool,
)

results = []

for threshold_step in range(5, 96, 5):
    threshold = threshold_step / 100
    predicted_positive = referral_probs >= threshold

    tp = (predicted_positive & referral_labels).sum().item()
    fn = ((~predicted_positive) & referral_labels).sum().item()
    tn = ((~predicted_positive) & (~referral_labels)).sum().item()
    fp = (predicted_positive & (~referral_labels)).sum().item()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    results.append((threshold, sensitivity, specificity))

print("\nReferral threshold experiment:")
print("Threshold | Sensitivity | Specificity")

for threshold, sensitivity, specificity in results:
    print(
        f"{threshold:8.2f} | "
        f"{sensitivity:10.2%} | "
        f"{specificity:10.2%}"
    )

eligible = [result for result in results if result[1] >= 0.90]

if eligible:
    best = max(eligible, key=lambda result: result[2])

    print("\nBest tested threshold meeting 90% sensitivity:")
    print(f"Threshold: {best[0]:.2f}")
    print(f"Sensitivity: {best[1]:.2%}")
    print(f"Specificity: {best[2]:.2%}")
else:
    best = max(results, key=lambda result: result[1])

    print("\nNo tested threshold reached 90% sensitivity.")
    print(f"Highest tested sensitivity: {best[1]:.2%}")
    print(f"Corresponding threshold: {best[0]:.2f}")
    print(f"Specificity: {best[2]:.2%}")

print("\nEvaluation complete.")
print("Note: Threshold selection is exploratory, not clinical validation.")
