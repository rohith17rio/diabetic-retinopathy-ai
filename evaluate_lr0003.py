import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.dataset import RetinalDataset
from src.model import RetinalCNN


# 1. Recreate the validation split used during training
random.seed(42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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

# 2. Load the trained model
model = RetinalCNN()

model.load_state_dict(
    torch.load(
        "models/retinal_cnn_lr0003.pth",
        map_location="cpu",
    )
)

model.eval()

# 3. Build the confusion matrix
num_classes = 5

confusion = torch.zeros(
    num_classes,
    num_classes,
    dtype=torch.long,
)

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for actual, predicted in zip(labels, predictions):
            confusion[actual.item(), predicted.item()] += 1

# 4. Display evaluation results
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
        if actual_count > 0 else 0.0
    )

    precision = (
        true_positive / predicted_count
        if predicted_count > 0 else 0.0
    )

    print(
        f"Grade {grade}: "
        f"Recall = {recall:.2%}, "
        f"Precision = {precision:.2%}, "
        f"Actual images = {actual_count}"
    )

print("\nEvaluation complete!")

# 4. Preliminary binary referral evaluation
# Provisional rule: grades 2-4 = referral-positive
tn = confusion[:2, :2].sum().item()
fp = confusion[:2, 2:].sum().item()
fn = confusion[2:, :2].sum().item()
tp = confusion[2:, 2:].sum().item()

sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
specificity = tn / (tn + fp) if (tn + fp) else 0.0

print("\nPreliminary referral screening metrics:")
print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)
print(f"Sensitivity: {sensitivity:.2%}")
print(f"Specificity: {specificity:.2%}")

# 5. Explore probability thresholds for referral screening
# Provisional rule: grades 2-4 are referral-positive.
# This is exploratory threshold tuning, not final validation.

all_referral_probs = []
all_referral_labels = []

model.eval()

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        # Add probabilities for grades 2, 3, and 4
        referral_probs = probabilities[:, 2:].sum(dim=1)

        all_referral_probs.extend(referral_probs.tolist())
        all_referral_labels.extend((labels >= 2).tolist())

referral_probs = torch.tensor(all_referral_probs)
referral_labels = torch.tensor(all_referral_labels, dtype=torch.bool)

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

eligible = [r for r in results if r[1] >= 0.90]

if eligible:
    best = max(eligible, key=lambda r: r[2])
    print("\nBest tested threshold meeting 90% sensitivity:")
    print(f"Threshold: {best[0]:.2f}")
    print(f"Sensitivity: {best[1]:.2%}")
    print(f"Specificity: {best[2]:.2%}")
else:
    best = max(results, key=lambda r: r[1])
    print("\nNo tested threshold reached 90% sensitivity.")
    print(f"Highest tested sensitivity: {best[1]:.2%}")
    print(f"Corresponding threshold: {best[0]:.2f}")
    print(f"Specificity: {best[2]:.2%}")

print("\nThreshold experiment complete.")
