import random
from collections import defaultdict
from torch.utils.data import Subset

from src.dataset import RetinalDataset

dataset = RetinalDataset(
    image_dir="data/train_images",
    csv_file="data/train.csv",
)

# Group image indices by their diagnosis grade
grade_indices = defaultdict(list)

for index, grade in enumerate(dataset.labels_df["diagnosis"]):
    grade_indices[int(grade)].append(index)

random.seed(42)
train_indices = []
val_indices = []

# Split each grade separately to preserve class proportions
for grade, indices in grade_indices.items():
    random.shuffle(indices)
    split_point = int(len(indices) * 0.8)

    train_indices.extend(indices[:split_point])
    val_indices.extend(indices[split_point:])

random.shuffle(train_indices)
random.shuffle(val_indices)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

print("Total images:", len(dataset))
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

print("\nTraining grade counts:")
print(dataset.labels_df.iloc[train_indices]["diagnosis"].value_counts().sort_index())

print("\nValidation grade counts:")
print(dataset.labels_df.iloc[val_indices]["diagnosis"].value_counts().sort_index())

print("\nDataset split test successful!")
