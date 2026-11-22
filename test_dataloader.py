import torch
from torch.utils.data import Dataset, DataLoader

# 1. Create a small fake dataset
class FakeRetinalDataset(Dataset):
    def __init__(self):
        self.images = [torch.rand(3, 4, 4) for _ in range(5)]
        self.grades = [0, 1, 2, 3, 4]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.grades[index]


# 2. Create the dataset
dataset = FakeRetinalDataset()

# 3. Create the DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# 4. Read each batch
for batch_number, (image_batch, grade_batch) in enumerate(loader, start=1):
    print(f"Batch {batch_number}")
    print("Image shape:", image_batch.shape)
    print("Grades:", grade_batch.tolist())