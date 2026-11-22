
import torch
from src.dataset import RetinalDataset

images = [
    torch.rand(3, 4, 4),
    torch.rand(3, 4, 4),
    torch.rand(3, 4, 4),
]

grades = [0, 2, 4]

dataset = RetinalDataset(images, grades)

print("Number of images:", len(dataset))

image, grade = dataset[1]

print("Image shape:", image.shape)
print("Grade:", grade.item())
