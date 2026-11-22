
from torchvision import transforms
from src.dataset import RetinalDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = RetinalDataset(
    image_dir="data/train_images",
    csv_file="data/train.csv",
    transform=transform,
)

image, label = dataset[0]

print("Dataset size:", len(dataset))
print("Image tensor shape:", image.shape)
print("Image data type:", image.dtype)
print("Minimum pixel value:", image.min().item())
print("Maximum pixel value:", image.max().item())
print("Diagnosis grade:", label.item())
