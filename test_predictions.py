import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.model import RetinalCNN

IMAGE_DIR = "data/train_images"
CSV_FILE = "data/train.csv"
MODEL_FILE = "models/best_retinal_cnn.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = torch.device("cpu")

model = RetinalCNN().to(device)
checkpoint = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

df = pd.read_csv(CSV_FILE).head(5)

correct = 0

print("\nPrediction comparison")
print("-" * 65)

with torch.no_grad():
    for _, row in df.iterrows():
        image_id = row["id_code"]
        actual_grade = int(row["diagnosis"])
        image_path = f"{IMAGE_DIR}/{image_id}.png"

        with Image.open(image_path) as img:
            image = transform(img.convert("RGB")).unsqueeze(0)

        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_grade = probabilities.argmax(dim=1).item()

        is_correct = predicted_grade == actual_grade
        correct += int(is_correct)

        result = "CORRECT" if is_correct else "INCORRECT"

        print(f"Image: {image_id}")
        print(f"Actual grade: {actual_grade}")
        print(f"Predicted grade: {predicted_grade}")
        print(f"Result: {result}")
        print("-" * 65)

print(f"Correct predictions: {correct}/5")
print(f"Accuracy on these 5 images: {correct / 5:.2%}")
print("\nNote: This is a small demonstration, not a reliable model evaluation.")
