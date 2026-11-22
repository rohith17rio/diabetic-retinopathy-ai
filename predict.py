from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

from src.model import RetinalCNN


THRESHOLD = 0.55

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(image_path):
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    model = RetinalCNN()
    checkpoint = torch.load(
        "models/best_retinal_cnn.pth",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint)
    model.eval()

    with Image.open(path) as img:
        image = transform(img.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(image), dim=1)[0]

    grade = probabilities.argmax().item()
    referral_probability = probabilities[2:].sum().item()

    print(f"Predicted grade: {grade}")
    print(f"Grade probabilities: {[round(p.item(), 4) for p in probabilities]}")
    print(f"Provisional referral score: {referral_probability:.2%}")

    if referral_probability >= THRESHOLD:
        print("Model result: REFER FOR FURTHER REVIEW")
    else:
        print("Model result: BELOW EXPERIMENTAL REFERRAL THRESHOLD")

    print("\nExperimental prototype only — not for medical decisions.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/retinal_image.png")
        raise SystemExit(1)

    predict_image(sys.argv[1])
