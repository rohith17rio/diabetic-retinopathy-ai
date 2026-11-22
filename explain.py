from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import RetinalCNN


MODEL_PATH = "models/best_retinal_cnn.pth"
OUTPUT_DIR = Path("outputs/gradcam")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def explain_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = RetinalCNN()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    # Use the final convolutional layer for Grad-CAM.
    target_layer = model.features[6]
    activations = []
    gradients = []

    def save_activation(module, inputs, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    activation_hook = target_layer.register_forward_hook(save_activation)
    gradient_hook = target_layer.register_full_backward_hook(save_gradient)

    try:
        with Image.open(image_path) as img:
            original_image = img.convert("RGB").resize((224, 224))
            image_tensor = transform(original_image).unsqueeze(0)

        # Forward pass: obtain the model's scores.
        outputs = model(image_tensor)
        predicted_grade = outputs.argmax(dim=1).item()

        # Backward pass: calculate gradients for the predicted grade.
        model.zero_grad()
        outputs[0, predicted_grade].backward()

        feature_maps = activations[0]
        gradient_maps = gradients[0]

        # Average gradients across the spatial dimensions.
        weights = gradient_maps.mean(dim=(2, 3), keepdim=True)

        # Combine feature maps using their gradient weights.
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize heat map to match the input image.
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().numpy()

        # Normalize heat-map values between 0 and 1.
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        probabilities = torch.softmax(outputs.detach(), dim=1)[0]
        grade_probabilities = [
            round(p.item() * 100, 2) for p in probabilities
        ]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{image_path.stem}_gradcam.png"

        # Display original image and heat-map overlay side by side.
        original_array = np.asarray(original_image)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(original_array)
        axes[0].set_title("Original retinal image")
        axes[0].axis("off")

        axes[1].imshow(original_array)
        axes[1].imshow(cam, cmap="jet", alpha=0.45)
        axes[1].set_title(f"Grad-CAM: predicted grade {predicted_grade}")
        axes[1].axis("off")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Image: {image_path}")
        print(f"Predicted grade: {predicted_grade}")
        print(f"Grade probabilities (%): {grade_probabilities}")
        print(f"Grad-CAM image saved to: {output_path}")
        print("Experimental explanation only; not a medical diagnosis.")

    finally:
        activation_hook.remove()
        gradient_hook.remove()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python explain.py path/to/retinal_image.png")
        raise SystemExit(1)

    explain_image(sys.argv[1])
