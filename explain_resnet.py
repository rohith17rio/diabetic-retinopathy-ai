import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from src.resnet_model import RetinalResNet18


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

model = RetinalResNet18(num_classes=5, pretrained=False)
checkpoint = "models/retinal_resnet18_finetuned.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.to(device)
model.eval()

activations = []
gradients = []
target_layer = model.network.layer4[-1]


def save_activation(module, inputs, output):
    activations.append(output)


def save_activation(module, inputs, output):
    activations.append(output)
    output.register_hook(save_gradient)


def save_gradient(gradient):
    gradients.append(gradient)


target_layer.register_forward_hook(save_activation)


def explain_image(image_path):
    path = Path(image_path)

    if not path.is_file():
        print(f"Image not found: {path}")
        return

    original = Image.open(path).convert("RGB")
    input_tensor = (
    transform(original)
    .unsqueeze(0)
    .to(device)
    .requires_grad_(True)
)

    activations.clear()
    gradients.clear()
    model.zero_grad(set_to_none=True)

    # Gradients must be enabled for Grad-CAM.
    with torch.enable_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_grade = int(output.argmax(dim=1).item())
        output[0, predicted_grade].backward()

    feature_maps = activations[0].detach()
    gradient_maps = gradients[0].detach()

    weights = gradient_maps.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feature_maps).sum(dim=1, keepdim=True))
    cam = F.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    display_image = original.resize((224, 224))
    image_array = np.asarray(display_image)

    output_dir = Path("outputs/gradcam_resnet")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_gradcam.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_array)
    axes[0].set_title("Original retinal image")
    axes[0].axis("off")

    axes[1].imshow(image_array)
    axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title(f"Grad-CAM - Predicted grade {predicted_grade}")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Predicted grade: {predicted_grade}")
    print("Probabilities by grade:",
          probabilities[0].detach().cpu().numpy().round(4))
    print(f"Grad-CAM saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python explain_resnet.py PATH_TO_IMAGE")
    else:
        explain_image(sys.argv[1])
