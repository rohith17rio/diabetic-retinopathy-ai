from pathlib import Path
from io import BytesIO
import base64

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms

from src.resnet_model import RetinalResNet18


ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "retinal_resnet18_weighted_experiment.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FILE_SIZE = 10 * 1024 * 1024
REFERRAL_THRESHOLD = 0.60

GRADE_NAMES = {
    0: "No diabetic retinopathy detected by the model",
    1: "Mild diabetic retinopathy",
    2: "Moderate diabetic retinopathy",
    3: "Severe diabetic retinopathy",
    4: "Proliferative diabetic retinopathy",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

app = FastAPI(
    title="RetinaCare AI API",
    description="Research prototype for explainable retinal image screening",
    version="1.0.0",
)

if not MODEL_PATH.is_file():
    raise RuntimeError(
        f"Model checkpoint not found: {MODEL_PATH}. "
        "Check that your fine-tuned model exists."
    )

model = RetinalResNet18(num_classes=5, pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

# Grad-CAM hooks for the final ResNet18 convolutional block.
activations = []
gradients = []
target_layer = model.network.layer4[-1]


def save_gradient(gradient):
    gradients.append(gradient)


def save_activation(module, inputs, output):
    activations.append(output)
    if output.requires_grad:
        output.register_hook(save_gradient)


target_layer.register_forward_hook(save_activation)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "Fine-tuned ResNet18",
        "device": str(DEVICE),
    }


@app.get("/")
def homepage():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/analyze")
async def analyze_image(image: UploadFile = File(...)):
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, JPEG, or PNG retinal image.",
        )

    raw_bytes = await image.read(MAX_FILE_SIZE + 1)

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        original = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a readable image.",
        )

    width, height = original.size

    if width < 224 or height < 224:
        raise HTTPException(
            status_code=400,
            detail="Image resolution is too small. Please upload an image at least 224 × 224 pixels.",
        )

    # Basic quality screening: flag possible blur or extreme brightness.
    rgb_array = np.asarray(original)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(gray.mean())

    quality_warnings = []

    if blur_score < 50:
        quality_warnings.append(
            "The image may be blurry. Consider retaking the photograph."
        )

    if mean_brightness < 35:
        quality_warnings.append(
            "The image may be too dark. Consider improving the lighting."
        )
    elif mean_brightness > 220:
        quality_warnings.append(
            "The image may be overexposed. Consider retaking the photograph."
        )

    input_tensor = TRANSFORM(original).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(True)

    activations.clear()
    gradients.clear()
    model.zero_grad(set_to_none=True)

    # Gradients are required to calculate Grad-CAM.
    with torch.enable_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_grade = int(logits.argmax(dim=1).item())
        referral_probability = float(probabilities[0, 2:].sum().item())
        logits[0, predicted_grade].backward()

    if not activations or not gradients:
        raise HTTPException(
            status_code=500,
            detail="Could not generate the model explanation.",
        )

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

    # Return a combined original image and Grad-CAM heatmap as a PNG.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_image = original.resize((224, 224))
    image_array = np.asarray(display_image)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(image_array)
    axes[0].set_title("Original retinal image")
    axes[0].axis("off")

    axes[1].imshow(image_array)
    axes[1].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM explanation")
    axes[1].axis("off")

    figure.tight_layout()

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)

    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "predicted_grade": predicted_grade,
        "grade_description": GRADE_NAMES[predicted_grade],
        "probabilities": {
            str(index): round(float(probabilities[0, index].item()), 6)
            for index in range(5)
        },
        "referral_probability": round(referral_probability, 6),
        "referral_threshold": REFERRAL_THRESHOLD,
        "referral_recommended_by_model": (
            referral_probability >= REFERRAL_THRESHOLD
        ),
        "quality": {
            "width": width,
            "height": height,
            "blur_score": round(blur_score, 2),
            "mean_brightness": round(mean_brightness, 2),
            "warnings": quality_warnings,
        },
        "gradcam_image": "data:image/png;base64," + heatmap_base64,
        "disclaimer": (
            "Research prototype only. Predictions and quality checks may be "
            "incorrect. This result is not a medical diagnosis."
        ),
    }


app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend",
)
