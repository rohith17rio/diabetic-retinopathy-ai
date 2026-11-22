# Explainable AI for Diabetic Retinopathy Screening

## Project Overview

Diabetic Retinopathy is an eye condition caused by diabetes that can lead to vision loss if not detected early. This project aims to develop a Deep Learning-based system to classify diabetic retinopathy severity from retinal fundus images and provide visual explanations for model predictions.

## Current Implementation

- **Deep Learning Model:** Uses a fine-tuned ResNet18 model to classify retinal images into diabetic retinopathy severity grades (0–4).
- **Severity Classification:** Predicts the severity grade from an input retinal image.
- **Explainable AI:** Uses Grad-CAM heatmaps to highlight image regions that influence the model's prediction.
- **Frontend:** Provides an interface for retinal image upload and displaying analysis results.
- **Backend:** Uses FastAPI to connect the frontend with the prediction model.

## Severity Grades

| Grade | Description |
|---|---|
| 0 | No Diabetic Retinopathy |
| 1 | Mild Non-Proliferative Diabetic Retinopathy |
| 2 | Moderate Non-Proliferative Diabetic Retinopathy |
| 3 | Severe Non-Proliferative Diabetic Retinopathy |
| 4 | Proliferative Diabetic Retinopathy |

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Grad-CAM
- FastAPI
- Uvicorn
- HTML, CSS, JavaScript
- NumPy, Pandas, Pillow, OpenCV

## Project Structure

```text
Diabetic-Retinopathy-AI/
├── backend/
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── dashboard.html
│   ├── results.html
│   ├── about.html
│   ├── app.js
│   ├── css/
│   └── js/
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── resnet_model.py
├── outputs/
│   ├── gradcam/
│   └── gradcam_resnet/
├── train.py
├── train_resnet.py
├── train_resnet_finetune.py
├── predict.py
├── explain.py
├── explain_resnet.py
├── evaluate.py
├── compare_models.py
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/rohith17rio/diabetic-retinopathy-ai.git
cd diabetic-retinopathy-ai
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Obtain an appropriately licensed retinal fundus image dataset and place the training images and CSV labels in the directory structure expected by the dataset loader.

### 5. Run the backend

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 6. Run the frontend

Open a separate terminal and run:

```bash
python3 -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500/index.html` in your browser.

> **Note:** The trained model checkpoint must be available at the path expected by the backend before prediction can work. The dataset and trained checkpoint are not necessarily included in the repository.

## Explainable AI with Grad-CAM

Grad-CAM generates a heatmap that highlights image regions contributing to the model's prediction. This provides a visual explanation of the model's output; it does not prove that the prediction is medically correct.

## Current Project Status

The project is at the functional prototype development stage. The initial deep learning classification model, Grad-CAM visualization, frontend, and backend have been developed. Further model testing, performance improvement, and clinical validation are still required.

## Limitations

- Predictions may be incorrect, particularly for some severity grades.
- The model is a research prototype and is not clinically validated.
- Grad-CAM heatmaps are explanatory visualizations, not medical confirmation.
- The system must not be used as a substitute for professional ophthalmic diagnosis.

## Future Improvements

- Improve grade-wise classification performance.
- Evaluate sensitivity, specificity, and false-negative cases.
- Test with diverse, expert-graded retinal images.
- Improve retinal image quality assessment.
- Conduct validation with ophthalmologists before any clinical deployment.

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical device and must not be used to diagnose, treat, or make clinical decisions about patients.
