const uploadInput = document.getElementById("retinal-image");
const uploadZone = document.querySelector(".upload-zone");
const previewContainer = document.getElementById("image-preview");
const previewImage = document.getElementById("preview-image");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const removeButton = document.getElementById("remove-image");
const screeningForm = document.getElementById("screening-form");
const analyzeButton = document.getElementById("analyze-button");
const resultMessage = document.getElementById("result-message");
const resultsContent = document.getElementById("results-content");

let selectedImage = null;
let previewUrl = null;

const MAX_FILE_SIZE = 10 * 1024 * 1024;

function showMessage(message) {
    resultMessage.textContent = message;
}

function clearPreview() {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    selectedImage = null;
    uploadInput.value = "";
    previewImage.removeAttribute("src");
    fileName.textContent = "";
    fileSize.textContent = "";
    previewContainer.hidden = true;
    uploadZone.hidden = false;
}

function handleImageSelection(file) {
    if (!file) return;

    if (!["image/jpeg", "image/png"].includes(file.type)) {
        clearPreview();
        showMessage("Please select a JPG or PNG retinal image.");
        return;
    }

    if (file.size === 0 || file.size > MAX_FILE_SIZE) {
        clearPreview();
        showMessage("Choose a valid image smaller than 10 MB.");
        return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);

    selectedImage = file;
    previewUrl = URL.createObjectURL(file);

    previewImage.onload = () => {
        if (
            previewImage.naturalWidth < 224 ||
            previewImage.naturalHeight < 224
        ) {
            clearPreview();
            showMessage("Please choose an image at least 224 × 224 pixels.");
            return;
        }

        fileName.textContent = file.name;
        fileSize.textContent =
            `${previewImage.naturalWidth} × ${previewImage.naturalHeight} pixels · ${(file.size / 1024).toFixed(1)} KB`;

        uploadZone.hidden = true;
        previewContainer.hidden = false;
        resultsContent.hidden = true;

        showMessage("Image ready. Click Analyze to run the AI model.");
    };

    previewImage.onerror = () => {
        clearPreview();
        showMessage("Unable to read this image. Please select another.");
    };

    previewImage.src = previewUrl;
}

uploadInput.addEventListener("change", (event) => {
    handleImageSelection(event.target.files[0]);
});

removeButton.addEventListener("click", () => {
    clearPreview();
    resultsContent.hidden = true;
    showMessage("Image removed. Select a retinal image to begin.");
});

screeningForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!selectedImage) {
        showMessage("Please upload a retinal image first.");
        return;
    }

    const originalButtonText = analyzeButton.textContent;
    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing image...";
    resultsContent.hidden = true;
    showMessage("The AI model is analyzing your image. Please wait...");

    try {
        const formData = new FormData();
        formData.append("image", selectedImage);

        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Image analysis failed.");
        }

        document.getElementById("predicted-grade").textContent =
            `Grade ${result.predicted_grade}`;

        document.getElementById("grade-description").textContent =
            result.grade_description;

        document.getElementById("prediction-confidence").textContent =
            `${(result.probabilities[result.predicted_grade] * 100).toFixed(1)}%`;

        const gradcamImage = document.getElementById("gradcam-image");
        gradcamImage.src = result.gradcam_image;
        gradcamImage.alt =
            "Grad-CAM heatmap showing image regions influencing the model prediction";

        resultsContent.hidden = false;

        let message = "AI analysis completed. This is a research result, not a diagnosis.";

        if (result.referral_recommended_by_model) {
            message += " The model's referral threshold was reached; professional eye assessment is recommended.";
        } else {
            message += " The model's referral threshold was not reached. This does not rule out diabetic retinopathy.";
        }

        if (result.quality && result.quality.warnings.length > 0) {
            message += " Image quality warning: " +
                result.quality.warnings.join(" ");
        }

        showMessage(message);

        resultsContent.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (error) {
        showMessage(
            `Analysis failed: ${error.message}. Check that the backend is running.`
        );
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.textContent = originalButtonText;
    }
});

document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", () => {
        document.querySelectorAll(".nav-link").forEach((item) => {
            item.classList.remove("active");
        });

        link.classList.add("active");
    });
});
