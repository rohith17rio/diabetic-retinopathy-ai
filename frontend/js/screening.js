
const uploadInput = document.getElementById("retinal-image");
const uploadZone = document.getElementById("upload-zone");
const previewContainer = document.getElementById("image-preview");
const previewImage = document.getElementById("preview-image");
const fileName = document.getElementById("file-name");
const fileDetails = document.getElementById("file-details");
const removeButton = document.getElementById("remove-image");
const screeningForm = document.getElementById("screening-form");
const analyzeButton = document.getElementById("analyze-button");
const uploadMessage = document.getElementById("upload-message");

const MAX_FILE_SIZE = 10 * 1024 * 1024;

// If your frontend is served by FastAPI, keep this relative URL.
// If using VS Code Live Server on port 5500, see the instructions below.
const API_URL = "http://127.0.0.1:8000/api/analyze";

let selectedImage = null;
let previewUrl = null;

function showMessage(message) {
    uploadMessage.textContent = message;
}

function resetPreview() {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    selectedImage = null;
    uploadInput.value = "";
    previewImage.removeAttribute("src");
    fileName.textContent = "";
    fileDetails.textContent = "";
    previewContainer.hidden = true;
    uploadZone.hidden = false;
}

function handleImageSelection(file) {
    if (!file) return;

    if (!["image/jpeg", "image/png"].includes(file.type)) {
        resetPreview();
        showMessage("Select a JPG or PNG image.");
        return;
    }

    if (file.size === 0 || file.size > MAX_FILE_SIZE) {
        resetPreview();
        showMessage("Choose an image smaller than 10 MB.");
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
            resetPreview();
            showMessage("Image must be at least 224 × 224 pixels.");
            return;
        }

        fileName.textContent = file.name;
        fileDetails.textContent =
            `${previewImage.naturalWidth} × ${previewImage.naturalHeight} pixels · ` +
            `${(file.size / 1024).toFixed(1)} KB`;

        uploadZone.hidden = true;
        previewContainer.hidden = false;
        showMessage("Image ready for analysis.");
    };

    previewImage.onerror = () => {
        resetPreview();
        showMessage("Unable to read this image. Try another file.");
    };

    previewImage.src = previewUrl;
}

function createSmallPreview(file) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        const tempUrl = URL.createObjectURL(file);

        image.onload = () => {
            const maxDimension = 1200;
            const scale = Math.min(
                1,
                maxDimension / Math.max(image.naturalWidth, image.naturalHeight)
            );

            const canvas = document.createElement("canvas");
            canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
            canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));

            const context = canvas.getContext("2d");

            if (!context) {
                URL.revokeObjectURL(tempUrl);
                reject(new Error("Could not prepare the image preview."));
                return;
            }

            context.drawImage(image, 0, 0, canvas.width, canvas.height);

            const previewData = canvas.toDataURL("image/jpeg", 0.75);
            URL.revokeObjectURL(tempUrl);
            resolve(previewData);
        };

        image.onerror = () => {
            URL.revokeObjectURL(tempUrl);
            reject(new Error("Could not read the selected image."));
        };

        image.src = tempUrl;
    });
}

uploadInput.addEventListener("change", event => {
    handleImageSelection(event.target.files[0]);
});

removeButton.addEventListener("click", () => {
    resetPreview();
    showMessage("Image removed. Choose another image to continue.");
});

screeningForm.addEventListener("submit", async event => {
    event.preventDefault();

    if (!selectedImage) {
        showMessage("Please select a retinal image first.");
        return;
    }

    const imageToAnalyze = selectedImage;
    const originalButtonText = analyzeButton.innerHTML;

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing image...";
    showMessage("Sending image to the AI model. Please wait...");

    try {
        const previewData = await createSmallPreview(imageToAnalyze);

        const formData = new FormData();
        formData.append("image", imageToAnalyze);

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        const responseText = await response.text();
        let result;

        try {
            result = JSON.parse(responseText);
        } catch {
            console.error("API URL:", API_URL);
            console.error("HTTP status:", response.status);
            console.error("Response received:", responseText.slice(0, 500));

            throw new Error(
                `Invalid response (HTTP ${response.status}). Check the browser console.`
            );
        }

        if (!response.ok) {
            throw new Error(
                result.detail || `Analysis failed (${response.status}).`
            );
        }

        if (
            !Number.isInteger(Number(result.predicted_grade)) ||
            Number(result.predicted_grade) < 0 ||
            Number(result.predicted_grade) > 4
        ) {
            throw new Error("The API did not return a valid predicted grade.");
        }

        // Save this result for the separate results page.
        // The resized preview keeps browser storage smaller.
        sessionStorage.setItem("retinaCareResult", JSON.stringify(result));
        sessionStorage.setItem("retinaCarePreview", previewData);
        sessionStorage.setItem("retinaCareFileName", imageToAnalyze.name);

        window.location.href = "results.html";
    } catch (error) {
        showMessage(
            `Analysis failed: ${error.message} Check that your FastAPI backend is running.`
        );
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.innerHTML = originalButtonText;
    }
});
