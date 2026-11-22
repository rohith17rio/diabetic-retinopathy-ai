
const emptyState = document.getElementById("empty-state");
const resultsContent = document.getElementById("results-content");

const gradeDescriptions = {
    0: "No diabetic retinopathy detected by the model.",
    1: "Mild non-proliferative diabetic retinopathy prediction.",
    2: "Moderate non-proliferative diabetic retinopathy prediction.",
    3: "Severe non-proliferative diabetic retinopathy prediction.",
    4: "Proliferative diabetic retinopathy prediction."
};

function setText(id, value) {
    document.getElementById(id).textContent = value;
}

function formatProbability(value) {
    const probability = Number(value);

    if (!Number.isFinite(probability)) return "—";

    // The API is expected to return probabilities from 0 to 1.
    return `${(Math.max(0, Math.min(1, probability)) * 100).toFixed(1)}%`;
}

function showImage(element, source) {
    if (typeof source !== "string" || !source.trim()) return false;

    element.src = source;
    return true;
}

function renderResults() {
    const rawResult = sessionStorage.getItem("retinaCareResult");
    const originalPreview = sessionStorage.getItem("retinaCarePreview");

    if (!rawResult || !originalPreview) {
        emptyState.hidden = false;
        resultsContent.hidden = true;
        return;
    }

    let result;

    try {
        result = JSON.parse(rawResult);
    } catch {
        emptyState.hidden = false;
        resultsContent.hidden = true;
        return;
    }

    const grade = Number(result.predicted_grade);

    if (!Number.isInteger(grade) || grade < 0 || grade > 4) {
        emptyState.hidden = false;
        resultsContent.hidden = true;
        return;
    }

    emptyState.hidden = true;
    resultsContent.hidden = false;

    setText("grade-title", `Grade ${grade}`);
    setText(
        "grade-description",
        result.grade_description || gradeDescriptions[grade]
    );
    setText("grade-number", String(grade));
    setText("metric-grade", `${grade} / 4`);

    const probabilities = Array.isArray(result.probabilities)
        ? result.probabilities
        : [];

    setText(
        "metric-confidence",
        formatProbability(probabilities[grade])
    );

    const referral = result.referral_recommended_by_model;

    if (typeof referral === "boolean") {
        setText("metric-referral", referral ? "Flagged" : "Not flagged");
        setText(
            "referral-details",
            referral
                ? "Model threshold reached; seek professional assessment."
                : "Threshold not reached; disease is not ruled out."
        );
    } else {
        setText("metric-referral", "Unavailable");
        setText("referral-details", "Referral flag not provided by the API.");
    }

    showImage(document.getElementById("original-image"), originalPreview);

    const heatmapCard = document.getElementById("heatmap-card");
    const heatmapMessage = document.getElementById("heatmap-message");
    const gradcamImage = document.getElementById("gradcam-image");

    const heatmap = result.gradcam_image;

    if (typeof heatmap === "string" && heatmap.trim()) {
        gradcamImage.onerror = () => {
            heatmapCard.hidden = true;
            heatmapMessage.textContent =
                "The explanation image could not be loaded.";
        };

        gradcamImage.onload = () => {
            heatmapCard.hidden = false;
            heatmapMessage.textContent =
                "Grad-CAM highlights image regions influencing the model. It does not prove disease is present.";
        };

        gradcamImage.src = heatmap;
    } else {
        heatmapCard.hidden = true;
        heatmapMessage.textContent =
            "No Grad-CAM image was returned by the backend.";
    }

    const probabilityList = document.getElementById("probability-list");
    probabilityList.replaceChildren();

    if (probabilities.length >= 5) {
        for (let i = 0; i < 5; i++) {
            const raw = Number(probabilities[i]);
            const value = Number.isFinite(raw)
                ? Math.max(0, Math.min(1, raw))
                : 0;

            const row = document.createElement("div");
            row.className = "probability-row";

            const label = document.createElement("span");
            label.className = "probability-label";
            label.textContent = `Grade ${i}`;

            const track = document.createElement("div");
            track.className = "probability-track";

            const fill = document.createElement("div");
            fill.className = "probability-fill";
            fill.style.width = `${value * 100}%`;
            track.appendChild(fill);

            const percentage = document.createElement("span");
            percentage.className = "probability-value";
            percentage.textContent = formatProbability(value);

            row.append(label, track, percentage);
            probabilityList.appendChild(row);
        }
    } else {
        probabilityList.textContent =
            "Class probabilities were not included in the API response.";
    }

    const qualityWarning = document.getElementById("quality-warning");
    const qualityMessage = document.getElementById("quality-message");
    const warnings = result.quality?.warnings;

    if (Array.isArray(warnings) && warnings.length > 0) {
        qualityWarning.hidden = false;
        qualityMessage.textContent = warnings.join(" ");
    } else {
        qualityWarning.hidden = true;
        qualityMessage.textContent = "";
    }
}

renderResults();
