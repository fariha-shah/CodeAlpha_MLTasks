# CodeAlpha Machine Learning Internship — Master Portfolio & Interactive Studio

<div align="center">

![CodeAlpha ML Studio](https://img.shields.io/badge/CodeAlpha-Machine%20Learning%20Internship-2563eb?style=for-the-badge&logo=python&logoColor=white)
![Netlify Ready](https://img.shields.io/badge/Netlify-Ready%20to%20Deploy-ccff00?style=for-the-badge&logo=netlify&logoColor=06080d)
![Inference Engine](https://img.shields.io/badge/Inference-Client--Side%20Neural%20ML-2563eb?style=for-the-badge&logo=javascript&logoColor=white)
![Icons](https://img.shields.io/badge/Icons-Lucide%20Icons%20(Zero%20Emojis)-ccff00?style=for-the-badge)

### Repository: [fariha-shah/CodeAlpha_MLTasks](https://github.com/fariha-shah/CodeAlpha_MLTasks)

<p align="center">
  <strong>An artisan Machine Learning Web Studio unifying 3 production-grade ML tasks into a high-performance, human-crafted dashboard styled in Midnight Black, Electric Cobalt Blue, and Acid Lime with Lucide Icons and zero server dependencies.</strong>
</p>

</div>

---

## Table of Contents

- [Overview & Highlights](#overview--highlights)
- [Bespoke Visual Design & Color Palette](#bespoke-visual-design--color-palette)
- [Lucide Icons Integration](#lucide-icons-integration)
- [Accurate Model Pipelines & Features](#accurate-model-pipelines--features)
  - [Task 1: Credit Risk Scoring (100 Decision Trees)](#task-1--credit-risk-scoring)
  - [Task 2: Speech Emotion Recognition (Real Acoustic Analysis)](#task-2--speech-emotion-recognition)
  - [Task 3: Handwritten Digit Recognition (MNIST Center-of-Mass Pipeline)](#task-3--handwritten-digit-recognition)
- [Comparative Benchmarks](#comparative-benchmarks)
- [Project Architecture & Directory Structure](#project-architecture--directory-structure)
- [Local Setup](#local-setup)
- [Pushing to GitHub](#pushing-to-github)
- [Internship Credentials](#internship-credentials)

---

## Overview & Highlights

During the **CodeAlpha Machine Learning Internship**, three distinct domains of machine learning were explored, implemented, and optimized:
1. **Tabular Financial ML**: Credit default risk classification on the German Credit dataset.
2. **Audio Acoustic ML**: Emotion recognition from speech waveforms using the RAVDESS dataset.
3. **Computer Vision & Pattern Recognition**: Handwritten digit recognition trained on the MNIST 70,000 dataset.

All three tasks run **100% natively in the web browser** via pure vectorized Float32 forward passes and serialized decision tree ensembles.

---

## Bespoke Visual Design & Color Palette

The entire studio is crafted in a signature three-color palette:
- **Midnight Black (`#06080d`)**: Deep obsidian velvet backdrop and subtle glassmorphism surfaces.
- **Electric Cobalt Blue (`#2563eb`)**: High-energy cobalt buttons, frequency visualizer spectrum, active badges, and structural borders.
- **Acid Lime (`#ccff00`)**: Ultra-vibrant, sharp neon acid lime for winning metrics, active digits, radial gauges, and oscilloscope waveforms.

---

## Lucide Icons Integration

All generic emojis have been replaced across the application with **Lucide Icons** (`lucide.min.js`), rendering clean, scalable vector graphics for all UI controls, navigation items, affective states, and analytical reports.

---

## Accurate Model Pipelines & Features

### Task 1 — Credit Risk Scoring

- **Dataset**: German Credit Data (1,000 bank customer profiles, 9 core features)
- **Model**: **100-Tree Random Forest Classifier** (Full ensemble exported without pruning)
- **Test Accuracy**: **74.50%** | **ROC-AUC**: **0.753**
- **Features**:
  * 1-Click Persona Presets: *Alice (Low Risk)*, *Bob (High Risk)*, and *Carol (Moderate)*.
  * Real-time FICO Radial Gauge (300 to 850 score) with calibrated risk tiers.
  * Monthly loan installment calculator with risk-adjusted APR.
  * AI underwriter advice and downloadable assessment reports.

---

### Task 2 — Speech Emotion Recognition

- **Dataset**: RAVDESS (1,440 audio files across 24 actors)
- **Model**: Deep Multilayer Perceptron `(40 -> 256 -> 128 -> 8)`
- **8 Emotional Classes**: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`
- **Features**:
  * 8 Verified RAVDESS 1-Click Samples: Play authentic actor audio and inspect exact verified classifications.
  * Real Acoustic Signal Processing: Live microphone recording calculates true RMS loudness, Zero-Crossing Rate (ZCR), and spectral formants.
  * 60 FPS HTML5 Canvas Dual Visualizer: Electric cobalt frequency bars + Acid lime oscilloscope waveform.

---

### Task 3 — Handwritten Digit Recognition

- **Dataset**: MNIST (70,000 grayscale images, 28x28 pixels)
- **Model**: Deep Multilayer Perceptron `(784 -> 256 -> 128 -> 10)`
- **Test Accuracy**: **94.61%** | **Inference Latency**: **< 1 ms**
- **Accurate Preprocessing Pipeline**:
  * **Center of Mass (CoM) Alignment**: Bounding box cropped, resized to fit within 20x20, and shifted by mass centroid to exactly `(13.5, 13.5)`.
  * **Zero-Variance Clamping**: Safe variance scaling avoids division-by-zero artifacts on outer border pixels.
  * 280x280 Retina touch and mouse drawing canvas with live 28x28 normalized matrix preview.
  * 0–9 authentic MNIST sample buttons and image upload support.

---

## Comparative Benchmarks

| Task | Model Evaluated | Dataset | Test Accuracy | ROC-AUC / Macro F1 | Selection |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1: Credit Scoring** | **Random Forest (100 Trees)** | German Credit | **74.50%** | **0.753** | **Selected Best** |
| Task 1: Credit Scoring | Logistic Regression | German Credit | 74.50% | 0.763 | Baseline Model |
| Task 1: Credit Scoring | Decision Tree | German Credit | 67.50% | 0.596 | Prone to Overfitting |
| **Task 2: Speech Emotion** | **MLP Neural Net (256-128)** | RAVDESS Audio | **60.42%** | **0.60 F1** | **Selected Best** |
| Task 2: Speech Emotion | Random Forest (200 Trees) | RAVDESS Audio | 60.42% | 0.59 F1 | Comparable Baseline |
| **Task 3: Handwritten Digits**| **MLP Neural Net (256-128)** | MNIST (70k) | **94.61%** | **0.95 F1** | **Selected Best** |
| Task 3: Handwritten Digits | Random Forest (100 Trees) | MNIST (70k) | 94.46% | 0.94 F1 | Competitive Alternative |

---

## Local Setup

```bash
# Navigate to the project directory
cd "D:\CodeAlpha-Tasks"

# Start local server
python -m http.server 8080
```
Open your browser and navigate to: **`http://localhost:8080`**

---

## Pushing to GitHub

```bash
cd "D:\CodeAlpha-Tasks"
git add .
git commit -m "Replace all emojis with Lucide icons across entire application"
git push origin main
```

---

## Internship Credentials

- **Organization**: [CodeAlpha](https://www.codealpha.tech/)
- **Domain**: Machine Learning & Artificial Intelligence
- **Intern Name**: **Fareeha**
- **GitHub**: [@fariha-shah](https://github.com/fariha-shah)
- **Repository**: [CodeAlpha_MLTasks](https://github.com/fariha-shah/CodeAlpha_MLTasks)
- **Internship Period**: May – June 2026
