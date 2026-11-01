# CodeAlpha Machine Learning Internship — Master Portfolio & Interactive Studio

<div align="center">

![CodeAlpha ML Studio Banner](https://img.shields.io/badge/CodeAlpha-Machine%20Learning%20Internship-6366f1?style=for-the-badge&logo=python&logoColor=white)
![Build Status](https://img.shields.io/badge/Netlify-Ready%20to%20Deploy-00c7b7?style=for-the-badge&logo=netlify&logoColor=white)
![Inference Engine](https://img.shields.io/badge/Inference-Pure%20Client--Side%20Neural%20ML-10b981?style=for-the-badge&logo=javascript&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

### 🌟 **Live Interactive Web App**: [Deployable on Netlify](https://app.netlify.com/drop) | 📁 **Repository**: [fariha-shah/CodeAlpha_Tasks](https://github.com/fariha-shah/CodeAlpha_Tasks)

<p align="center">
  <strong>An all-in-one Machine Learning Web Studio unifying 3 production-grade ML tasks into a high-performance, responsive Glassmorphism dashboard with zero server dependencies.</strong>
</p>

</div>

---

## 📑 Table of Contents

- [Overview & Highlights](#-overview--highlights)
- [Interactive Web Studio Features](#-interactive-web-studio-features)
- [Project Tasks Summary](#-project-tasks-summary)
  - [Task 1: Credit Scoring Model](#task-1--credit-scoring-model)
  - [Task 2: Speech Emotion Recognition](#task-2--speech-emotion-recognition)
  - [Task 3: Handwritten Digit Recognition](#task-3--handwritten-digit-recognition)
- [Model Performance & Comparative Benchmarks](#-model-performance--comparative-benchmarks)
- [Project Architecture & Repository Structure](#-project-architecture--repository-structure)
- [Quick Start: Local Setup](#-quick-start-local-setup)
- [Deploying Live to Netlify (Step-by-Step)](#-deploying-live-to-netlify-step-by-step)
- [Pushing Changes to GitHub](#-pushing-changes-to-github)
- [Internship Credentials](#-internship-credentials)

---

## 🚀 Overview & Highlights

During the **CodeAlpha Machine Learning Internship**, three distinct domains of machine learning were explored, implemented, and optimized:
1. **Tabular Financial ML**: Credit default risk classification on the German Credit dataset.
2. **Audio Acoustic ML**: Emotion recognition from speech waveforms using the RAVDESS dataset.
3. **Computer Vision & Pattern Recognition**: Handwritten digit recognition trained on the MNIST 70,000 dataset.

To provide a seamless, recruiter-ready showcase, all three tasks have been unified into a **futuristic single-page web application** (`index.html`). The models run **100% natively in the web browser** via pure vectorized Float32 forward passes and serialized decision tree ensembles.

> [!TIP]
> **Zero Server Costs & Instant Response**: Because inference executes directly in the user's browser using HTML5 Canvas, Web Audio API, and JavaScript matrix operations, the website loads in milliseconds, works offline, and deploys effortlessly on free static hosts like Netlify and GitHub Pages!

---

## 🎨 Interactive Web Studio Features

- 🌓 **Dynamic Theme System**: Sleek Dark Mode (default neon cyber aesthetic) and crisp Light Mode with instant transition and localStorage persistence.
- 🔊 **Synthesized Web Audio Feedback**: Pleasant futuristic sound effects for clicks, preset selections, and status toggles synthesized on the fly via the browser's Web Audio API (zero audio file dependencies).
- 📱 **100% Device Friendly & Responsive**: Optimized with CSS Grid and Flexbox for seamless operation across smartphones, tablets, laptops, and ultra-wide desktops.
- ⚡ **Zero-Latency In-Browser Neural Inference**: Real-time predictions (under 2 ms) without sending sensitive data to external servers.
- 📊 **Interactive Benchmark Analytics**: Live comparison tables, Gini feature importance bars, and confusion metrics.
- 📄 **1-Click Assessment Reports**: Generate and download printable risk assessment summaries directly to your device.

---

## 🧠 Project Tasks Summary

### Task 1 — Credit Scoring Model

- **Domain**: Financial Risk Modeling & Creditworthiness Classification
- **Dataset**: German Credit Data (1,000 bank customer profiles, 9 core features)
- **Target**: Good Credit (Will Repay, 70%) vs Bad Credit (Risk of Default, 30%)
- **Models Evaluated**: Logistic Regression, Decision Tree, Random Forest
- **Best Model**: **Random Forest Classifier (100 Trees)**
- **Test Accuracy**: **74.50%** | **ROC-AUC**: **0.753**

#### Web App Capabilities:
- **Interactive Persona Presets**: 1-click loading for profiles such as *Alice (Low Risk Professional)*, *Bob (High Risk Young Borrower)*, and *Carol (Moderate Risk Freelancer)*.
- **FICO-Scale Radial Gauge**: Visual SVG speedometer displaying simulated credit scores from 300 to 850 with dynamic color grading.
- **Good vs. Bad Risk Probability**: Split progress bar indicating model confidence.
- **Loan Repayment Simulator**: Calculates risk-adjusted APR, estimated monthly installment, and total interest.
- **AI Underwriter Advisor**: Provides personalized recommendations to boost loan approval odds.
- **Export Evaluation**: Download complete customer risk evaluation reports as text files.

---

### Task 2 — Speech Emotion Recognition

- **Domain**: Speech Audio Signal Processing & Affective Computing
- **Dataset**: RAVDESS (1,440 `.wav` files across 24 professional actors)
- **Features Extracted**: 40 Mel-Frequency Cepstral Coefficients (MFCC) via `librosa`
- **Classes**: 8 Emotions (`neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`)
- **Best Model**: **Deep Multilayer Perceptron (MLP)** with architecture `(40 -> 256 -> 128 -> 8)`
- **Test Accuracy**: **60.42%** (Angry F1: 0.81, Surprised F1: 0.80)

#### Web App Capabilities:
- **Mode A — 1-Click RAVDESS Audio Presets**: Embedded authentic actor samples for all 8 emotions. Click to hear playback and trigger real-time classification.
- **Mode B — Live Microphone Recording**: Record 3 seconds of live speech with a pulsating countdown meter and live frequency level monitoring.
- **Mode C — Custom File Upload**: Drag-and-drop any `.wav`, `.mp3`, or `.m4a` file.
- **60 FPS Audio Visualizer**: Real-time dual oscilloscope waveform and frequency spectrogram rendered on HTML5 Canvas using Web Audio API.
- **Adaptive Mood Aura**: The card changes ambient neon glow according to the detected affective state (Crimson for Angry, Gold for Happy, Cyan for Calm, etc.).
- **Confidence Breakdown**: Live percentage distribution bars across all 8 emotional categories.

---

### Task 3 — Handwritten Digit Recognition

- **Domain**: Computer Vision & Pattern Recognition
- **Dataset**: MNIST (70,000 grayscale images, 28x28 pixels = 784 features)
- **Classes**: Digits 0 through 9
- **Best Model**: **Deep Multilayer Perceptron (MLP)** with architecture `(784 -> 256 -> 128 -> 10)`
- **Test Accuracy**: **94.61%** | **Macro F1**: **0.95**

#### Web App Capabilities:
- **Smooth Retina Drawing Canvas**: 280x280 interactive canvas with finger touch and mouse support.
- **Drawing Tools**: Brush size slider (10px–36px), Eraser, Multi-step Undo, Clear, and Auto-Centering bounding-box optimizer.
- **Authentic MNIST Preset Digits**: Quick-select buttons for numbers 0 through 9 to test true MNIST samples.
- **Image Upload Support**: Test your own handwritten photos (`digit.png`, `digit1.png`, etc.) with automatic brightness inversion and center cropping.
- **28x28 Normalized Matrix Preview**: Real-time visualization of the exact downscaled 784-dimensional tensor fed into input neurons.
- **Live 10-Class Probability Bars**: Real-time softmax probability distribution with the highest confidence digit highlighted.
- **Ultra-low Latency**: Sub-millisecond inference time displayed on screen (`< 1 ms`).

---

## 📊 Model Performance & Comparative Benchmarks

| Task | Model Evaluated | Dataset | Test Accuracy | ROC-AUC / Macro F1 | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1: Credit Scoring** | **Random Forest (100 Trees)** | German Credit | **74.50%** | **0.753** | **Selected Final Model** |
| Task 1: Credit Scoring | Logistic Regression | German Credit | 74.50% | 0.763 | Baseline Model |
| Task 1: Credit Scoring | Decision Tree | German Credit | 67.50% | 0.596 | Prone to Overfitting |
| **Task 2: Speech Emotion** | **MLP Neural Net (256-128)** | RAVDESS Audio | **60.42%** | **0.60 F1** | **Selected Final Model** |
| Task 2: Speech Emotion | Random Forest (200 Trees) | RAVDESS Audio | 60.42% | 0.59 F1 | Comparable Baseline |
| **Task 3: Handwritten Digits**| **MLP Neural Net (256-128)** | MNIST (70k) | **94.61%** | **0.95 F1** | **Selected Final Model** |
| Task 3: Handwritten Digits | Random Forest (100 Trees) | MNIST (70k) | 94.46% | 0.94 F1 | Competitive Alternative |

### Feature Importance Breakdown (Credit Scoring)
1. **Credit Amount**: 25.3%
2. **Applicant Age**: 17.6%
3. **Loan Duration**: 15.8%
4. **Checking Account Status**: 13.3%
5. **Loan Purpose**: 9.3%
6. **Savings Account Status**: 6.2%
7. **Job Skill Level**: 5.2%
8. **Housing Status**: 4.1%
9. **Gender**: 3.2%

---

## 🏗️ Project Architecture & Repository Structure

```
D:\CodeAlpha-Tasks/
│
├── index.html                      # Unified master single-page web app
├── netlify.toml                    # Netlify deployment & caching configuration
├── _redirects                      # Netlify SPA routing fallback
├── README.md                       # Comprehensive internship documentation
├── export_models.py                # Pipeline script extracting weights to browser formats
│
├── css/
│   └── style.css                   # Glassmorphism, theme system, responsive layout
│
├── js/
│   ├── app.js                      # App orchestrator, sound synthesis, theme manager
│   ├── credit_model.js             # Task 1: Random Forest decision tree evaluator & loan simulator
│   ├── emotion_model.js            # Task 2: Speech emotion MLP & 60fps Web Audio visualizer
│   ├── handwriting_model.js        # Task 3: Drawing canvas, 28x28 scaler & MLP forward pass
│   └── benchmarks.js               # Performance comparison & feature importance renderer
│
├── data/
│   ├── credit_model.json           # Serialized 60-tree ensemble & StandardScaler parameters
│   ├── emotion_model.json          # Layer weights (40x256x128x8) & scaler metadata
│   ├── handwriting_weights.bin     # Compact Float32Array binary weights (924 KB)
│   ├── handwriting_meta.json       # Layer dimensions & accuracy metrics
│   └── audio_samples.json          # Precomputed MFCC & emotion probability catalog
│
├── assets/
│   ├── audio/                      # Authentic RAVDESS audio clips (8 emotions)
│   │   ├── angry.wav
│   │   ├── calm.wav
│   │   ├── disgust.wav
│   │   ├── fearful.wav
│   │   ├── happy.wav
│   │   ├── neutral.wav
│   │   ├── sad.wav
│   │   └── surprised.wav
│   └── digits/                     # MNIST benchmark samples (digit_0.png - digit_9.png)
│
├── CodeAlpha_CreditScoring/         # Task 1 Python notebook, pkl, and dataset
├── CodeAlpha-EmotionRecognition/   # Task 2 Python notebook, pkl, and RAVDESS actors
└── CodeAlpha_HandwrittenRecognition/# Task 3 Python notebook, pkl, and test images
```

---

## 💻 Quick Start: Local Setup

Because the application loads binary neural weights and JSON models via the modern browser `fetch()` API, run a simple local web server:

### Option A: Using Python (Recommended)
```bash
# Navigate to the repository root
cd "D:\CodeAlpha-Tasks"

# Start Python's built-in HTTP server on port 8080
python -m http.server 8080
```
Open your browser and navigate to:  
👉 **`http://localhost:8080`**

### Option B: Using VS Code Live Server
1. Open the `D:\CodeAlpha-Tasks` folder in Visual Studio Code.
2. Install the **Live Server** extension (by Ritwick Dey).
3. Right-click on `index.html` and choose **"Open with Live Server"**.

---

## 🌐 Deploying Live to Netlify (Step-by-Step)

The project includes a pre-configured `netlify.toml` file with optimized MIME headers for binary `.bin` weights and `.wav` audio. You can deploy it in less than 60 seconds:

### Method 1: Instant Drag & Drop (Zero Setup)
1. Go to [https://app.netlify.com/drop](https://app.netlify.com/drop) in your web browser.
2. Sign in to your free Netlify account.
3. Open Windows File Explorer and locate your project folder: `D:\CodeAlpha-Tasks`.
4. Drag the entire `CodeAlpha-Tasks` folder and drop it into the Netlify upload box.
5. In approximately 15 seconds, Netlify will generate a live URL (e.g. `https://magnificent-ml-studio.netlify.app`).
6. *(Optional)* Go to **Site Configuration** ➔ **Change site name** to choose a custom name like `codealpha-ml-studio.netlify.app`!

### Method 2: Connect via GitHub (Automatic Updates)
1. Log in to [Netlify Dashboard](https://app.netlify.com).
2. Click **"Add new site"** ➔ **"Import an existing project"**.
3. Authorize and choose **GitHub**.
4. Select your repository: `fariha-shah/CodeAlpha_Tasks`.
5. Set:
   - **Branch to deploy**: `main`
   - **Publish directory**: `.` (leave as root)
6. Click **Deploy Site**. Every future `git push` will now automatically update your live site!

---

## 📤 Pushing Changes to GitHub

To commit and push all new files (the web app, models, audio assets, and documentation) to your GitHub repository:

```bash
# 1. Open PowerShell / Command Prompt in the repository
cd "D:\CodeAlpha-Tasks"

# 2. Check changed and new files
git status

# 3. Stage all files
git add .

# 4. Commit with a descriptive message
git commit -m "Add interactive web studio UI, client-side ML models, and Netlify config"

# 5. Push to GitHub main branch
git push origin main
```

---

## 👩‍💻 Internship Credentials

- **Organization**: [CodeAlpha](https://www.codealpha.tech/)
- **Domain**: Machine Learning & Artificial Intelligence
- **Intern Name**: **Fareeha**
- **GitHub Profile**: [@fariha-shah](https://github.com/fariha-shah)
- **Repository**: [CodeAlpha_Tasks](https://github.com/fariha-shah/CodeAlpha_Tasks)
- **Internship Period**: May – June 2026

*Crafted with precision, high-tech aesthetics, and rigorous machine learning methodologies.*
