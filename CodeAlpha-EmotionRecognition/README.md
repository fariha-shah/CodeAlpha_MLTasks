# Emotion Recognition from Speech

### CodeAlpha Machine Learning Internship — Task 2

---

## Project Overview

This project builds a machine learning model that detects human emotions from speech audio files. The model takes a `.wav` audio file as input and predicts the emotion of the speaker — whether they sound angry, happy, sad, calm, neutral, fearful, disgusted or surprised.

---

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)

- 24 professional actors (12 male, 12 female)
- 1440 audio files in `.wav` format
- 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Source: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

---

## Project Structure

```
CodeAlpha_EmotionRecognition/
    emotion_recognition.ipynb    # Main notebook
    emotion_model.pkl            # Trained MLP model
    scaler.pkl                   # StandardScaler
    label_encoder.pkl            # Label encoder
    Actor_01/                    # RAVDESS audio files
    Actor_02/
    ...
    Actor_24/
    README.md
```

---

## Approach

**Feature Extraction**

Raw audio files cannot be fed directly into a machine learning model. We extract MFCC (Mel Frequency Cepstral Coefficients) features from each audio file using the `librosa` library. Each audio file is converted into 40 MFCC values by taking the mean across time frames. This gives us a feature vector of 40 numbers per audio file.

**Models Used**

Two models were trained and compared:

| Model              | Accuracy |
| ------------------ | -------- |
| Random Forest      | 60.42%   |
| MLP Neural Network | 60.42%   |

MLP Neural Network was selected as the final model because neural networks are better suited for audio-based tasks.

**Best Performing Emotion:** Angry (F1-Score: 0.81)

**Most Challenging Emotion:** Neutral (F1-Score: 0.43) — neutral speech is often confused with calm because both have similar low-energy patterns.

---

## Libraries Used

```
librosa       — audio loading and MFCC feature extraction
scikit-learn  — MLP model, Random Forest, preprocessing, evaluation
numpy         — numerical operations
pandas        — data handling
matplotlib    — plotting
seaborn       — visualization
joblib        — saving and loading models
ipywidgets    — interactive prediction app
```

---

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/fariha-shah/CodeAlpha_Tasks/CodeAlpha_EmotionRecognition
cd CodeAlpha_EmotionRecognition
```

**2. Install required libraries**

```bash
pip install librosa scikit-learn numpy pandas matplotlib seaborn soundfile ipywidgets
```

**3. Download the RAVDESS dataset**

Download from Kaggle and place all Actor folders in the project directory.

**4. Open the notebook**

```bash
jupyter notebook emotion_recognition.ipynb
```

**5. Run all cells in order**

Use Kernel -> Restart & Run All to run everything from top to bottom.

---

## Results

### Classification Report

| Emotion     | Precision | Recall   | F1-Score |
| ----------- | --------- | -------- | -------- |
| Angry       | 0.79      | 0.82     | 0.81     |
| Calm        | 0.65      | 0.63     | 0.64     |
| Disgust     | 0.55      | 0.61     | 0.57     |
| Fearful     | 0.65      | 0.67     | 0.66     |
| Happy       | 0.57      | 0.44     | 0.49     |
| Neutral     | 0.39      | 0.47     | 0.43     |
| Sad         | 0.57      | 0.63     | 0.60     |
| Surprised   | 0.57      | 0.51     | 0.54     |
| **Overall** | **0.61**  | **0.60** | **0.60** |

### Key Observations

- The model performs best on angry speech because angry audio has very distinct loud and sharp features that MFCCs capture well.
- Neutral speech is the hardest to classify because it sounds very similar to calm speech.
- The overall accuracy of 60% is reasonable given that the task involves distinguishing between 8 different emotions, where random guessing would only achieve 12.5%.

---

## Interactive App

The notebook includes an interactive widget where you can enter any `.wav` file path and the model will predict the emotion along with confidence scores for all 8 emotions.

---

**Fareeha**
CodeAlpha Machine Learning Intern
