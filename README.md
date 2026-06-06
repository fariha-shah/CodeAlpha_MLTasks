# CodeAlpha Machine Learning Internship

This repository contains all the machine learning projects completed during the CodeAlpha Machine Learning Internship. The internship involved building real-world machine learning models using Python and various ML libraries.

---

## Internship Overview

**Organization:** CodeAlpha  
**Domain:** Machine Learning  
**Tools Used:** Python, Scikit-learn, Librosa, Numpy, Pandas, Matplotlib, Seaborn

---

## Projects Completed

### Task 1 — Credit Scoring Model

The goal of this project was to predict whether a bank customer is a good or bad credit risk based on their financial information. The model was trained on the German Credit Dataset which contains 1000 customer records. Three classification algorithms were compared and the best performing model was selected.

- Dataset: German Credit Data (1000 records)
- Models: Logistic Regression, Decision Tree, Random Forest
- Best Accuracy: 74.50%
- Best ROC-AUC: 0.763

---

### Task 2 — Emotion Recognition from Speech

The goal of this project was to recognize human emotions from speech audio files. MFCC features were extracted from each audio file and fed into a machine learning model to classify the emotion of the speaker into one of eight categories.

- Dataset: RAVDESS (1440 audio files, 24 actors)
- Features: MFCC (Mel Frequency Cepstral Coefficients)
- Models: MLP Neural Network, Random Forest
- Best Accuracy: 60.42%
- Emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

---

### Task 3 — Handwritten Digit Recognition

The goal of this project was to recognize handwritten digits from images. The model was trained on the MNIST dataset which contains 70,000 grayscale images of handwritten digits from 0 to 9. An interactive app was also built where users can test their own handwritten digit images.

- Dataset: MNIST (70,000 images)
- Models: MLP Neural Network, Random Forest
- Best Accuracy: 94.61%
- Classes: Digits 0 to 9

---

## Repository Structure

```
CodeAlpha_MachineLearning/
    Task1_CreditScoring/
        credit_scoring_model.ipynb
        german_credit_data_with_risk.csv
        credit_scoring_model.pkl
        scaler.pkl
        README.md
    Task2_EmotionRecognition/
        emotion_recognition.ipynb
        emotion_model.pkl
        scaler.pkl
        label_encoder.pkl
        README.md
    Task3_HandwrittenRecognition/
        handwritten_recognition.ipynb
        handwriting_model.pkl
        scaler.pkl
        README.md
    README.md
```

---

## Results Summary

| Task                    | Model Used         | Accuracy |
| ----------------------- | ------------------ | -------- |
| Credit Scoring          | Random Forest      | 74.50%   |
| Emotion Recognition     | MLP Neural Network | 60.42%   |
| Handwritten Recognition | MLP Neural Network | 94.61%   |

---

**Fareeha**  
CodeAlpha Machine Learning Intern
