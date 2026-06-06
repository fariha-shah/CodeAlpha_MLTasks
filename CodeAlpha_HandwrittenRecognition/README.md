# Handwritten Digit Recognition

### CodeAlpha Machine Learning Internship — Task 3

---

## Project Overview

This project builds a machine learning model that recognizes handwritten digits from images. The model takes a 28x28 pixel grayscale image of a handwritten digit as input and predicts which digit it is from 0 to 9.

This is an image classification problem where the model learns visual patterns from thousands of handwritten digit images and uses those patterns to identify new unseen digits.

---

## Dataset

**MNIST (Modified National Institute of Standards and Technology)**

- 70,000 grayscale images of handwritten digits
- Each image is 28x28 pixels which gives 784 pixel values per image
- Digits range from 0 to 9 (10 classes)
- Loaded directly from sklearn — no manual download needed

**Dataset Split:**

```
Training images  →  56,000  (80%)
Testing images   →  14,000  (20%)
```

**Digit Distribution:**

```
Digit 0 : 6903 images
Digit 1 : 7877 images
Digit 2 : 6990 images
Digit 3 : 7141 images
Digit 4 : 6824 images
Digit 5 : 6313 images
Digit 6 : 6876 images
Digit 7 : 7293 images
Digit 8 : 6825 images
Digit 9 : 6958 images
```

The dataset is well balanced across all digits so no special balancing technique was needed.

---

## Project Structure

```
CodeAlpha_HandwrittenRecognition/
    handwritten_recognition.ipynb    # Main notebook
    handwriting_model.pkl            # Saved MLP model
    scaler.pkl                       # Saved StandardScaler
    README.md
```

---

## Approach

**Data Preprocessing**

- Pixel values were normalized from 0-255 to 0-1 by dividing by 255
- StandardScaler was applied to bring all values to the same scale
- Labels were converted from strings to integers

**Models Trained**

Two models were trained and compared on 10,000 training samples:

| Model              | Accuracy |
| ------------------ | -------- |
| MLP Neural Network | 94.61%   |
| Random Forest      | 94.46%   |

MLP Neural Network was selected as the final model because it is better suited for image data and achieved slightly higher accuracy.

**Model Architecture**

```
Input Layer     : 784 neurons  (one per pixel)
Hidden Layer 1  : 256 neurons  (relu activation)
Hidden Layer 2  : 128 neurons  (relu activation)
Output Layer    : 10 neurons   (one per digit)
```

---

## Results

### Classification Report — MLP Neural Network

| Digit       | Precision | Recall   | F1-Score |
| ----------- | --------- | -------- | -------- |
| 0           | 0.95      | 0.99     | 0.97     |
| 1           | 0.97      | 0.98     | 0.98     |
| 2           | 0.95      | 0.93     | 0.94     |
| 3           | 0.93      | 0.94     | 0.93     |
| 4           | 0.95      | 0.94     | 0.94     |
| 5           | 0.93      | 0.93     | 0.93     |
| 6           | 0.97      | 0.97     | 0.97     |
| 7           | 0.94      | 0.95     | 0.95     |
| 8           | 0.94      | 0.92     | 0.93     |
| 9           | 0.92      | 0.93     | 0.92     |
| **Overall** | **0.95**  | **0.95** | **0.95** |

### Key Observations

- Digit 1 was the easiest to recognize with an f1-score of 0.98 because it has a very unique shape that no other digit resembles.
- Digit 9 was the hardest to recognize with an f1-score of 0.92 because it is often confused with digit 4 and digit 7 in poor handwriting.
- The model confused digit 4 with digit 9 about 47 times which makes sense as they look similar when handwritten quickly.
- Overall accuracy of 94.61% on 14,000 test images is a strong result for a model trained without any deep learning framework.

---

## Libraries Used

```
numpy         — numerical operations and array handling
matplotlib    — visualizing digit images and charts
seaborn       — confusion matrix heatmap
scikit-learn  — model training, preprocessing and evaluation
joblib        — saving and loading the trained model
ipywidgets    — interactive digit prediction app
```

---

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/fariha-shah/CodeAlpha_Tasks/CodeAlpha_HandwrittenRecognition
cd CodeAlpha_HandwrittenRecognition
```

**2. Install required libraries**

```bash
pip install numpy matplotlib seaborn scikit-learn ipywidgets
```

**3. Open the notebook**

```bash
jupyter notebook Handwritten_Recognition.ipynb
```

**4. Run all cells in order**

Use Kernel -> Restart & Run All to run everything from top to bottom. The MNIST dataset will be downloaded automatically on first run which may take 1 to 2 minutes.

---

## Interactive App

The notebook includes an interactive digit recognizer built with ipywidgets. You can enter any image index between 0 and 13999 to select an image from the test set. The app displays the handwritten digit image along with the model prediction and a confidence chart showing the probability for each digit from 0 to 9.

- Green title means the model predicted correctly
- Red title means the model made a wrong prediction

---

**Fareeha**
CodeAlpha Machine Learning Intern
