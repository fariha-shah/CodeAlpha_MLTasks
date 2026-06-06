# Credit Scoring Model

### CodeAlpha Machine Learning Internship — Task 1

---

## Project Overview

This project builds a machine learning model that predicts whether a bank customer is a good or bad credit risk. Given basic information about a person such as age, income, savings, housing situation and loan purpose, the model decides whether that person is likely to repay their loan or not.

This is a binary classification problem where the output is either **good** (will repay) or **bad** (will not repay).

---

## Dataset

**German Credit Data**

- 1000 customers from a German bank
- 9 features per customer
- Target column: Risk (good / bad)
- Source: Kaggle — https://www.kaggle.com/datasets/uciml/german-credit
- Original labels obtained from UCI ML Repository

| Feature          | Description                |
| ---------------- | -------------------------- |
| Age              | Age of the customer        |
| Sex              | Male or Female             |
| Job              | Job skill level (0 to 3)   |
| Housing          | Own, Rent or Free          |
| Saving accounts  | Amount in savings          |
| Checking account | Amount in checking account |
| Credit amount    | Loan amount requested      |
| Duration         | Loan duration in months    |
| Purpose          | Reason for the loan        |

**Target Distribution:**

```
Good credit (will repay)     → 700 customers  (70%)
Bad credit  (will not repay) → 300 customers  (30%)
```

---

## Project Structure

```
CodeAlpha_CreditScoring/
    credit_scoring_model.ipynb     # Main notebook
    german_credit_data.csv         # Raw dataset
    german_credit_data_with_risk.csv  # Dataset with Risk column added
    credit_scoring_model.pkl       # Saved Random Forest model
    scaler.pkl                     # Saved StandardScaler
    README.md
```

---

## Approach

**Data Preprocessing**

- Missing values in Saving accounts (183 missing) and Checking account (394 missing) were filled with the value unknown
- Categorical columns were converted to numbers using Label Encoding
- Numerical columns (Age, Credit amount, Duration) were scaled using StandardScaler
- Data was split into 80% training and 20% testing

**Models Trained**

Three classification models were trained and compared:

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 74.50%   | 0.763   |
| Decision Tree       | 67.50%   | 0.596   |
| Random Forest       | 74.50%   | 0.753   |

Random Forest was selected as the final model because it had the best balance between accuracy and recall for bad credit customers.

---

## Results

### Classification Report — Random Forest

| Class            | Precision | Recall | F1-Score |
| ---------------- | --------- | ------ | -------- |
| Bad Credit       | 0.69      | 0.55   | 0.61     |
| Good Credit      | 0.76      | 0.86   | 0.81     |
| Overall Accuracy |           |        | 74.50%   |

### Feature Importance

The most influential features in predicting credit risk:

```
1. Credit amount      25.3%
2. Age                17.6%
3. Duration           15.8%
4. Checking account   13.3%
5. Purpose             9.3%
```

Credit amount is the single most important factor. Customers requesting higher loan amounts tend to carry more risk.

---

## Libraries Used

```
pandas        — data loading and manipulation
numpy         — numerical operations
matplotlib    — plotting and visualization
seaborn       — statistical charts
scikit-learn  — model training and evaluation
joblib        — saving and loading models
ipywidgets    — interactive prediction app
```

---

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/fariha-shah/CodeAlpha_Tasks/CodeAlpha_CreditScoring
cd CodeAlpha_CreditScoring
```

**2. Install required libraries**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets
```

**3. Open the notebook**

```bash
jupyter notebook credit_scoring_model.ipynb
```

**4. Run all cells in order**

Use Kernel -> Restart & Run All to run everything from top to bottom.

---

## Interactive App

The notebook includes an interactive credit scoring app built with ipywidgets. You can enter customer details using sliders and dropdowns and the model will instantly predict whether the customer is a good or bad credit risk along with a confidence percentage.

---

## Key Observations

- Younger customers (age 20 to 30) tend to have higher bad credit rates compared to older customers
- Higher loan amounts are associated with higher credit risk
- Customers with unknown or little savings are more likely to be bad credit risks
- The Decision Tree model overfits the training data which is why its accuracy is lower than the other two models

---

**Fareeha**
CodeAlpha Machine Learning Intern
