# 🏦 Loan Approval Prediction using Machine Learning

A Machine Learning project that predicts whether a loan application will be **Approved or Rejected** based on applicant financial, employment, credit, and asset-related information.

The project uses **Logistic Regression** for binary classification and includes data preprocessing, categorical encoding, feature scaling, model training, prediction, and evaluation.

---

## 📌 Project Overview

Loan approval decisions depend on several factors such as:

* Number of dependents
* Education
* Self-employment status
* Annual income
* Loan amount
* Loan term
* CIBIL score
* Residential asset value
* Commercial asset value

This project uses these features to predict the loan status as:

* **1 → Approved**
* **0 → Rejected**

---

## 🎯 Project Objective

The main objective of this project is to build a Machine Learning classification model that can:

1. Load and explore the loan dataset
2. Clean and preprocess the data
3. Convert categorical values into numerical values
4. Select important features
5. Split the dataset into training and testing data
6. Scale numerical features
7. Train a Logistic Regression model
8. Predict loan approval status
9. Calculate approval probability
10. Evaluate model performance using accuracy, confusion matrix, and classification report
11. Predict the result for a new loan applicant

---

## 🧠 Machine Learning Algorithm

### Logistic Regression

The project uses **Logistic Regression**, a supervised machine learning algorithm commonly used for binary classification.

In this project:

```text
0 → Rejected
1 → Approved
```

The model calculates the probability of loan approval and uses a threshold of **0.5** to make the final decision.

```text
Probability ≥ 0.5 → Approved
Probability < 0.5 → Rejected
```

---

## 📊 Features Used

The following 9 features are used for prediction:

| Feature                    | Description                 |
| -------------------------- | --------------------------- |
| `no_of_dependents`         | Number of dependents        |
| `education`                | Education status            |
| `self_employed`            | Self-employment status      |
| `income_annum`             | Annual income               |
| `loan_amount`              | Requested loan amount       |
| `loan_term`                | Loan repayment term         |
| `cibil_score`              | Applicant's CIBIL score     |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value`  | Value of commercial assets  |

### Target Variable

```text
loan_status
```

Encoded as:

```text
Approved → 1
Rejected → 0
```

---

## 🔄 Data Preprocessing

The following preprocessing steps are performed:

### 1. Remove extra spaces from column names

```python
df.columns = df.columns.str.strip()
```

### 2. Encode Loan Status

```text
Approved → 1
Rejected → 0
```

### 3. Encode Education

```text
Graduate → 1
Not Graduate → 0
```

### 4. Encode Self Employment

```text
Yes → 1
No → 0
```

### 5. Feature Selection

The selected features are separated into:

```text
X → Input Features
y → Target Variable
```

---

## 📂 Project Workflow

```text
Loan Dataset
     ↓
Data Loading
     ↓
Data Exploration
     ↓
Data Cleaning
     ↓
Categorical Encoding
     ↓
Feature Selection
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Logistic Regression
     ↓
Model Training
     ↓
Prediction
     ↓
Approval Probability
     ↓
Model Evaluation
     ↓
Approved / Rejected
```

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Split Ratio

* **80% → Training Data**
* **20% → Testing Data**

`random_state=42` is used to make the split reproducible.

`stratify=y` helps maintain the class distribution in both training and testing datasets.

---

## 📏 Feature Scaling

Before training the Logistic Regression model, the features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Feature scaling helps bring features with different numerical ranges to a comparable scale.

---

## 🤖 Model Training

The Logistic Regression model is created using:

```python
model = LogisticRegression(max_iter=1000)
```

The model is then trained using:

```python
model.fit(X_train_scaled, y_train)
```

---

## 🔮 Prediction

After training, the model predicts loan status for the test dataset:

```python
y_pred = model.predict(X_test_scaled)
```

The model also calculates the probability of each class:

```python
y_probability = model.predict_proba(X_test_scaled)
```

The probability of loan approval is extracted using:

```python
approved_probability = model.predict_proba(X_test_scaled)[:, 1]
```

---

## 📈 Model Evaluation

The model is evaluated using the following metrics:

### Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)
```

Accuracy represents the percentage of correctly classified loan applications.

### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

The confusion matrix shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Classification Report

```python
classification_report(
    y_test,
    y_pred,
    target_names=["Rejected", "Approved"]
)
```

The classification report provides:

* Precision
* Recall
* F1-score
* Support

---

## 👤 New Applicant Prediction

The project also demonstrates how the trained model can predict the result for a new applicant.

### Example Applicant

| Feature            |     Value |
| ------------------ | --------: |
| Dependents         |         2 |
| Education          |  Graduate |
| Self Employed      |        No |
| Annual Income      | 8,000,000 |
| Loan Amount        | 2,000,000 |
| Loan Term          |  10 years |
| CIBIL Score        |       750 |
| Residential Assets | 5,000,000 |
| Commercial Assets  | 2,000,000 |

The new applicant is scaled using the same scaler:

```python
new_applicant_scaled = scaler.transform(new_applicant)
```

Then the trained model predicts the loan status:

```python
prediction = model.predict(new_applicant_scaled)
```

The project also calculates the applicant's approval probability:

```python
probability = model.predict_proba(new_applicant_scaled)
```

Finally, the decision is made using a **0.5 threshold**:

```python
if probability[0][1] >= 0.5:
    print("Loan status: Approved")
else:
    print("Loan status: Rejected")
```

---

## 🛠️ Technologies Used

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 📊 Matplotlib
* 🤖 Scikit-learn
* 📓 Google Colab / Jupyter Notebook

---

## 📚 Python Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

---

## 📁 Project Structure

```text
Loan-Approval-Prediction/
│
├── Loan_approval.ipynb
├── loan_approval_dataset.csv
└── README.md
```

> Make sure the dataset file name and notebook path are correctly configured before running the project.

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the notebook

Open:

```text
Loan_approval.ipynb
```

using **Google Colab** or **Jupyter Notebook**.

### 3. Upload the dataset

Upload:

```text
loan_approval_dataset.csv
```

to the environment.

### 4. Run the cells

Run the notebook cells in sequence from data loading to final prediction.

---

## 📌 Key Concepts Demonstrated

This project demonstrates practical Machine Learning concepts including:

* Data Loading
* Data Exploration
* Data Cleaning
* Missing Value Checking
* Categorical Encoding
* Feature Selection
* Train-Test Split
* Feature Scaling
* Logistic Regression
* Model Coefficients
* Probability Prediction
* Classification
* Accuracy
* Confusion Matrix
* Classification Report
* New Applicant Prediction

---

## 🚀 Future Improvements

The project can be improved further by:

* Comparing Logistic Regression with other classification algorithms
* Performing hyperparameter tuning
* Adding cross-validation
* Creating more visualizations
* Using feature importance/model interpretation techniques
* Developing a simple web interface for loan prediction
* Deploying the model as a web application or API

---

## 🎓 Conclusion

This project demonstrates how Machine Learning can be used to predict loan approval based on applicant information.

Using **data preprocessing, feature scaling, and Logistic Regression**, the project creates a complete classification pipeline from raw loan data to final **Approved/Rejected** prediction.

---

## 👩‍💻 Project

**Project Name:** Loan Approval Prediction
**Algorithm:** Logistic Regression
**Task:** Binary Classification
**Environment:** Google Colab / Jupyter Notebook
**Language:** Python
