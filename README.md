# Wine Quality Testing using Artificial Intelligence & Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)

## 📌 Project Overview
This project explores the effectiveness of machine learning algorithms in evaluating and predicting wine quality based on physicochemical attributes (such as pH, alcohol content, residual sugar, acidity, and sulfur dioxide levels). Traditionally assessed subjectively by human experts, this data-driven approach automates and standardizes quality evaluation.

* **Author:** Rajesh Chakraborty (Roll No: 13342723056)
* **Degree:** Bachelor of Computer Application (BCA), George Group of Colleges
* **Mentors:** Mr. Partha Koley, Mr. Sutanu Sinha

---

## 🎯 Scope & Objectives
* Perform Exploratory Data Analysis (EDA) and visualization on the Red Wine Quality dataset.
* Apply multiple supervised machine learning algorithms to classify wine quality.
* Evaluate and compare performance across algorithms using accuracy scores, confusion matrices, and classification reports.
* Analyze key physicochemical properties affecting wine quality metrics.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (`sklearn`)

---

## 📊 Models Implemented & Performance Comparison

Multiple supervised learning classification models were implemented and evaluated on the dataset:

| Model | Accuracy Score | Key Highlights |
| :--- | :---: | :--- |
| **Random Forest Classifier** | **73.75%** | Highest overall performance; handles complex feature interactions well. |
| **Decision Tree Classifier** | **67.50%** | Good balance across precision and recall for majority classes. |
| **Logistic Regression** | **63.12%** | Standard linear decision boundary benchmark. |
| **Support Vector Machine (SVM)** | **64.00%** | Linear kernel evaluation. |
| **Gaussian Naïve Bayes** | **53.75%** | Baseline probabilistic classifier. |
| **K-Nearest Neighbors (KNN)** | **48.12%** | Distance-based classification ($k=5$). |

---

## 💻 Code Structure & Workflow

```python
# 1. Load Libraries & Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("winequality-red.csv")

# 2. Train / Test Split
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 3. Model Training (Example: Random Forest)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# 4. Evaluation
rf_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
