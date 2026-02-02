# University Football Injury Prediction

## Overview

This project focuses on predicting sports injuries among university football players using **machine learning and statistical analysis**. The goal is to model the probability of a player sustaining a medically verified injury in the following season based on physical, fitness, workload, and lifestyle features.

The study applies **Gaussian Naive Bayes (GNB)** in two ways:

* A **from-scratch implementation** to demonstrate understanding of probabilistic modeling
* A **scikit-learn implementation** used as a benchmark for validation

Both approaches are evaluated on raw and outlier-cleaned datasets to analyze robustness and performance.

---

## Dataset

* **Source:** Kaggle – University Football Injury Prediction Dataset
* **Samples:** 800 university football players
* **Task:** Binary classification

  * `0` → No injury
  * `1` → Injury
* **Features:**

  * 16 quantitative features (e.g., age, BMI, sprint speed, knee strength, sleep hours)
  * 2 categorical features (e.g., position, warm-up adherence)
* **Age Range:** 18–24 years
* **Class Balance:** Approximately balanced

---

## Methodology

### 1. Data Preprocessing

* Label encoding applied to categorical variables
* Outlier detection using the **Interquartile Range (IQR)** method:

  * Observations outside ([Q1 - 2·IQR,; Q3 + 2·IQR]) were removed
* Two datasets used for comparison:

  * **Raw dataset** (no outlier removal)
  * **Cleaned dataset** (24 samples removed)

### 2. Train–Test Split

* Stratified **80% training / 20% testing** split
* Outlier removal performed **before** splitting to allow controlled comparison

### 3. Feature Standardization

* Z-score normalization applied to quantitative features
* Mean and standard deviation computed **only on training data** to prevent data leakage

---

## Statistical Analysis

* **Descriptive statistics** for all features (mean, median, variance, range)
* **Distribution analysis** using histograms
* **Normality testing** with the Shapiro–Wilk test (α = 0.05)
* **Conditional KDE plots** to analyze feature behavior given injury status

These analyses were used to assess the validity of the Gaussian assumption underlying Naive Bayes.

---

## Models Implemented

### 1. Gaussian Naive Bayes (From Scratch)

* Implemented using only **NumPy and Pandas**
* Key components:

  * Class prior estimation
  * Feature-wise mean and variance per class
  * Gaussian log-probability density function
  * Log-posterior computation for numerical stability
* Zero-variance handled using a small constant (ε = 1e-9)

### 2. Gaussian Naive Bayes (scikit-learn)

* Implemented using `sklearn.naive_bayes.GaussianNB`
* Used as a reference model for correctness and performance comparison

---

## Results

### Performance on Full Dataset (800 samples)

| Metric             | From-Scratch GNB | Sklearn GNB |
| ------------------ | ---------------- | ----------- |
| Accuracy           | 0.975            | 0.975       |
| Precision (Injury) | 0.987            | 0.987       |
| Recall (Injury)    | 0.9625           | 0.9625      |
| F1-score (Injury)  | 0.9747           | 0.9747      |

### Performance After Outlier Removal

| Metric             | From-Scratch GNB | Sklearn GNB |
| ------------------ | ---------------- | ----------- |
| Accuracy           | 0.9487           | 0.9487      |
| Precision (Injury) | 0.972            | 0.972       |
| Recall (Injury)    | 0.921            | 0.921       |
| F1-score (Injury)  | 0.946            | 0.946       |

Both models produced **identical results**, confirming the correctness of the custom implementation.

---

## Key Findings

* Outlier removal improved Gaussian-like behavior in several features
* Removing outliers reduced recall due to disproportionate removal of injured cases
* Some features (e.g., knee strength) showed stronger predictive power than others
* Gaussian Naive Bayes proved effective despite its strong independence assumptions

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* SciPy

---

## Team Contributions

* **Mona Elkholy**: Data preprocessing and dataset preparation
* **Khadija Mabrouk**: Statistical analysis and visualizations
* **Rowida Mohammed**: Gaussian Naive Bayes implementation from scratch
* **Malak El-Hamshary**: Sklearn model implementation and performance comparison

---

## References

* Kaggle: University Football Injury Prediction Dataset

---

## Notes

This project was developed as part of a **Biostatistics** and emphasizes both theoretical understanding and practical implementation.
