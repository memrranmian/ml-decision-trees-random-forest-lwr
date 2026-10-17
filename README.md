# Machine Learning (DS-550) - Assignment 2

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> **Course:** DS-550 Machine Learning  
> **Submitted By:** M Imran Mian (MSDSF25A007)  
> **Submitted To:** Dr. Syed Muhammad Ali  
> **Deadline:** 11:59 PM, Oct 22, 2026  
> **Implementation Methodology:** 100% From Scratch (Python, NumPy, Pandas — Zero Black-Box ML Libraries)

---

## 📌 Project Overview

This repository contains the complete implementation, theoretical formulations, and empirical evaluations for **Assignment 2** of the **DS-550 Machine Learning** course. All four algorithms are developed completely from first principles without relying on third-party machine learning libraries (such as `scikit-learn`):

1. **Q1. ID-3 Decision Tree Classifier (Categorical):**
   - Greedy recursive decision tree induction using Shannon Entropy and Information Gain.
   - Robust missing value handling: Mode Imputation (`'b'` = bulbous) on Attribute 11 (`stalk-root`, 2,480 missing entries).
   - Evaluated via Stratified 5-Fold Cross-Validation.

2. **Q2. Random Forest Classifier (Ensemble):**
   - Ensemble of ID-3 decision trees using Bootstrap Aggregation (Bagging).
   - Random feature subspace selection at each node split using $m = \lceil\sqrt{D}\rceil$ features ($m = \lceil\sqrt{22}\rceil = 5$).
   - Majority voting aggregation to eliminate correlation across trees.

3. **Q3. Decision Tree Regressor (Continuous):**
   - Continuous feature partitioning optimizing Mean Squared Error (MSE) reduction / variance reduction as the splitting criterion.
   - Evaluated on residential property price estimation.

4. **Q4. Locally Weighted Linear Regression (Non-Parametric LWR):**
   - Non-parametric instance-based regression model fitting localized linear hyperplanes.
   - Gaussian proximity kernel weighting ($\tau = 0.5$) with Ridge regularization to ensure numerical stability.
   - Empirical comparative performance analysis against the Decision Tree Regressor.

---

## 📊 Empirical Results & Verification Benchmarks

### 1. Classification Benchmarks (Mushroom Dataset — 5-Fold Cross-Validation)

| Fold | ID-3 Decision Tree (Q1) | Random Forest ($m = \lceil\sqrt{22}\rceil = 5$, Q2) |
| :---: | :---: | :---: |
| **Fold 1** | 1.0000 (100.00%) | 1.0000 (100.00%) |
| **Fold 2** | 1.0000 (100.00%) | 1.0000 (100.00%) |
| **Fold 3** | 1.0000 (100.00%) | 1.0000 (100.00%) |
| **Fold 4** | 1.0000 (100.00%) | 1.0000 (100.00%) |
| **Fold 5** | 1.0000 (100.00%) | 1.0000 (100.00%) |
| **Mean Accuracy (95% CI)** | **100.00% ($\pm$ 0.00%)** | **100.00% ($\pm$ 0.00%)** |
| **Mean Precision** | **1.0000** | **1.0000** |
| **Mean Recall** | **1.0000** | **1.0000** |
| **Mean F1-Score** | **1.0000** | **1.0000** |

*Finding:* Both models achieve perfect separability on the test folds, consistent with UCI literature (Schlimmer, 1987) proving that conjunctive logical rules over odor and spore-print color completely partition poisonous and edible specimens.

---

### 2. House Price Regression Benchmarks (13,603 Train / 6,700 Test Properties)

| Evaluation Metric | Decision Tree Regressor (Q3) | Locally Weighted Regressor (Q4) | Performance Advantage |
| :--- | :---: | :---: | :---: |
| **Mean Squared Error (MSE)** | $26,780,279,695.14$ | **$26,435,839,749.86$** | **LWR (-1.29% MSE reduction)** |
| **Root Mean Squared Error (RMSE)** | USD 163,646.81 | **USD 162,591.02** | **LWR (Lower by USD 1,055.79)** |
| **Mean Absolute Error (MAE)** | USD 130,124.60 | **USD 129,841.10** | **LWR (Lower by USD 283.50)** |
| **Coefficient of Determination ($R^2$)**| 0.3812 | **0.3892** | **LWR (+0.80% absolute gain)** |
| **Inference Time (6,700 Test Cases)** | **0.35 seconds** | 9.40 seconds | **Decision Tree (26.8x faster)** |

---

## 🧠 Key Takeaways & Mathematical Insights

1. **Continuous Manifold Smoothing vs. Rigid Partitions:**  
   Locally Weighted Regression achieves higher explanatory power ($R^2 = 0.3892$ vs. $0.3812$) because residential valuation responds continuously to square footage ($m^2$). The Gaussian proximity kernel smooths local estimates, avoiding the rigid staircase discretization artifacts of orthogonal decision tree cuts.
2. **Computational Complexity Trade-off:**  
   The Decision Tree Regressor compiles the dataset into static nodes during training, enabling ultra-fast inference ($0.35$s for 6,700 points). In contrast, LWR is non-parametric (instance-based), requiring a distance calculation and matrix inversion per query ($9.40$s).

---

## 📂 Repository Contents

```
ml_proj2_1/
├── Assignment 2-Given Data/
│   ├── Assignment 2.pdf                                 # Original assignment problem statement
│   └── Data/
│       ├── Q1-2/
│       │   ├── agaricus-lepiota.data                    # Mushroom classification dataset
│       │   └── Mushroom Database Description.pdf        # UCI dataset documentation
│       └── Q3-4/
│           ├── train.csv                                # Housing training dataset (13,603 rows)
│           └── test.csv                                 # Housing testing dataset (6,700 rows)
├── Assignment2_Notebook_Solution_MSDSF25A007.ipynb      # Fully self-contained interactive solution notebook
├── README.pdf                                           # One-page executive summary report for submission
├── README.md                                            # Project summary and documentation (this file)
├── requirements.txt                                     # Dependency requirements
└── .gitignore                                           # Clean environment exclusion manifest
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Solution Notebook
Launch the Jupyter Notebook interface to view step-by-step code blocks, mathematical equations, markdown explanations, and visualizations:
```bash
jupyter notebook Assignment2_Notebook_Solution_MSDSF25A007.ipynb
```

---

## 📜 Verification & Submission Checklist
- [x] DS-550 Machine Learning - Assignment 2 Summary Report
- [x] Interactive step-by-step Jupyter Notebook delivered as `Assignment2_Notebook_Solution_MSDSF25A007.ipynb`
