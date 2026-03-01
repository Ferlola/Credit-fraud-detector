https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

#### 💳 Credit Card Fraud Detection – Imbalanced Learning Pipeline
#### 📌 Overview

This project implements a **robust fraud detection pipeline** designed for **extreme class imbalance (0.17% fraud rate)** using gradient boosting models and threshold-aware cross-validation.

📊 Dataset Summary

- **284,807** transactions
- **30 features**
- **284,315 non-fraud**
- **492 fraud**
- **Fraud rate: 0.17%**

The goal is not only to optimize ranking metrics (ROC-AUC / PR-AUC), but to **explicitly optimize classification thresholds** under business constraints such as:

- 🎯 Maximize **Recall at Precision ≥ 90%**
- 🎯 Maximize **Precision at Recall ≥ target**
- 🎯 Maximize **F1-score**

---

#### 🧠 Methodology
#### 1️⃣ Train/Test Strategy

- 70% training / 30% testing
- Stratified split to preserve class distribution
- Fixed <span style="background-color: lightgrey;">random_state=42</span> for reproducibility

---


#### 2️⃣ ⚖️ Extreme Class Imbalance Handling

Given the severe imbalance (0.17%), the pipeline includes:

- <span style="background-color: lightgrey;">scale_pos_weight = (# negatives / # positives)</span>
- 5-fold <span style="background-color: lightgrey;">StratifiedKFold</span>
- PR-AUC as primary optimization metric
- In-fold threshold optimization
- Zero leakage evaluation design

---

#### 3️⃣ 🔍 Cross-Validated Threshold Optimization

A custom evaluation function performs:

- 📂 Stratified 5-fold CV
- 📈 Probability-based threshold tuning **only on training fold**
- 🧪 Evaluation on unseen fold
- 📊 Aggregated metrics (mean + threshold std)

Supported optimization modes:

- <span style="background-color: lightgrey;">f1</span>
- <span style="background-color: lightgrey;">recall_at_precision</span>
- <span style="background-color: lightgrey;">precision_at_recall</span>

This design mimics production fraud systems where decision thresholds are business-driven.

---

#### 🚀 Models

Hyperparameter tuning is performed using **Optuna (TPE sampler, 8 trials per model).***

#### 🔵 XGBoost

- Objective: binary:<span style="background-color: lightgrey;">logistic</span>
- Eval metric: <span style="background-color: lightgrey;">aucpr</span>
- Early stopping: 50 rounds
- Tuned: depth, learning rate, regularization, subsampling, gamma, etc.

#### 🟢 LightGBM

- Objective: <span style="background-color: lightgrey;">binary</span>
- Metric: <span style="background-color: lightgrey;">average_precision</span>
- High <span style="background-color: lightgrey;">n_estimators</span> + low learning rate
- Regularization-focused configuration

#### 🎯 Optimization Target

All models are optimized for:

>   **Maximize Recall subject to Precision ≥ 0.90**

This reflects a high-precision fraud detection business requirement.

---

#### 📊 Evaluation Metrics

Each model is evaluated via cross-validation on:

- 📈 ROC AUC
- 📉 PR AUC
- 🎯 Precision
- 🔎 Recall
- ⚖️ F1-score
- 🔧 Optimal Threshold
- 📦 Positives per fold

⚠️ PR-AUC is prioritized due to extreme imbalance, where ROC-AUC can be overly optimistic.

---

#### 🏁 Final Model Training

After hyperparameter tuning:

- ✅ Best model retrained on full training data
- 📤 Predictions generated on hold-out test set
- 📋 Results stored for inspection

---

#### 🛠 Technical Highlights

- ✔️ Proper handling of extreme imbalance
- ✔️ Threshold optimization without leakage
- ✔️ Business-constrained metric optimization
- ✔️ Cross-validated model comparison
- ✔️ Optuna-based hyperparameter tuning
- ✔️ PR-focused evaluation framework