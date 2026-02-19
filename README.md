#### 🚀 Imbalanced Classification with XGBoost & LightGBM

This project tackles an **extremely imbalanced binary classification problem (0.17% positive class)** using gradient boosting models and automated hyperparameter optimization.

#### 🔧 Approach

- Hyperparameter tuning with Optuna
- Models:
    - **XGBoost (XGBClassifier)
    - **LightGBM (LGBMClassifier)

- Class imbalance handled using:
```Python
scale_pos_weight = n_negative / n_positive
```
#### 📊 Evaluation

Given the severe imbalance, the main optimization metric was:

- **PR-AUC (Precision-Recall AUC)**

Additional metrics:
- ROC-AUC
- Recall
- F1-Score

#### 🏆 Results

Despite the **0.17% minority class**, the combination of:

- Proper class weighting
- Bayesian hyperparameter optimization
- Gradient boosting models

achieved strong minority class detection while maintaining controlled precision.

💡 This approach is suitable for fraud detection, anomaly detection, and other rare-event prediction problems.
