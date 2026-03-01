https://archive.ics.uci.edu/dataset/27/credit+approval

#### Credit Approval – Binary Classification
#### 📌 Overview

This project implements a **credit approval classification pipeline** using supervised machine learning.
The goal is to predict whether a credit application should be **approved or rejected** based on structured tabular features.

Two models are developed, tuned, and compared:

- **Logistic Regression** (baseline linear model)
- **XGBoost Classifier** (gradient boosting model)

Model selection is performed using **Stratified Cross-Validation, Optuna hyperparameter optimization**, and evaluation via **ROC-AUC** with **bootstrap confidence intervals.**

---
#### 📊 Dataset

- **Rows**: 690
- **Features:** 15 predictors + 1 target
- **Feature types:**
    - 12 categorical <span style="background-color: lightgrey;">(str)</span>
    - 2 float
    - 2 integer
- **Target:** Binary classification <span style="background-color: lightgrey;">(Target)</span>

All categorical variables are encoded using <span style="background-color: lightgrey;">LabelEncoder.</span>
``` Python
for c in df.columns:
    if df[c].dtype == "str":
        lbl = LabelEncoder()
        lbl.fit(list(df[c].values))
        df[c] = lbl.transform(list(df[c].values))
```

---

#### 🔀 Train-Test Split

- 80% training
- 20% testing
- Stratified by target
- <span style="background-color: lightgrey;">random_state=42</span> for reproducibility

---

#### ⚙️ Modeling Strategy
#### 1️⃣ Logistic Regression

Pipeline:
- <span style="background-color: lightgrey;">StandardScaler</span>
- <span style="background-color: lightgrey;">LogisticRegression</span>

Hyperparameters optimized with **Optuna:**
- <span style="background-color: lightgrey;">C</span> (log scale)
- <span style="background-color: lightgrey;">penalty</span> (<span>l1<span>, <span style="background-color: lightgrey;">l2</span>,)
- Solver selected automatically Evaluation:
- 5-fold <span style="background-color: lightgrey;">StratifiedKFold</span>
- Metric: **ROC-AUC**

---

#### 2️⃣ XGBoost

Model: <span style="background-color: lightgrey;">XGBClassifier</span>

Optimized hyperparameters:

- <span style="background-color: lightgrey;">max_depth</span>
- <span style="background-color: lightgrey;">learning_rate</span>
- <span style="background-color: lightgrey;">subsample</span>
- <span style="background-color: lightgrey;">colsample_bytree</span>
- <span style="background-color: lightgrey;">reg_lambda</span>

Configuration:

- <span style="background-color: lightgrey;">n_estimators=5000</span>
- <span style="background-color: lightgrey;">early_stopping_rounds=100</span>
- <span style="background-color: lightgrey;">tree_method="hist"</span>
- Evaluation metric: <span style="background-color: lightgrey;">auc

Best iteration is selected via median across folds.

---

#### 📈 Model Evaluation

Final evaluation is performed on the held-out test set.

**Bootstrap AUC (1000 resamples)**

| Model              | Mean CV | Std CV | Test AUC | 95% CI Low | 95% CI High |
| ------------------ | ------- | ------ | -------- | ---------- | ----------- |
| XGBClassifier      | 0.9544  | 0.0169 | 0.9191   | 0.8666     | 0.9605      |
| LogisticRegression | 0.9322  | 0.0211 | 0.9005   | 0.8461     | 0.9504      |

#### Key Observations

- XGBoost outperforms Logistic Regression in both CV and test AUC.
- Performance variance across folds is low → stable models.
- Bootstrap confidence intervals confirm strong generalization.

#### 🧪 Validation Approach

- Stratified 5-fold cross-validation
- Hyperparameter optimization with Bayesian search (TPE sampler)
- Early stopping for boosting model
- ootstrap confidence intervals for uncertainty estimation

This ensures:

- Robust performance estimation
- Reduced overfitting risk
- Reproducibility

---

#### 🏁 Final Models

- final_model_xgb
- final_model_lgb (Logistic Regression pipeline)

Predictions are generated as:
``` Python
y_pred = final_model_xgb.predict(X_test)
```

---

#### 🧠 Conclusion

- Gradient boosting (XGBoost) provides superior discriminative power for this credit approval problem.
- Logistic Regression remains competitive and interpretable.
- The combination of cross-validation, Bayesian optimization, and bootstrap inference delivers statistically robust model comparison.
