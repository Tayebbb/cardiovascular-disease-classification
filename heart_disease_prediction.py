"""
CSE 3208 - Term Assignment
Binary Classification on a Healthcare Dataset:
Comparing Decision Tree, Naive Bayes and K-Nearest Neighbors
for Cardiovascular Disease Prediction

Dataset: Cardiovascular Disease Dataset (cardiovascular_disease_dataset.csv)
Source: Mendeley Data (Mishra et al.), a hospital-collected clinical dataset of
1,000 patients with 13 clinical/diagnostic features, publicly mirrored on GitHub.
Target: target (1 = patient has cardiovascular/heart disease, 0 = patient does not)

This script performs the same steps as the accompanying Jupyter notebook
(Healthcare_ML_Assignment.ipynb) and produces the same results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
df = pd.read_csv("cardiovascular_disease_dataset.csv")

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe())

# ---------------------------------------------------------------------------
# 2. Data preprocessing
# ---------------------------------------------------------------------------

# 2a. Drop irrelevant feature.
# 'patientid' is just a record identifier with no predictive value.
df = df.drop(columns=["patientid"])

# 2b. Check missing values (as recorded by pandas)
print("\nMissing values per column (NaN check):")
print(df.isnull().sum())

# 2c. Mark biologically-impossible values as missing.
# 'serumcholestrol' has 53 rows recorded as 0 mg/dL, which is not physiologically
# possible for a living patient - these are missing values encoded as 0, not
# genuine readings. They are converted to NaN here, but the replacement VALUE
# (the median) is computed later from the training set only, after the split,
# so no information from the test set leaks into preprocessing.
zero_chol_count = (df["serumcholestrol"] == 0).sum()
print(f"\n'serumcholestrol' rows recorded as 0 (treated as missing): {zero_chol_count}")
df["serumcholestrol"] = df["serumcholestrol"].replace(0, np.nan)

# All remaining features (age, gender, chestpain, restingBP, fastingbloodsugar,
# restingrelectro, maxheartrate, exerciseangia, oldpeak, slope, noofmajorvessels)
# are already numeric, so no further encoding is needed.

print("\nShape after dropping 'patientid':", df.shape)
print("Target class balance:\n", df["target"].value_counts())

# 2d. Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# 2e. Train/test split (80/20), stratified to preserve the class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("\nTraining set size:", X_train.shape, " Testing set size:", X_test.shape)

# 2f. Impute missing 'serumcholestrol' values using the TRAINING set's median
# only, then apply that same value to both sets, to avoid data leakage.
median_chol = X_train["serumcholestrol"].median()
X_train = X_train.copy()
X_test = X_test.copy()
X_train["serumcholestrol"] = X_train["serumcholestrol"].fillna(median_chol)
X_test["serumcholestrol"] = X_test["serumcholestrol"].fillna(median_chol)
print(f"Imputed missing serumcholestrol with training-set median = {median_chol}")

# 2g. Feature scaling.
# KNN is distance-based, so features on larger numeric scales (e.g. restingBP,
# serumcholestrol) would dominate the distance calculation unless all features
# are scaled. The scaler is fit ONLY on the training data and then applied to
# the test data, to avoid data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Hyperparameter experiment (training data only, via cross-validation)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n--- Decision Tree: max_depth experiment (5-fold CV on training set) ---")
dt_depths = [3, 4, 5, 6, 8, 10, None]
dt_cv_scores = []
best_dt_depth, best_dt_score = None, -1
for depth in dt_depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
    dt_cv_scores.append(scores.mean())
    print(f"max_depth={depth}: mean CV F1 = {scores.mean():.4f}")
    if scores.mean() > best_dt_score:
        best_dt_score, best_dt_depth = scores.mean(), depth
print("Selected max_depth:", best_dt_depth)

print("\n--- KNN: n_neighbors experiment (5-fold CV on scaled training set) ---")
knn_ks = [3, 5, 7, 9, 11, 15, 21]
knn_cv_scores = []
best_k, best_k_score = None, -1
for k in knn_ks:
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="f1")
    knn_cv_scores.append(scores.mean())
    print(f"k={k}: mean CV F1 = {scores.mean():.4f}")
    if scores.mean() > best_k_score:
        best_k_score, best_k = scores.mean(), k
print("Selected n_neighbors:", best_k)

print("\n--- Naive Bayes: var_smoothing experiment (5-fold CV on scaled training set) ---")
best_vs, best_vs_score = None, -1
for vs in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
    clf = GaussianNB(var_smoothing=vs)
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="f1")
    print(f"var_smoothing={vs}: mean CV F1 = {scores.mean():.4f}")
    if scores.mean() > best_vs_score:
        best_vs_score, best_vs = scores.mean(), vs
print("Selected var_smoothing:", best_vs)

# ---------------------------------------------------------------------------
# 4. Final model training (on the full training set, using selected hyperparameters)
# ---------------------------------------------------------------------------
dt_model = DecisionTreeClassifier(max_depth=best_dt_depth, random_state=RANDOM_STATE)
dt_model.fit(X_train, y_train)

nb_model = GaussianNB(var_smoothing=best_vs)
nb_model.fit(X_train_scaled, y_train)

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------------
# 5. Final evaluation on the untouched test set
# ---------------------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }

dt_pred = dt_model.predict(X_test)
nb_pred = nb_model.predict(X_test_scaled)
knn_pred = knn_model.predict(X_test_scaled)

results = pd.DataFrame([
    evaluate("Decision Tree", y_test, dt_pred),
    evaluate("Naive Bayes", y_test, nb_pred),
    evaluate("KNN", y_test, knn_pred),
])

print("\n=== Final Test Set Results ===")
print(results.to_string(index=False))

results.to_csv("results.csv", index=False)
print("\nSaved final results to results.csv")

# ---------------------------------------------------------------------------
# 6. Simple visualizations (optional, as allowed by the assignment)
# ---------------------------------------------------------------------------

plt.figure(figsize=(5, 4))
df["target"].value_counts().sort_index().plot(kind="bar", color=["#4C72B0", "#C44E52"])
plt.xticks([0, 1], ["No disease (0)", "Disease (1)"], rotation=0)
plt.ylabel("Number of patients")
plt.title("Target Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, pred) in zip(
    axes, [("Decision Tree", dt_pred), ("Naive Bayes", nb_pred), ("KNN", knn_pred)]
):
    cm = confusion_matrix(y_test, pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
    kind="bar", figsize=(7, 4)
)
plt.ylabel("Score")
plt.title("Model Comparison on Test Set")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# 6d. Hyperparameter search curves (shows how CV F1 changes with each setting)
depth_labels = [str(d) if d is not None else "None" for d in dt_depths]
plt.figure(figsize=(6, 4))
plt.plot(depth_labels, dt_cv_scores, marker="o")
plt.xlabel("max_depth")
plt.ylabel("Mean CV F1-score")
plt.title("Decision Tree: CV F1-score vs. max_depth")
plt.grid(True)
plt.tight_layout()
plt.savefig("dt_depth_tuning.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(knn_ks, knn_cv_scores, marker="o")
plt.xlabel("k (n_neighbors)")
plt.ylabel("Mean CV F1-score")
plt.title("KNN: CV F1-score vs. Number of Neighbors (k)")
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_k_tuning.png", dpi=150)
plt.close()

print("\nSaved plots: class_distribution.png, confusion_matrices.png, model_comparison.png, "
      "dt_depth_tuning.png, knn_k_tuning.png")
