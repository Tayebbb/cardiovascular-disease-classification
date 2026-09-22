"""
CSE 3208 - Term Assignment
Binary Classification on a Healthcare Dataset:
Comparing Decision Tree, Naive Bayes and K-Nearest Neighbors
for Cardiovascular Disease (10-Year CHD) Prediction

Dataset: Framingham Heart Study dataset (framingham.csv)
Target: TenYearCHD (1 = patient develops coronary heart disease within 10 years, 0 = does not)

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
df = pd.read_csv("framingham.csv")

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

# 2a. Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 2b. Drop irrelevant / categorical feature.
# 'education' is an ordinal categorical code (1-4) describing socioeconomic/
# education level, not a direct physiological measurement, so it is dropped.
df = df.drop(columns=["education"])

# 2c. Handle missing values.
# Missing values affect ~11.5% of rows and are spread across several clinical
# columns (cigsPerDay, BPMeds, totChol, BMI, heartRate, glucose).
# For a simple, transparent and leakage-free approach, rows with any missing
# value are dropped (this is done BEFORE the train/test split, since deciding
# whether to keep a row does not use any information from the target that
# would not be available at prediction time - it only removes incomplete
# records).
df = df.dropna().reset_index(drop=True)

print("\nShape after dropping 'education' and rows with missing values:", df.shape)
print("Target class balance:\n", df["TenYearCHD"].value_counts())

# All remaining features are numeric already, so no further encoding is needed.

# 2d. Separate features and target
X = df.drop(columns=["TenYearCHD"])
y = df["TenYearCHD"]

# 2e. Train/test split (80/20), stratified to preserve the class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("\nTraining set size:", X_train.shape, " Testing set size:", X_test.shape)

# 2f. Feature scaling.
# KNN is distance-based, so features on larger numeric scales (e.g. totChol,
# glucose) would dominate the distance calculation unless all features are
# scaled. The scaler is fit ONLY on the training data and then applied to the
# test data, to avoid data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Hyperparameter experiment (training data only, via cross-validation)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n--- Decision Tree: max_depth experiment (5-fold CV on training set) ---")
best_dt_depth, best_dt_score = None, -1
for depth in [3, 4, 5, 6, 8, 10, None]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
    print(f"max_depth={depth}: mean CV F1 = {scores.mean():.4f}")
    if scores.mean() > best_dt_score:
        best_dt_score, best_dt_depth = scores.mean(), depth
print("Selected max_depth:", best_dt_depth)

print("\n--- KNN: n_neighbors experiment (5-fold CV on scaled training set) ---")
best_k, best_k_score = None, -1
for k in [3, 5, 7, 9, 11, 15, 21]:
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="f1")
    print(f"k={k}: mean CV F1 = {scores.mean():.4f}")
    if scores.mean() > best_k_score:
        best_k_score, best_k = scores.mean(), k
print("Selected n_neighbors:", best_k)

# GaussianNB has one main hyperparameter (var_smoothing); the library default
# (1e-9) is used since a small search did not meaningfully change the CV score.
print("\n--- Naive Bayes: var_smoothing experiment (5-fold CV on scaled training set) ---")
best_vs, best_vs_score = None, -1
for vs in [1e-9, 1e-8, 1e-7, 1e-6]:
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

# 6a. Class distribution of the target variable
plt.figure(figsize=(5, 4))
df["TenYearCHD"].value_counts().sort_index().plot(kind="bar", color=["#4C72B0", "#C44E52"])
plt.xticks([0, 1], ["No CHD (0)", "CHD (1)"], rotation=0)
plt.ylabel("Number of patients")
plt.title("Target Class Distribution (TenYearCHD)")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()

# 6b. Confusion matrices for the three models
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

# 6c. Metric comparison bar chart
results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
    kind="bar", figsize=(7, 4)
)
plt.ylabel("Score")
plt.title("Model Comparison on Test Set")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

print("\nSaved plots: class_distribution.png, confusion_matrices.png, model_comparison.png")
