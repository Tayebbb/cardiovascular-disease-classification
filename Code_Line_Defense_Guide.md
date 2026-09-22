# Code Line Defense Guide

Explanations for every important line actually used in `heart_disease_prediction.py` /
`Healthcare_ML_Assignment.ipynb`.

```python
df = pd.read_csv("cardiovascular_disease_dataset.csv")
```
Loads the CSV file into a pandas DataFrame — the standard way to bring tabular data into Python.

```python
df.head()
```
Shows the first 5 rows, to sanity-check that the data loaded correctly and to see example values.

```python
df.shape
```
Returns (rows, columns) — used to confirm the dataset size (1,000 rows × 14 columns originally).

```python
df.info()
```
Lists each column's data type and non-null count — used to spot which columns are numeric and whether pandas detects any missing (`NaN`) values.

```python
df.describe()
```
Gives summary statistics (mean, std, min, max, quartiles) for numeric columns — this is how the implausible `serumcholestrol` minimum of 0 was first noticed.

```python
df.isnull().sum()
```
Counts explicit `NaN` values per column — returns all zeros here, which is why the separate `serumcholestrol == 0` check was needed to catch the disguised missing values.

```python
df = df.drop(columns=["patientid"])
```
Removes the `patientid` identifier column, which carries no predictive information (satisfies the "drop irrelevant features" instruction).

```python
zero_chol_count = (df["serumcholestrol"] == 0).sum()
df["serumcholestrol"] = df["serumcholestrol"].replace(0, np.nan)
```
Counts, then converts, the physiologically-impossible 0 values in `serumcholestrol` into proper missing values (`NaN`). The actual replacement value is computed later (see below), after the train/test split, so no test-set information leaks into preprocessing.

```python
median_chol = X_train["serumcholestrol"].median()
X_train["serumcholestrol"] = X_train["serumcholestrol"].fillna(median_chol)
X_test["serumcholestrol"] = X_test["serumcholestrol"].fillna(median_chol)
```
Computes the median **only from the training set's** valid (non-missing) `serumcholestrol` values, then fills missing values in both the training and test sets with that same number. Computing the median from training data only (not the full dataset) keeps this step leakage-free, exactly like the later `StandardScaler` step.

```python
X = df.drop(columns=["target"])
y = df["target"]
```
Separates the input features (`X`) from the target label (`y`), which every Scikit-learn model requires.

```python
train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
```
Splits the data into 80% training / 20% testing. `test_size=0.20` sets the required 20% test size. `random_state=42` makes the split reproducible (same split every run). `stratify=y` keeps the same class ratio (~58/42) in both the train and test sets.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
`StandardScaler` rescales each feature to mean 0, standard deviation 1 — needed because KNN uses distances between points. `fit_transform` learns the mean/std **from training data only** and scales it; `transform` applies those same learned values to the test data, so no information from the test set leaks into preprocessing.

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
```
Performs 5-fold cross-validation **within the training set only**, using F1-score, to compare hyperparameter choices without ever touching the test set.

```python
DecisionTreeClassifier(max_depth=best_dt_depth, random_state=RANDOM_STATE)
```
Creates a Decision Tree classifier with the chosen `max_depth` (6). `random_state` makes tie-breaking during tree construction reproducible.

```python
GaussianNB(var_smoothing=best_vs)
```
Creates a Gaussian Naive Bayes classifier, which models each feature's distribution per class as a Gaussian curve. `var_smoothing` adds a small value to variances for numerical stability.

```python
KNeighborsClassifier(n_neighbors=best_k)
```
Creates a KNN classifier that will vote among the `best_k` (21) nearest training points.

```python
model.fit(X_train, y_train)
```
Training: the model learns patterns from the training features and labels (e.g. the tree learns its splits, KNN simply stores/indexes the training points).

```python
model.predict(X_test)
```
Prediction: the trained model outputs a class label (0 or 1) for each row in the test set.

```python
accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred, zero_division=0)
recall_score(y_true, y_pred, zero_division=0)
f1_score(y_true, y_pred, zero_division=0)
```
Compute the four required evaluation metrics by comparing predicted labels to the true test labels. `zero_division=0` prevents an error/warning if a model predicts no positives at all for some fold.

```python
confusion_matrix(y_test, pred)
```
Produces a 2×2 table of true vs. predicted classes (true negatives, false positives, false negatives, true positives), used for the confusion-matrix visualization.

```python
results.to_csv("results.csv", index=False)
```
Saves the final comparison table to a CSV file so results are stored outside the script/notebook as well.
