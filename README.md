# Binary Classification on a Healthcare Dataset
### Comparing Decision Tree, Naive Bayes and K-Nearest Neighbors for Cardiovascular Disease Prediction

CSE 3208 Machine Learning term assignment — a binary classification pipeline that predicts the
presence of cardiovascular disease from clinical and diagnostic measurements, comparing three
Scikit-learn classifiers: **Decision Tree**, **Naive Bayes (GaussianNB)**, and **K-Nearest
Neighbors**.

## Dataset

**Cardiovascular Disease Dataset** — a hospital-collected clinical dataset of 1,000 patients,
published on [Mendeley Data](https://data.mendeley.com/) (Mishra et al.). 12 clinical features
(chest pain type, blood pressure, cholesterol, ECG results, exercise test results, etc.) and a
binary target `target` (1 = disease present, 0 = absent), reasonably balanced at 58%/42%.

## Results (test set, n=200)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Decision Tree** | **0.9750** | 0.9912 | 0.9655 | **0.9782** |
| Naive Bayes | 0.9400 | 0.9333 | 0.9655 | 0.9492 |
| KNN | 0.9350 | 0.9402 | 0.9483 | 0.9442 |

All hyperparameters (`max_depth`, `n_neighbors`, `var_smoothing`) were selected using 5-fold
stratified cross-validation on the training set only — the test set was evaluated exactly once,
after all model selection was finished, to avoid data leakage.

## Repository contents

| File | Description |
|---|---|
| `heart_disease_prediction.py` | Standalone Python script — the full pipeline |
| `Healthcare_ML_Assignment.ipynb` | Same pipeline as a Colab-ready Jupyter notebook |
| `cardiovascular_disease_dataset.csv` | The dataset used |
| `results.csv` | Final test-set metrics for all three models |
| `*.png` | Class distribution, confusion matrices, model comparison, and hyperparameter tuning curves |
| `Term_Assignment_Report_v2.docx` | Full written report (introduction, dataset, methods, results, conclusion) |
| `Viva_Cheat_Sheet.md` | Quick-reference sheet for viva/defense |
| `Viva_Question_Bank.md` | ~35 rehearsed viva questions and answers |
| `Mock_Viva.md` | Full mock viva with examiner-style Q&A |
| `Code_Line_Defense_Guide.md` | Line-by-line explanation of the code |
| `Design_Decisions_and_Thinking.md` | The reasoning behind every preprocessing/modeling choice |
| `Requirement_Mapping.md` | Assignment requirement → implementation → evidence table |
| `previous_version_framingham/` | An earlier iteration using the Framingham Heart Study dataset, kept for transparency |

## Methodology notes

- **Missing values**: `serumcholestrol` had 53 rows recorded as `0` (physiologically
  impossible) — treated as missing and imputed using the **training set's** median only,
  applied to both train and test, to avoid leakage.
- **Scaling**: `StandardScaler` fit on training data only, applied to both sets — used for the
  distance-based KNN and for Naive Bayes; the Decision Tree uses unscaled features.
- **No manipulation**: no rebalancing of the test set, no tuning on the test set, and no
  removal of difficult examples. All reported numbers come from a single final evaluation.

## Running it

```bash
pip install pandas numpy matplotlib scikit-learn
python heart_disease_prediction.py
```

or open `Healthcare_ML_Assignment.ipynb` in Jupyter / Google Colab.
