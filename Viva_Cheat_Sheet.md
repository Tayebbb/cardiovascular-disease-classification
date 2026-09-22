# Viva Cheat Sheet — 5-Minute Revision

## Dataset
- **Name:** Cardiovascular Disease Dataset (`cardiovascular_disease_dataset.csv`)
- **Source:** Mendeley Data (Mishra et al.) — a hospital-collected clinical dataset, mirrored on GitHub (RishBez/CVD-Disease-Detection)
- **Rows:** 1,000 (no rows were dropped — implausible values were imputed instead, see below)
- **Columns:** 14 original → 13 after cleaning (12 features + target), after dropping `patientid`
- **Target:** `target` (1 = cardiovascular disease present, 0 = absent)
- **Classes:** 580 positive (58.0%) / 420 negative (42.0%) — reasonably balanced
- **Healthcare problem:** detecting the presence of cardiovascular disease from clinical/diagnostic measurements collected during a cardiac work-up
- **Sampling:** none — full dataset (1,000 rows) used as-is
- **Missing values:** no explicit `NaN`s, but `serumcholestrol` had 53 rows recorded as 0 (physiologically impossible) — treated as missing and imputed with the median of valid readings, to avoid dropping below the 1,000-sample minimum

## Preprocessing
- Dropped `patientid` (identifier, no predictive value)
- Replaced `serumcholestrol == 0` with `NaN`, then (after the train/test split) filled with the **training-set-only** median (median = 325.0) — applied to both train and test to avoid leakage
- Train/test split: 80/20, `random_state=42`, `stratify=y`
- Scaling: `StandardScaler`, fit on train only, applied to test — used for KNN (distance-based) and NB

## Models & Final Hyperparameters
| Model | Hyperparameter(s) | Final value | How chosen |
|---|---|---|---|
| Decision Tree | `max_depth` | `6` | 5-fold CV on training data, best mean F1 (0.963) |
| Naive Bayes | `var_smoothing` | `1e-9` (library default) | All 5 tested values gave identical CV F1 (0.950) |
| KNN | `n_neighbors` | `21` | 5-fold CV on training data, best mean F1 (0.951) |

The report includes tuning curves (`dt_depth_tuning.png`, `knn_k_tuning.png`) plotting mean CV F1 against `max_depth` and `k` — useful if the examiner asks to see how the chosen values were picked visually.

## Final Test Results (200 patients, test set)
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Decision Tree** | **0.9750** | 0.9912 | 0.9655 | **0.9782** |
| Naive Bayes | 0.9400 | 0.9333 | 0.9655 | 0.9492 |
| KNN | 0.9350 | 0.9402 | 0.9483 | 0.9442 |

## Error Breakdown (out of 200 test patients)
| Model | Correct | Misclassified | False Positives | False Negatives |
|---|---|---|---|---|
| Decision Tree | 195 | 5 | 1 | 4 |
| Naive Bayes | 188 | 12 | 8 | 4 |
| KNN | 187 | 13 | 7 | 6 |

## Figure Explanations

**Figure 1 — `class_distribution.png` (Class Distribution of the Target Variable)**
A bar chart with two bars: "No Disease (0)" and "Disease (1)". The y-axis is the number of
patients. Bar 1 (No Disease) is at 420, bar 2 (Disease) is at 580. Purpose: to visually confirm
the target is reasonably balanced (58%/42%) before modeling, so accuracy is a meaningful metric
and no class-imbalance correction (e.g. oversampling) is needed.

**Figure 2a — `dt_depth_tuning.png` (Decision Tree: CV F1-score vs. max_depth)**
A line plot. X-axis: `max_depth` values tested (3, 4, 5, 6, 8, 10, None). Y-axis: mean
cross-validated F1-score (from 5-fold CV on the training set only). The curve rises from a low
F1 at depth 3, peaks at **depth 6**, then slightly dips and plateaus toward `None` (unrestricted
depth). Purpose: shows *why* depth 6 was chosen — it is the point of best generalization
performance on held-out training folds, not an arbitrary guess, and not deeper is not always
better (very shallow trees underfit; the curve shows the trade-off).

**Figure 2b — `knn_k_tuning.png` (KNN: CV F1-score vs. Number of Neighbors k)**
A line plot. X-axis: `k` values tested (3, 5, 7, 9, 11, 15, 21). Y-axis: mean cross-validated
F1-score (5-fold CV, training set only, scaled features). The curve generally trends upward as
k increases, peaking at **k=21**, the largest value tested. Purpose: shows that this dataset's
two classes form broad, well-separated regions in feature space — larger neighborhoods vote more
reliably than small, noise-sensitive ones here — which is why a comparatively large k was
selected rather than the commonly-assumed "default" k=5.

**Figure 3 — `model_comparison.png` (Model Comparison: Accuracy, Precision, Recall, F1-score)**
A grouped bar chart. X-axis: the three models (Decision Tree, Naive Bayes, KNN). For each model,
four bars (Accuracy, Precision, Recall, F1-score), y-axis 0 to 1. All bars are visibly high
(roughly 0.93-0.99) and close together across models, with Decision Tree's bars slightly taller
than the other two on every metric. Purpose: a single-glance comparison confirming Decision Tree
is the best performer across all four metrics simultaneously, not just accuracy.

**Figure 4 — `confusion_matrices.png` (Confusion Matrices for the Three Models)**
Three side-by-side 2x2 grids (one per model), each cell shaded by count (darker = more
patients). Rows = actual class, columns = predicted class. Reading a matrix: top-left =
correctly predicted "no disease" (true negative), bottom-right = correctly predicted "disease"
(true positive), top-right = false positive (predicted disease, actually healthy), bottom-left =
false negative (predicted healthy, actually diseased — the more clinically costly error).
Decision Tree's matrix has the fewest off-diagonal (error) counts (1 FP + 4 FN = 5 total errors
out of 200), confirming its top F1-score numerically. See the Error Breakdown table above for
exact counts per model.

## Key Explanations (rapid-fire)
- **Why scale for KNN?** Distance-based; unscaled features like `restingBP`/`serumcholestrol` (large numbers) would dominate distance over small-scale features like `oldpeak`.
- **Why not scale for Decision Tree?** It splits on raw thresholds per feature independently; scaling doesn't change which splits are chosen.
- **Why max_depth=6?** Best cross-validated F1 among {3,4,5,6,8,10,None} on training data only.
- **Why K=21 (fairly large)?** Best cross-validated F1 among {3,5,7,9,11,15,21}; a larger K suggests the two classes form broad, well-separated regions rather than needing fine local boundaries.
- **Why is accuracy so high (93-98%) here?** The features (chest pain type, number of major vessels, exercise-induced angina, ST slope) are strong, direct diagnostic indicators of cardiovascular disease, and classes are reasonably balanced (58/42) — unlike a rare-event dataset, this task is genuinely easier.
- **Why 20% test size?** Standard practice; 200 patients is enough for a stable estimate while keeping 800 for training.
- **Why this dataset (and not the Framingham 10-year CHD dataset also considered)?** Framingham's target was highly imbalanced (~15% positive), which capped realistic accuracy/F1 around 75-83%/0.16-0.27. This dataset is balanced and more directly diagnostic, allowing honestly higher, well-supported performance while still meeting every assignment requirement.
- **Was the test set touched during tuning?** No — hyperparameters were chosen using 5-fold cross-validation on the training set only; the test set was used exactly once, at the end. (A common shortcut is to pick hyperparameters by directly checking test-set accuracy — that quietly leaks test-set information into model selection and makes the final number overly optimistic; this pipeline avoids that.)
- **Why impute cholesterol instead of dropping those rows?** Dropping the 53 rows would leave only 947 samples, below the assignment's 1,000-sample minimum; median imputation keeps the full sample size while removing an implausible value.
- **How was the disguised missing value found?** `isnull().sum()` alone shows all zeros (no explicit `NaN`s) — the 0 values in `serumcholestrol` were only caught by checking `describe()`'s minimum value, which is a physiologically impossible 0 mg/dL.
