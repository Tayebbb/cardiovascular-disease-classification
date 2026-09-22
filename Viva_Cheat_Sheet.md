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
