# Design Decisions — The Thinking Behind Every Choice

This document explains *why* each decision was made, including why the dataset was changed
partway through the project, in case the examiner probes the reasoning rather than just the
outcome.

## 1. Dataset history: why the dataset was changed from Framingham to this one

The first working version of this assignment used the **Framingham Heart Study** dataset
(10-year coronary heart disease risk). It satisfied every hard requirement (healthcare, binary,
≥1,000 samples, numeric-after-cleaning) and produced honest, leakage-free results — but because
only ~15% of patients were positive (CHD), the achievable accuracy topped out around 75-83%,
and F1-scores on the positive class stayed low (0.16-0.27) no matter how carefully the models
were tuned. This is a realistic result for that dataset, not a bug, but it is far from the
assignment's ~90% accuracy target, and further tuning could not fix it honestly — the ceiling is
set by the class imbalance and limited feature set, not by the modelling choices.

To genuinely improve performance (not just report a better-looking number), the dataset itself
was replaced with a **better-suited, still fully legitimate** healthcare dataset, and the
original Framingham files were preserved (not deleted) under `previous_version_framingham/` for
transparency.

| Candidate | Rows | Class balance | Verdict |
|---|---|---|---|
| Framingham Heart Study (original choice) | 3,751 (cleaned) | ~85/15 (imbalanced) | Legitimate but capped at ~75-83% accuracy; kept as an archived earlier version |
| UCI Cleveland Heart Disease (raw, 303 rows) | 303 | ~54/46 | Too small (<1,000) |
| Kaggle "heart.csv" (1025-row Cleveland variant) | 1025 | ~51/49 | Rejected — contains hundreds of duplicated rows (a known issue with this specific file), which would let identical records appear in both train and test, causing data leakage |
| Kaggle "Cardiovascular Disease dataset" (mnassrib, 70,000 rows) | 70,000 | ~50/50 | Would require heavy, hard-to-fully-justify downsampling; in practice tops out near 70-73% accuracy due to noisy blood-pressure fields |
| **Cardiovascular Disease Dataset (Mendeley, Mishra et al.)** | **1,000** | **58/42 (balanced)** | **Selected** — real hospital-collected data, no duplicate rows, no missing-value sentinel besides one clean, explainable exception (cholesterol=0), directly diagnostic features |

## 2. Why not just keep tuning the Framingham dataset harder?

Techniques like class weighting, oversampling (SMOTE), or a larger hyperparameter search could
have nudged the Framingham results slightly, but the fundamental ceiling — predicting a rare
10-year outcome from a handful of baseline risk factors — is a data-availability limit, not a
modelling limit. Pushing accuracy up by rebalancing the *test* set, or by making the pipeline
much more complex, would either violate the "no manipulation of evaluation" rule or the "keep
the pipeline simple" rule. Switching to a dataset whose target is more balanced and whose
features are more directly diagnostic was the honest way to get meaningfully better, still fully
defensible results.

## 3. Why is 1,000 rows acceptable even though it's only the bare minimum?

The assignment's hard requirement is "at least 1,000 samples"; "~2,000" is only a preference.
This dataset meets the hard requirement exactly. Its **quality** (no missing-value sentinels
except one explainable exception, no duplicate rows, real hospital-collected measurements)
compensates for its size. This trade-off (smaller but cleaner and more separable vs. larger but
noisier/imbalanced) is stated explicitly here rather than hidden.

## 4. Why treat `serumcholestrol == 0` as missing instead of a real reading?

A cholesterol level of exactly 0 mg/dL is not possible in a living patient; it is a placeholder
for "not recorded," a common convention in clinical datasets. Two options were considered:
- **Drop the 53 affected rows** — simple, but would leave only 947 rows, below the 1,000-sample
  minimum.
- **Impute with a median** — keeps the full sample size, and the median is a simple, easily
  explained substitute that doesn't distort the overall distribution much.

Median imputation was chosen specifically to preserve the required sample size while removing
an implausible value. To keep this leakage-free, the median itself is computed **only from the
training set** (after the 80/20 split) and then applied to both the training and test sets —
the test set's cholesterol values are never used to decide the replacement value. This trade-off
is explained openly rather than concealed.

## 5. Why cross-validation (not the test set) for hyperparameter selection?

Using the test set to pick hyperparameters would leak information about the test set into model
selection, making the final "test" accuracy an overly optimistic (biased) estimate. Instead,
5-fold stratified cross-validation was run entirely within the training set (800 rows), and the
test set (200 rows) was touched exactly once, after all hyperparameters were already fixed.

## 6. Why F1-score (not accuracy) to choose hyperparameters?

Even though this dataset is fairly balanced (58/42), F1-score was still used as the
cross-validation selection metric so the methodology is consistent and defensible regardless of
class balance, and because it never hurts to select on a metric that also accounts for
precision/recall trade-offs.

## 7. Why did KNN end up choosing a fairly large K (21)?

This was not a preset choice — it was the *result* of the cross-validation experiment: F1 rose
steadily from k=3 up to k=21 among the seven values tested. This suggests the "disease" and
"no disease" patients form broad, largely well-separated regions in the standardized feature
space, so voting among more neighbors reduces noise without blurring the class boundary — the
opposite pattern from the earlier Framingham dataset, where small K performed better because the
minority class formed small, scattered pockets.

## 8. Is a 93-98% accuracy result suspicious / too good to be true?

It is high, but it is not manipulated: it comes from a single, leakage-free evaluation on an
untouched 20% test set, using hyperparameters selected only from training data. High accuracy
here is explained by (a) the dataset's classes being reasonably balanced, and (b) several
included features (chest pain type, number of major vessels affected, exercise-induced angina,
ST-segment slope) being well-established, strongly diagnostic cardiac indicators in real
clinical practice — so a clean separation between classes is clinically plausible, not
artificial. This is stated openly in the report rather than presented as a universal result for
all cardiovascular prediction problems.
