# Final Mock Viva

Format: **Question → Ideal short answer → What the examiner is testing → Common mistake to avoid**

---

**Q1 (Easy). What problem are you solving?**
A: Predicting whether a patient has cardiovascular disease from clinical/diagnostic measurements collected during a cardiac work-up — a binary classification problem.
Tests: basic understanding of the project.
Avoid: vague answers without naming the actual target/outcome.

---

**Q2 (Easy). Why binary classification and not regression?**
A: The target has exactly two discrete outcomes (disease present or absent), not a continuous value.
Tests: understanding of classification vs. regression.
Avoid: confusing risk *scores* (continuous) with the actual binary label used for training.

---

**Q3 (Medium). Your accuracy is 93-98%. Isn't that suspiciously high for a medical dataset?**
A: It's high because this dataset's classes are reasonably balanced (58/42) and several features (chest pain type, number of major vessels, exercise-induced angina, ST slope) are strong, clinically established diagnostic indicators — not because of any data manipulation. The evaluation was done once on an untouched 20% test set, with hyperparameters selected only from training-only cross-validation.
Tests: whether the student can defend a strong result without either overclaiming or panicking.
Avoid: claiming "all healthcare datasets give this accuracy" — this is dataset-specific, and the report explicitly contrasts it with the earlier, harder Framingham dataset.

---

**Q4 (Medium). Why did you switch datasets partway through the project?**
A: The first dataset (Framingham, 10-year CHD risk) was correctly implemented but had a severe ~85/15 class imbalance that capped F1-scores around 0.16-0.27 no matter how the models were tuned. Switching to a more balanced, more directly diagnostic cardiovascular dataset legitimately improved results without manipulating data or evaluation.
Tests: whether the student understands that some performance ceilings come from the data, not the model, and that switching datasets is a valid response to that (versus artificially inflating results).
Avoid: implying the first dataset or the code was "wrong" — it wasn't; it was simply a harder task.

---

**Q5 (Medium). Why did you treat `serumcholestrol == 0` as missing instead of a real value?**
A: A cholesterol level of exactly 0 mg/dL is impossible in a living patient; it's a placeholder for "not recorded." It was replaced with the column median (computed from the other valid readings) rather than dropped, to keep the sample size at the required 1,000.
Tests: whether the student can spot disguised/sentinel missing values (not just explicit `NaN`s).
Avoid: saying "there were no missing values" — `isnull().sum()` shows none explicitly, but the 0s are a disguised case that should be found via `describe()`.

---

**Q6 (Unexpected). What would happen if you had fit the StandardScaler on the entire dataset before splitting?**
A: The scaler would learn the mean/std using test-set values too, causing data leakage — the test set would no longer be a fair, independent evaluation.
Tests: real understanding of data leakage, not just repeating the correct code.
Avoid: saying "it wouldn't matter" — it would bias the evaluation.

---

**Q7 (Medium). Why did KNN select a fairly large K (21) instead of a small one?**
A: Cross-validation showed F1 improving steadily as K increased from 3 to 21, suggesting the "disease" and "no disease" classes form broad, largely well-separated regions in the feature space — voting among more neighbors reduces noise here without blurring the class boundary.
Tests: ability to connect an algorithm's mechanism to an observed result.
Avoid: assuming small K is always better — it depends on the dataset's structure.

---

**Q8 (Hard/unexpected). If I gave you a new patient's data, walk me through what your pipeline would do.**
A: Drop `patientid` if present, check `serumcholestrol` for a 0 value and replace it with the same median learned from training data if needed, scale the features using the same `StandardScaler` fitted on training data, then feed them to each trained model's `.predict()` to get a 0/1 prediction.
Tests: whether the student understands the pipeline as a sequence of learned transformations, not just training-time code.
Avoid: forgetting that the *same* fitted scaler and the *same* median (not newly computed ones) must be used at prediction time.

---

**Q9 (Dataset). What does `noofmajorvessels = 3` mean?**
A: Three major blood vessels were found to be colored/visible via fluoroscopy — a diagnostic imaging finding used in real cardiac assessments, encoded here as an integer 0-3.
Tests: whether the student actually understands the columns, not just their names.
Avoid: confusing `noofmajorvessels` (an imaging finding) with `chestpain` (a reported symptom type).

---

**Q10 (Report/consistency). Does your report's accuracy number match your code's output?**
A: Yes — Decision Tree: 0.9750, Naive Bayes: 0.9400, KNN: 0.9350, matching `results.csv` and the notebook's printed output exactly.
Tests: consistency between deliverables.
Avoid: rounding differently in different documents without noticing.

---

**Q11 (Code). Explain `stratify=y` in your train_test_split call.**
A: It ensures both the training and test sets keep the same proportion of disease/no-disease cases (~58%/42%) as the full dataset, rather than a random split that could over- or under-represent either class.
Tests: understanding of stratified splitting, not just recognizing the keyword.
Avoid: confusing stratification with scaling or shuffling.

---

**Q12 (Hard). Could you have reached high accuracy by just predicting the majority class?**
A: No — the majority class (disease present) is only 58% of patients, so always predicting it would give ~58% accuracy, far below the 93-98% actually achieved; the models are genuinely learning from the features, not just exploiting class imbalance.
Tests: whether the student checks their result against the trivial baseline.
Avoid: treating high accuracy as automatically suspicious without checking the baseline first.
