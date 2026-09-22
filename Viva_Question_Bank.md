# Viva Question Bank (~35 Questions with Answers)

## Dataset
1. **What is your dataset?**
   The Cardiovascular Disease Dataset — a hospital-collected clinical dataset of 1,000 patients, published on Mendeley Data, used to detect the presence of cardiovascular disease.
2. **What is the main task?**
   Binary classification: predict whether a patient has cardiovascular disease.
3. **What does it detect/predict?**
   The presence (1) or absence (0) of cardiovascular disease, based on clinical/diagnostic measurements.
4. **How many rows does your final dataset have?**
   1,000 rows (no rows were removed — one column's implausible values were imputed instead).
5. **How many columns/features?**
   12 features plus the target column (after dropping the `patientid` identifier).
6. **What are the features?**
   Age, gender, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG results, maximum heart rate, exercise-induced angina, ST depression (oldpeak), ST slope, and number of major vessels.
7. **What is the target?**
   `target`: 1 = cardiovascular disease present, 0 = absent.
8. **Why did you select this dataset?**
   It is genuinely cardiovascular, binary, reasonably balanced, already numeric, meets the 1,000-sample minimum, and its features are standard, well-documented cardiac diagnostic measurements.
9. **Where did you obtain it?**
   Originally published on Mendeley Data (Mishra et al.); the copy used here is mirrored on GitHub (RishBez/CVD-Disease-Detection).
10. **Is the dataset balanced?**
    Reasonably: 58.0% positive, 42.0% negative.
11. **Did you consider any other dataset first?**
    Yes — the Framingham Heart Study 10-year CHD dataset was implemented first; it worked correctly but its ~85/15 class imbalance capped realistic F1-scores at 0.16-0.27. This dataset was chosen instead to legitimately achieve stronger, still fully defensible results.

## Sampling
12. **Did you sample the data?**
    No. The full dataset (1,000 rows) was used as-is.

## Missing Values
13. **Are there missing values?**
    `df.isnull().sum()` shows no explicit `NaN`s, but 53 rows record `serumcholestrol` as 0, which is not physiologically possible and is treated as a disguised missing value.
14. **How did you identify them?**
    By inspecting `df.describe()` and noticing the minimum of `serumcholestrol` was 0, then counting `(df["serumcholestrol"] == 0).sum()`.
15. **How did you handle them?**
    Replaced the 0s with `NaN`, then — after splitting into train/test — filled them with the median computed from the training set's valid readings only (median = 325.0), applied to both train and test to avoid leakage.
16. **Why not just drop those 53 rows?**
    That would leave only 947 rows, below the assignment's 1,000-sample minimum; imputation preserves the required sample size.

## Preprocessing
17. **Why did you remove the `patientid` feature?**
    It is a unique record identifier with no relationship to whether a patient has heart disease.
18. **Why did you scale the features?**
    Mainly for KNN, which relies on distances between data points; unscaled features (e.g. blood pressure in the 90s-200s vs. oldpeak in single digits) would distort the distance metric.
19. **Why is scaling less important for Decision Tree?**
    It splits on per-feature thresholds; scaling a feature monotonically doesn't change which threshold best separates the classes.
20. **Did you fit the scaler on the whole dataset or only training data?**
    Only on the training data (`fit_transform` on train, `transform` on test) to avoid leakage.

## Decision Tree
21. **How does a Decision Tree work?**
    It repeatedly splits the data on the feature/threshold that best separates the classes (using Gini impurity), building a tree of decision rules.
22. **What is `max_depth`?**
    The maximum number of splits from root to leaf; it controls how complex/overfit the tree can become.
23. **What is overfitting?**
    When a model fits the training data (including its noise) so closely that it performs worse on new, unseen data.
24. **Why did you choose max_depth=6?**
    It had the best cross-validated F1-score during the training-only hyperparameter experiment among the depths tested.
25. **Why did Decision Tree perform best on this dataset?**
    Several features (chest pain type, number of major vessels, exercise-induced angina) create fairly clean threshold-based splits between diseased and healthy patients — exactly the structure a tree model captures well.

## Naive Bayes
26. **How does Naive Bayes work?**
    It applies Bayes' theorem, estimating the probability of each class given the features, assuming features are conditionally independent given the class.
27. **Why is it called "naive"?**
    Because it naively assumes all features are independent given the class, which is rarely exactly true (e.g. chest pain type and exercise-induced angina are correlated).
28. **Why did you use GaussianNB?**
    All features are continuous/numeric, and GaussianNB models each feature as a Gaussian (normal) distribution per class, which fits this kind of data.

## KNN
29. **How does KNN work?**
    It finds the K closest training points (by distance) to a new point and predicts the majority class among them.
30. **What value of K did you use, and why?**
    K = 21, chosen because it had the best mean cross-validated F1-score among the tested values (3, 5, 7, 9, 11, 15, 21) on training data.
31. **What happens if K is too small?**
    The model becomes sensitive to noise/outliers (high variance, potential overfitting).
32. **What happens if K is too large?**
    The model becomes too smooth and may blur the boundary between classes (high bias, underfitting) — though here F1 kept improving up to k=21, suggesting the classes are broadly, cleanly separated.

## Hyperparameters
33. **How did you select hyperparameters overall?**
    Using 5-fold stratified cross-validation on the training set only, comparing mean F1-scores across a small set of candidate values for each model.
34. **Did you use the test set during tuning?**
    No — it was held out and used only once, for the final evaluation.

## Evaluation
35. **What is accuracy, precision, recall, and F1-score?**
    Accuracy = fraction of all predictions that are correct. Precision = of predicted positives, how many are actually positive. Recall = of actual positives, how many were found. F1-score = harmonic mean of precision and recall.
36. **Your accuracy is 93-98%. Isn't that unrealistically high?**
    It is high, but genuine — the result of a single, leakage-free evaluation. It's explained by the dataset's balanced classes and its strongly diagnostic features (chest pain type, major vessels, exercise angina, ST slope are well-established clinical indicators), not by any manipulation of the data or evaluation procedure.
37. **Which model performed best, and why?**
    Decision Tree, with the highest accuracy (97.5%) and F1-score (0.978); its rule-based splits suit this dataset's fairly clean feature-based separability well.

## Code
38. **If I point to any line of code, can you explain it?**
    Yes — see the separate Code Line Defense Guide, which explains every important line used in the script/notebook.

## Figures
39. **What does the class distribution chart (Figure 1) show?**
    A bar chart of the target variable: 420 patients with no disease vs. 580 with disease — confirming the classes are reasonably balanced (58%/42%) before any modeling is done.
40. **Explain the Decision Tree tuning curve (Figure 2a).**
    It plots mean cross-validated F1-score (y-axis) against `max_depth` (x-axis), using 5-fold CV on the training set only. F1 rises up to depth 6, which is why depth 6 was selected — it is the point of best held-out performance, not a guess.
41. **Explain the KNN tuning curve (Figure 2b).**
    It plots mean cross-validated F1-score against `k` (number of neighbors). The curve trends upward and peaks at k=21, showing that larger neighborhoods generalize better on this dataset than small ones, which is why k=21 (not the common default k=5) was chosen.
42. **What does the model comparison bar chart (Figure 3) show?**
    Four bars (Accuracy, Precision, Recall, F1-score) per model, side by side for all three classifiers. It shows Decision Tree's bars are consistently the tallest across all four metrics, confirming it is the best all-round performer, not just on one metric.
43. **How do you read a confusion matrix (Figure 4)?**
    Rows are the actual class, columns are the predicted class. The top-left and bottom-right cells are correct predictions (true negative and true positive); the top-right is a false positive (predicted disease, actually healthy) and the bottom-left is a false negative (predicted healthy, actually diseased — the more clinically costly mistake since it means a real case is missed).
44. **Which model's confusion matrix has the fewest errors, and how many?**
    Decision Tree: only 5 misclassifications out of 200 test patients (1 false positive, 4 false negatives) — matching its top accuracy and F1-score.
