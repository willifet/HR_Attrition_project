# HR_Attrition_projectProject: Logistic Regression — Baseline vs. Optimization
Files:

Logistic Regression Practice.ipynb — baseline pipeline (minimal transforms, default LogisticRegression)

Logistic Regression FT.ipynb — fine-tuned pipeline (feature engineering, encoding, regularization, solver & threshold tuning)

HR_commObjective

**Objective**
Demonstrate how algorithm optimization (feature engineering, regularization, solver selection, hyperparameter search, and threshold tuning) can measurably improve the performance of a Logistic Regression model for employee attrition prediction using the HR_comma_sep.csv dataset.a_sep.csv — dataset (expected in project root)

Step-by-step procedure

This section walks through the steps used in the notebooks and how to run the experiments in an ordered, reproducible way.

1. Data loading & quick EDA

Load HR_comma_sep.csv with pandas.read_csv.

Inspect shapes, nulls, and distributions (df.info(), df.describe(), value_counts()).

Visual checks: class balance on left, relationships between satisfaction_level, average_montly_hours, sales, salary, etc.

Why: baseline understanding identifies features to keep, discard, or transform (and whether class imbalance needs attention).

2. Baseline model (Notebook: Logistic Regression Practice.ipynb)

Select a small set of features (example: satisfaction_level, average_montly_hours, promotion_last_5years, salary).

Encode categorical features minimally (or not at all).

Split data into train/test (e.g., train_test_split(random_state=42)).

Train sklearn.linear_model.LogisticRegression() with defaults.

Evaluate: accuracy, precision, recall, f1-score, ROC AUC, and confusion matrix.

Output: baseline performance metrics and baseline confusion matrix.

3. Optimization pipeline (Notebook: Logistic Regression FT.ipynb)

Improvements applied (explainers and typical code hints below):

Feature engineering

One-hot encode salary (or sales) with pd.get_dummies() or OneHotEncoder.

Create interaction terms (example: satisfaction_level * average_montly_hours), polynomial terms if needed, or binning for numerical features.

Feature scaling

Standardize continuous variables (e.g., StandardScaler) before training (especially necessary for regularized solvers).

Regularization & hyperparameter tuning

Tune C (inverse regularization strength) and penalty (l1/l2/elasticnet) with GridSearchCV or RandomizedSearchCV.

Solver selection

Try liblinear, lbfgs, saga — some solvers handle l1 better (saga/liblinear), others scale to larger feature spaces (lbfgs).

Cross-validation

Use StratifiedKFold to ensure class balance across folds during parameter search.

Threshold tuning

Choose classification threshold based on business objective (optimize for recall vs precision). Use ROC/precision-recall curves and choose threshold that gives desired tradeoff.

Evaluation

Evaluate tuned model on holdout test set. Report metrics and compare to baseline.

Output: tuned model metrics, confusion matrix, ROC AUC, and a clear demonstration of improvement vs baseline.

How algorithm optimization improves accuracy — explainer

Below are concrete reasons the optimizations above typically improve a Logistic Regression model’s performance, with intuitive cause → effect:

Feature engineering → stronger signals

Categorical variables like salary hide non-linear relationships. One-hot encoding exposes those relationships so the linear model can assign distinct coefficients. Creating interaction terms reveals multiplicative effects between variables.

Effect: the model can separate classes better, improving accuracy and AUC.

Feature scaling → better solver behavior & regularization

Regularization penalizes coefficients. If features are on different scales, the penalty distorts training. Scaling ensures fair penalization and faster convergence.

Effect: better generalization and numerical stability.

Regularization tuning → bias-variance balance

C controls under/overfitting. Proper tuning prevents overfitting to noise and improves test set accuracy.

Effect: reduces variance without increasing bias too much.

Solver choice → convergence & performance

Solvers differ in speed, stability, and ability to use l1 penalty. Choosing the appropriate solver for the penalty and feature size ensures fit converges to the correct solution.

Effect: avoids suboptimal fits and training instability.

Cross-validation & hyperparameter search → robust parameter estimates

CV prevents overfitting hyperparameters to a particular train/test split, yielding more reliable metrics.

Effect: improvements generalize to unseen data.

Threshold tuning → business-aligned decisions

Accuracy alone ignores business costs of false positives vs false negatives. Tuning the probability threshold (e.g., from 0.5 to 0.35) can maximize recall for risk detection tasks, improving actionable performance.

Baseline:
- Accuracy: 0.73
- Precision: 0.70
- Recall: 0.68
- ROC AUC: 0.78

Fine-tuned:
- Accuracy: 0.87
- Precision: 0.84
- Recall: 0.86
- ROC AUC: 0.92

Notes: Feature engineering + regularization tuned with GridSearchCV led to +14pp in accuracy in this run.


Effect: better practical performance even if raw accuracy changes modestly.