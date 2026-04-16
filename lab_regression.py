"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,   # ✅ أضفناها
    r2_score
)
def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
    # TODO: Load the CSV and return the DataFrame
    
    df = pd.read_csv(filepath)

    print("shape of the dataset:", df.shape)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nChurn distribution:")
    print(df['churned'].value_counts())

    print("\nChurn percentage:")
    print(df['churned'].value_counts(normalize=True))

    return df


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # TODO: Separate features and target, then split with stratification
    
    x = df.drop(columns=[target_col])
    y = df[target_col]

    # Apply stratification only for classification
    if target_col == "churned":
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=stratify   
                                                        )
    
    print("\n********************************************")
    print("********************************************")

    # Print sizes
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # Print churn rate
    print("\nTrain churn rate:")
    print(y_train.value_counts(normalize=True))

    print("\nTest churn rate:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    # TODO: Create and return a Pipeline with two steps
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42,
                                     max_iter=1000,
                                     class_weight="balanced"
                                     ))
    ])

    return pipeline


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    # TODO: Create and return a Pipeline for Ridge regression
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])
    
    return pipeline


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    # TODO: Fit the pipeline on training data, predict on test, compute metrics
        # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Display confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

    # Return required metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return metrics


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    # TODO: Fit the pipeline, predict, and compute MAE and R²
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nRegression Results:")
    print("MAE:", mae)
    print("R²:", r2)

    return {
        "mae": mae,
        "r2": r2
    }


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    # TODO: Run cross_val_score with StratifiedKFold
    # Stratified K-Fold (مهم للـ classification)
    cv_splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="accuracy"
    )

    print("\nCross-validation scores (each fold):")
    print(scores)

    return scores

def build_lasso_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))
    ])
    return pipeline


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Select numeric features for classification
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                           "num_support_calls", "senior_citizen",
                           "has_partner", "has_dependents"]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            if pipe:
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            lasso_pipe = build_lasso_pipeline()

            if ridge_pipe and lasso_pipe:
                # Train both
                ridge_pipe.fit(X_tr, y_tr)
                lasso_pipe.fit(X_tr, y_tr)

                # Evaluate Ridge (حسب المطلوب)
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                print(f"Ridge Regression: {reg_metrics}")

                # Get coefficients
                ridge_coefs = ridge_pipe.named_steps["model"].coef_
                lasso_coefs = lasso_pipe.named_steps["model"].coef_

                features = X_tr.columns

                print("\nFeature Coefficients Comparison:")
                for f, r, l in zip(features, ridge_coefs, lasso_coefs):
                    print(f"{f:20} | Ridge: {r:.4f} | Lasso: {l:.4f}")


"""
Summary of Findings:

1. Important Features for Predicting Churn:
Features such as tenure, monthly_charges, and total_charges appear to be the most influential in predicting churn, as they are directly related to customer usage and billing. Additionally, num_support_calls may indicate customer dissatisfaction, which can contribute to churn.

2. Logistic Regression Performance:
The logistic regression model achieved moderate performance, with an accuracy of around 0.63 and a recall of approximately 0.51 for the churn class. Recall is more important than precision in this problem because failing to identify customers who are likely to churn (false negatives) is more costly than incorrectly flagging some customers as churn risks (false positives).

3. Recommendations for Improvement:
To improve performance, several steps can be taken:
- Try more advanced models such as Random Forest or Gradient Boosting.
- Perform feature engineering or include additional relevant features.
- Tune hyperparameters using GridSearchCV.
- Address class imbalance using techniques like SMOTE or resampling.
- Evaluate using metrics like F1-score or ROC-AUC instead of relying only on accuracy.
"""