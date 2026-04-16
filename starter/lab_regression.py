"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

from logging import config
from xml.parsers.expat import model

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
import json


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



def run_model_sweep(config_path, X_train, y_train):
    with open(config_path, "r") as f:
        config = json.load(f)

    results = []

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_cfg in config["models"]:
        model_type = model_cfg["type"]
        params = model_cfg["params"]

        # ===== choose model =====
        if model_type == "logistic":
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight="balanced",
                **params
            )

        elif model_type == "ridge":
            model = Ridge(**params)

        elif model_type == "lasso":
            model = Lasso(**params)

        else:
            continue

        # ===== pipeline =====
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        # ===== CV =====
        if model_type == "logistic":
            scores = cross_val_score(pipe, X_train, y_train,
                                     cv=cv_splitter,
                                     scoring="accuracy")
        else:
            scores = cross_val_score(pipe, X_train, y_train,
                                     cv=5,
                                     scoring="r2")

        results.append({
            "model": model_type,
            "params": params,
            "mean_score": scores.mean(),
            "std": scores.std()
        })

    # ===== print results =====
    print("\nModel Sweep Results:")
    for r in results:
        print(f"{r['model']} | {r['params']} | "
              f"{r['mean_score']:.3f} +/- {r['std']:.3f}")

    return results

#Tier 3: Do logistic Regression from scratch only using NumPy without sklearn.
class MyLogisticRegression:
    def __init__(self, lr=0.01, n_iters=2000, lambda_=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_ = lambda_
        
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_pred):
        m = len(y)
        return - (1/m) * np.sum(
            y * np.log(y_pred + 1e-9) +
            (1 - y) * np.log(1 - y_pred + 1e-9)
        )

    def fit(self, X, y):
        X = X.values
        y = y.values

        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.n_iters):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear)

            dw = (1/m) * np.dot(X.T, (y_pred - y)) + (self.lambda_/m)*self.weights
            db = (1/m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                loss = self.compute_loss(y, y_pred)
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict(self, X):
        X = X.values
        linear = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear)

        print("Min prob:", y_pred.min())
        print("Max prob:", y_pred.max())

        return (y_pred >= 0.3).astype(int)

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
                # ================== LAB ==================
                print("\n**************************************")
                print("************ LAB RESULTS ************")
                print("**************************************")

                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                pipe.fit(X_train, y_train)

                # ================== TIER 1 ==================
                print("\n**************************************")
                print("********* Tier 1: Threshold *********")
                print("**************************************")

                y_probs = pipe.predict_proba(X_test)[:, 1]

                thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

                precisions = []
                recalls = []
                f1s = []

                for t in thresholds:
                    y_pred = (y_probs >= t).astype(int)

                    p = precision_score(y_test, y_pred)
                    r = recall_score(y_test, y_pred)
                    f = f1_score(y_test, y_pred)

                    precisions.append(p)
                    recalls.append(r)
                    f1s.append(f)

                    print(f"\nThreshold: {t}")
                    print(f"Precision: {p:.3f}")
                    print(f"Recall: {r:.3f}")
                    print(f"F1: {f:.3f}")

                best_idx = np.argmax(f1s)
                print(f"\nBest Threshold: {thresholds[best_idx]} with F1: {f1s[best_idx]:.3f}")

                plt.plot(thresholds, precisions, label="Precision")
                plt.plot(thresholds, recalls, label="Recall")
                plt.plot(thresholds, f1s, label="F1")
                plt.xlabel("Threshold")
                plt.ylabel("Score")
                plt.title("Threshold Tuning")
                plt.legend()
                plt.savefig("threshold_plot.png")

                # ================== CROSS VALIDATION ==================
                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

                # ================== TIER 2 ==================
                print("\n**************************************")
                print("********* Tier 2: Model Sweep ********")
                print("**************************************")

                print("\nRUNNING SWEEP...")
                run_model_sweep("starter/model_config.json", X_train, y_train)
                
                
                # ================== TIER 3 ==================
                print("\n**************************************")
                print("***** Tier 3: From Scratch Model *****")
                print("**************************************")

                # 1. Scaling
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # 2. Model
                custom_model = MyLogisticRegression(lr=0.01, n_iters=2000)

                # 3. Train
                custom_model.fit(pd.DataFrame(X_train_scaled), y_train)

                # 4. Predict
                y_pred_custom = custom_model.predict(pd.DataFrame(X_test_scaled))

                # 5. Metrics
                print("\nCustom Logistic Results:")
                print("Accuracy:", accuracy_score(y_test, y_pred_custom))
                print("Precision:", precision_score(y_test, y_pred_custom))
                print("Recall:", recall_score(y_test, y_pred_custom))
                print("F1:", f1_score(y_test, y_pred_custom))

                # ================== Comparison ==================
                print("\n**************************************")
                print("***** Comparison with sklearn *****")
                print("**************************************")

                # sklearn coefficients
                sklearn_weights = pipe.named_steps["model"].coef_[0]

                print("\nFeature Coefficients Comparison:")
                for f, w1, w2 in zip(X_train.columns, sklearn_weights, custom_model.weights):
                    print(f"{f:20} | sklearn: {w1:.4f} | custom: {w2:.4f}")

                sk_preds = pipe.predict(X_test)

                diff = np.sum(sk_preds != y_pred_custom)

                print(f"\nPrediction differences: {diff} out of {len(y_test)} samples")
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

# NOTE:
# Results from custom implementation may differ from sklearn because:
# - sklearn uses optimized solvers (like lbfgs)
# - different convergence criteria
# - default regularization handling