import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("\nFRAUDSHIELD AI - MODEL TRAINING STARTED\n")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "creditcard.csv not found. Put it inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

required_columns = ["Time", "Amount", "Class"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

if "Class" not in df.columns:
    raise ValueError("Dataset must contain the Class column.")

X = df.drop(columns=["Class"])
y = df["Class"]

if set(y.unique()) != {0, 1}:
    raise ValueError("Class must contain both 0 (genuine) and 1 (fraud).")

print("Dataset loaded:", df.shape)
print("Genuine transactions:", int((y == 0).sum()))
print("Fraudulent transactions:", int((y == 1).sum()))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), ["Time", "Amount"])
    ],
    remainder="passthrough"
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )
}

results = []
trained_models = {}
predictions = {}

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("smote", SMOTE(
            sampling_strategy=0.25,
            random_state=42
        )),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predicted = (probabilities >= 0.5).astype(int)

    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    metrics = {
        "Model": name,
        "PR-AUC": float(pr_auc),
        "ROC-AUC": float(roc_auc),
        "Precision": float(
            precision_score(y_test, predicted, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_test, predicted, zero_division=0)
        ),
        "F1 Score": float(
            f1_score(y_test, predicted, zero_division=0)
        )
    }

    results.append(metrics)
    trained_models[name] = pipeline

    predictions[name] = {
        "probabilities": probabilities,
        "predicted": predicted
    }

    print(metrics)

best_model_name = max(
    results,
    key=lambda item: item["PR-AUC"]
)["Model"]

best_model = trained_models[best_model_name]

best_prediction = predictions[best_model_name]["predicted"]
best_probabilities = predictions[best_model_name]["probabilities"]

comparison = pd.DataFrame(results)

artifact = {
    "model": best_model,
    "best_model_name": best_model_name,
    "comparison": comparison,
    "confusion_matrix": confusion_matrix(
        y_test, best_prediction
    ),
    "y_test": y_test.to_numpy(),
    "probabilities": best_probabilities,
    "feature_names": list(X.columns),
    "test_metrics": next(
        item for item in results
        if item["Model"] == best_model_name
    )
}

joblib.dump(artifact, f"{MODEL_DIR}/fraud_model.pkl")

comparison.to_csv(
    f"{MODEL_DIR}/model_comparison.csv",
    index=False
)

print("\n----------------------------------")
print("BEST MODEL:", best_model_name)
print("----------------------------------")
print("\nModel comparison:")
print(comparison.to_string(index=False))
print("\nModel saved successfully!")
print("FRAUDSHIELD AI TRAINING COMPLETED.")