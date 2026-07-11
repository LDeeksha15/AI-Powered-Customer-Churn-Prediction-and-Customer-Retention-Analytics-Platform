"""
churn_model.py
Trains and compares churn prediction models on the real Online Retail
RFM feature table. Recency/R_score are excluded from model inputs to
avoid data leakage (churn label is defined directly from recency).
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# recency / R_score deliberately excluded -- see leakage note in README/notebook
FEATURES = [
    "frequency", "monetary", "avg_order_value",
    "tenure_days", "purchase_rate", "clv",
    "F_score", "M_score"
]


def load_and_prepare():
    df = pd.read_csv("data/rfm_features.csv")
    X = df[FEATURES]
    y = df["churned"]
    return X, y, df


def train_and_compare():
    X, y, df = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
    }

    results = {}
    print("=" * 60)
    print("MODEL COMPARISON (Real Online Retail Dataset)")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "model": model, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc, "cv_mean": cv_scores.mean()
        }

        print(f"\n{name}")
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  Precision: {prec*100:.2f}%")
        print(f"  Recall:    {rec*100:.2f}%")
        print(f"  F1 Score:  {f1*100:.2f}%")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    final_name = "Logistic Regression"
    final_model = results[final_name]["model"]
    final_preds = final_model.predict(X_test_scaled)

    print("\n" + "=" * 60)
    print(f"FINAL MODEL SELECTED: {final_name}")
    print("=" * 60)
    print(f"Test Accuracy: {results[final_name]['accuracy']*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, final_preds))
    print("\nClassification Report:")
    print(classification_report(y_test, final_preds, target_names=["Retained", "Churned"]))

    coef_df = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": final_model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    print("\nFeature Importance (Logistic Regression coefficients):")
    print(coef_df.to_string(index=False))

    joblib.dump(final_model, "models/churn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(FEATURES, "models/feature_columns.pkl")

    # Persist evaluation metrics so the Flask app can display real numbers
    # (not hardcoded/fabricated ones) on the dashboard
    import json
    metrics = {
        "model_name": final_name,
        "accuracy": round(results[final_name]["accuracy"] * 100, 2),
        "precision": round(results[final_name]["precision"] * 100, 2),
        "recall": round(results[final_name]["recall"] * 100, 2),
        "f1_score": round(results[final_name]["f1"] * 100, 2),
        "roc_auc": round(results[final_name]["auc"], 4),
        "cv_mean_accuracy": round(results[final_name]["cv_mean"] * 100, 2),
        "features_used": FEATURES,
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Also persist the FULL comparison across all 3 algorithms, so the UI
    # can show "we compared these models" rather than just the final pick
    comparison = []
    for name, r in results.items():
        comparison.append({
            "model_name": name,
            "accuracy": round(r["accuracy"] * 100, 2),
            "precision": round(r["precision"] * 100, 2),
            "recall": round(r["recall"] * 100, 2),
            "f1_score": round(r["f1"] * 100, 2),
            "roc_auc": round(r["auc"], 4),
            "cv_mean_accuracy": round(r["cv_mean"] * 100, 2),
            "selected": name == final_name,
        })
    with open("models/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nSaved model, scaler, feature list, metrics.json, and model_comparison.json to models/")

    return results


if __name__ == "__main__":
    train_and_compare()
