"""
save_model.py
=============
Trains the validated Random Forest pipeline from the TapLend notebook
and saves the fitted pipeline to models/ for use by predict_applicant.py.

This script is a faithful reproduction of the training workflow in
notebooks/taplend_analysis.ipynb. It does NOT change any hyperparameters,
feature sets, or random seeds. Run it once after cloning the project.

Usage:
    python save_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "AssessmentDataScientist.xlsx"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature lists — must stay identical to the notebook
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "amount_requested",
    "loan_duration",
    "payment_amount",
    "monthly_income_amount",
    "monthly_rent_amount",
    "num_payments",
    "raw_FICO_telecom",
    "raw_FICO_retail",
    "raw_FICO_bank_card",
    "raw_FICO_money",
    "raw_l2c_score",
    "payment_to_income",
    "rent_to_income",
    "total_repayment_ratio",
    "mean_fico",
]

CATEGORICAL_FEATURES = [
    "payment_frequency",
    "residence_rent_or_own",
    "bank_account_direct_deposit",
    "email_duration",
    "residence_duration",
    "bank_account_duration",
    "home_phone_type",
]


# ---------------------------------------------------------------------------
# Data loading & matching — identical to the notebook
# ---------------------------------------------------------------------------
def load_and_match():
    applications = pd.read_excel(DATA_PATH, sheet_name="Application Data")
    loan_performance = pd.read_excel(DATA_PATH, sheet_name="Loan Performance")

    # Extract customer keys
    applications["customer_key"] = (
        applications["customer_id"].astype("string").str.strip().str.upper()
    )

    # Justified single correction: leading zero stripped by Excel in Row 567
    mask_nine = applications["customer_key"] == "902431691"
    if mask_nine.any():
        print("Restoring leading zero: Row 567 (902431691 -> 0902431691)")
        applications.loc[mask_nine, "customer_key"] = "0902431691"

    loan_performance["customer_key"] = (
        loan_performance["idLoan"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.split("-")
        .str[0]
    )
    loan_performance["loan_seq"] = (
        loan_performance["idLoan"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.split("-")
        .str[1]
    )

    merged = loan_performance.merge(
        applications,
        on="customer_key",
        how="left",
        indicator=True,
        validate="many_to_one",
        suffixes=("_loan", "_application"),
    )

    matched = merged[merged["_merge"] == "both"].copy()
    print(f"Matched loan records: {len(matched)}")
    print(f"Unmatched loan records: {(merged['_merge'] == 'left_only').sum()}")
    return matched


# ---------------------------------------------------------------------------
# Feature engineering — identical to the notebook
# ---------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    # Target: 1 = Bad (default), 0 = Good (repaid)
    df["target"] = (df["flgGood"] == "Bad").astype(int)

    df["payment_to_income"] = df["payment_amount"] / (
        df["monthly_income_amount"] + 1.0
    )
    df["rent_to_income"] = df["monthly_rent_amount"] / (
        df["monthly_income_amount"] + 1.0
    )
    df["total_repayment_ratio"] = (
        df["num_payments"] * df["payment_amount"]
    ) / (df["amount_requested"] + 1.0)
    df["mean_fico"] = df[
        ["raw_FICO_telecom", "raw_FICO_retail", "raw_FICO_bank_card", "raw_FICO_money"]
    ].mean(axis=1)
    return df


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------
def build_rf_pipeline():
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", num_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ]
    )
    rf = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        class_weight="balanced",
    )
    return Pipeline([("prep", preprocessor), ("model", rf)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("TapLend — Model Training & Saving Script")
    print("=" * 60)

    matched = load_and_match()
    df = engineer_features(matched)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["target"]
    groups = df["customer_key"]

    # Grouped split — same seed as notebook
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(
        f"\nTrain: {len(X_train)} loans  |  Test: {len(X_test)} loans"
    )

    # Train RF pipeline
    rf_pipeline = build_rf_pipeline()
    rf_pipeline.fit(X_train, y_train)

    # Validate on hold-out test set
    y_pred = rf_pipeline.predict(X_test)
    y_prob = rf_pipeline.predict_proba(X_test)

    # Confirm class ordering (0 = Good, 1 = Bad)
    classes = rf_pipeline.named_steps["model"].classes_.tolist()
    assert classes == [0, 1], f"Unexpected class ordering: {classes}"
    good_idx = classes.index(0)
    bad_idx = classes.index(1)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob[:, bad_idx])
    prec_bad = precision_score(y_test, y_pred, pos_label=1)
    rec_bad = recall_score(y_test, y_pred, pos_label=1)
    f1_bad = f1_score(y_test, y_pred, pos_label=1)
    prec_good = precision_score(y_test, y_pred, pos_label=0)
    rec_good = recall_score(y_test, y_pred, pos_label=0)
    f1_good = f1_score(y_test, y_pred, pos_label=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\nHold-out test metrics (should match notebook outputs):")
    print(f"  Accuracy:          {acc:.4f}  (expected 0.6241)")
    print(f"  ROC-AUC:           {auc:.4f}  (expected 0.6313)")
    print(f"  Precision (Bad):   {prec_bad:.4f}  (expected 0.6265)")
    print(f"  Recall (Bad):      {rec_bad:.4f}  (expected 0.7324)")
    print(f"  F1 (Bad):          {f1_bad:.4f}  (expected 0.6753)")
    print(f"  Precision (Good):  {prec_good:.4f}  (expected 0.6200)")
    print(f"  Recall (Good):     {rec_good:.4f}  (expected 0.5000)")
    print(f"  F1 (Good):         {f1_good:.4f}  (expected 0.5536)")
    print(f"\nConfusion matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # Tolerance check vs. expected notebook values
    expected = {
        "accuracy": 0.6241,
        "roc_auc": 0.6313,
        "precision_bad": 0.6265,
        "recall_bad": 0.7324,
        "f1_bad": 0.6753,
        "precision_good": 0.6200,
        "recall_good": 0.5000,
        "f1_good": 0.5536,
    }
    actual = {
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "precision_bad": round(prec_bad, 4),
        "recall_bad": round(rec_bad, 4),
        "f1_bad": round(f1_bad, 4),
        "precision_good": round(prec_good, 4),
        "recall_good": round(rec_good, 4),
        "f1_good": round(f1_good, 4),
    }
    mismatches = [k for k, v in expected.items() if abs(actual[k] - v) > 0.0005]
    if mismatches:
        print(f"\n⚠️  Metric mismatches vs. notebook: {mismatches}")
        print("    This suggests a training environment or seed difference.")
    else:
        print("\n✅ All metrics match notebook outputs within tolerance.")

    # Save test indices and probabilities for interface validation
    test_snapshot = pd.DataFrame(
        {
            "test_row_index": test_idx,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "prob_good": y_prob[:, good_idx],
            "prob_bad": y_prob[:, bad_idx],
        }
    )

    # Save artefacts
    joblib.dump(rf_pipeline, MODELS_DIR / "rf_pipeline.joblib")
    test_snapshot.to_csv(MODELS_DIR / "test_predictions_snapshot.csv", index=False)

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "classes": classes,
        "good_class_index": good_idx,
        "bad_class_index": bad_idx,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": actual,
    }
    with open(MODELS_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Saved:")
    print(f"   models/rf_pipeline.joblib")
    print(f"   models/test_predictions_snapshot.csv")
    print(f"   models/training_metadata.json")


if __name__ == "__main__":
    main()
