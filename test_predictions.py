"""
test_predictions.py
===================
Validates that predict_applicant.py produces results identical to the
notebook-trained pipeline for real test-set records.

Tests performed:
  1. Metric reproduction: all 8 key metrics match within 0.0005 tolerance.
  2. Per-record agreement: predicted class and probability agree between
     notebook snapshot and the prediction interface for a sample of test records.
  3. Derived-feature consistency: engineered features produced by
     engineer_features() match those computed directly from the raw inputs.
  4. Input validation: appropriate errors for missing / out-of-range inputs.

Usage:
    python test_predictions.py

Requirements:
    - Run  python save_model.py  first.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap — add project root to path so we can import predict_applicant
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from predict_applicant import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    engineer_features,
    load_model,
    predict,
)

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "AssessmentDataScientist.xlsx"
MATCHED_CSV = PROJECT_ROOT / "outputs" / "matched_analysis_data.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "[PASS]"
FAIL = "[FAIL]"


def check(condition, msg_pass, msg_fail=""):
    if condition:
        print(f"  {PASS} {msg_pass}")
        return True
    else:
        msg = msg_fail or msg_pass
        print(f"  {FAIL} {msg}")
        return False


# ---------------------------------------------------------------------------
# Test 1 — Metric reproduction
# ---------------------------------------------------------------------------
def test_metric_reproduction(pipeline):
    print("\nTest 1: Metric Reproduction")
    print("-" * 50)

    # Load training metadata saved by save_model.py
    meta_path = MODELS_DIR / "training_metadata.json"
    if not meta_path.exists():
        print(f"  {FAIL} training_metadata.json not found. Run save_model.py first.")
        return False

    with open(meta_path) as f:
        meta = json.load(f)

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

    all_ok = True
    for k, exp in expected.items():
        actual = meta["metrics"].get(k)
        if actual is None:
            check(False, "", f"Metric '{k}' missing from training_metadata.json")
            all_ok = False
        else:
            ok = abs(actual - exp) <= 0.0005
            check(
                ok,
                f"{k}: {actual:.4f} == {exp:.4f}",
                f"{k}: {actual:.4f} != {exp:.4f} (expected {exp:.4f})",
            )
            all_ok = all_ok and ok
    return all_ok


# ---------------------------------------------------------------------------
# Test 2 — Per-record agreement between notebook snapshot & interface
# ---------------------------------------------------------------------------
def test_per_record_agreement(pipeline):
    print("\nTest 2: Per-Record Agreement with Notebook Snapshot")
    print("-" * 50)

    snapshot_path = MODELS_DIR / "test_predictions_snapshot.csv"
    if not snapshot_path.exists():
        print(f"  {FAIL} test_predictions_snapshot.csv not found. Run save_model.py first.")
        return False

    snapshot = pd.read_csv(snapshot_path)
    print(f"  Snapshot contains {len(snapshot)} test-set records.")

    # Load the matched analysis data (same CSV produced by the notebook)
    if not MATCHED_CSV.exists():
        print(f"  {FAIL} outputs/matched_analysis_data.csv not found.")
        return False

    matched = pd.read_csv(MATCHED_CSV)
    matched_indexed = matched.reset_index(drop=True)

    # Sample up to 10 test records for agreement check
    n_sample = min(10, len(snapshot))
    sample = snapshot.sample(n=n_sample, random_state=0).reset_index(drop=True)

    all_ok = True
    for _, row in sample.iterrows():
        orig_idx = int(row["test_row_index"])

        # Recover raw applicant record
        if orig_idx >= len(matched_indexed):
            check(False, "", f"Row index {orig_idx} out of bounds in matched_analysis_data.csv")
            all_ok = False
            continue

        applicant_row = matched_indexed.iloc[orig_idx]

        # Build raw inputs (same raw columns as the notebook).
        # NaN categorical values must be passed as None so the sklearn
        # imputer handles them identically to the notebook pipeline.
        def _cat(v):
            """Return np.nan for NaN/missing, otherwise string."""
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return np.nan
            except (TypeError, ValueError):
                pass
            return str(v)

        raw = {
            "amount_requested": float(applicant_row["amount_requested"]),
            "loan_duration": float(applicant_row["loan_duration"]),
            "payment_amount": float(applicant_row["payment_amount"]),
            "monthly_income_amount": float(applicant_row["monthly_income_amount"]),
            "monthly_rent_amount": float(applicant_row["monthly_rent_amount"]),
            "num_payments": float(applicant_row["num_payments"]),
            "raw_FICO_telecom": float(applicant_row["raw_FICO_telecom"]),
            "raw_FICO_retail": float(applicant_row["raw_FICO_retail"]),
            "raw_FICO_bank_card": float(applicant_row["raw_FICO_bank_card"]),
            "raw_FICO_money": float(applicant_row["raw_FICO_money"]),
            "raw_l2c_score": float(applicant_row["raw_l2c_score"]),
            "payment_frequency": _cat(applicant_row["payment_frequency"]),
            "residence_rent_or_own": bool(applicant_row["residence_rent_or_own"]),
            "bank_account_direct_deposit": bool(applicant_row["bank_account_direct_deposit"]),
            "email_duration": _cat(applicant_row["email_duration"]),
            "residence_duration": _cat(applicant_row["residence_duration"]),
            "bank_account_duration": _cat(applicant_row["bank_account_duration"]),
            "home_phone_type": _cat(applicant_row["home_phone_type"]),
        }

        input_df = engineer_features(raw)
        result = predict(pipeline, input_df)

        snapshot_pred = int(row["y_pred"])
        snapshot_prob_good = float(row["prob_good"])
        snapshot_prob_bad = float(row["prob_bad"])

        class_ok = result["predicted_class"] == snapshot_pred
        prob_good_ok = abs(result["prob_good"] - snapshot_prob_good) < 1e-6
        prob_bad_ok = abs(result["prob_bad"] - snapshot_prob_bad) < 1e-6

        ok = class_ok and prob_good_ok and prob_bad_ok
        check(
            ok,
            f"Row {orig_idx}: class={result['predicted_class']}, "
            f"P(Good)={result['prob_good']:.4f}, P(Bad)={result['prob_bad']:.4f}",
            f"Row {orig_idx}: class mismatch "
            f"(interface={result['predicted_class']} vs snapshot={snapshot_pred}) "
            f"OR prob mismatch "
            f"(P(Bad)={result['prob_bad']:.6f} vs snapshot={snapshot_prob_bad:.6f})",
        )
        all_ok = all_ok and ok

    return all_ok


# ---------------------------------------------------------------------------
# Test 3 — Derived-feature consistency
# ---------------------------------------------------------------------------
def test_derived_feature_consistency():
    print("\nTest 3: Derived-Feature Consistency")
    print("-" * 50)
    all_ok = True

    raw = {
        "amount_requested": 500.0,
        "loan_duration": 6.0,
        "payment_amount": 100.0,
        "monthly_income_amount": 2000.0,
        "monthly_rent_amount": 600.0,
        "num_payments": 10.0,
        "raw_FICO_telecom": 580.0,
        "raw_FICO_retail": 600.0,
        "raw_FICO_bank_card": 650.0,
        "raw_FICO_money": 590.0,
        "raw_l2c_score": 600.0,
        "payment_frequency": "Monthly",
        "residence_rent_or_own": False,
        "bank_account_direct_deposit": True,
        "email_duration": "1 year or more",
        "residence_duration": "1-2 years",
        "bank_account_duration": "3+ years",
        "home_phone_type": "Mobile",
    }

    df = engineer_features(raw)

    expected_pto = 100.0 / (2000.0 + 1.0)
    expected_rto = 600.0 / (2000.0 + 1.0)
    expected_trr = (10.0 * 100.0) / (500.0 + 1.0)
    expected_mfico = np.mean([580.0, 600.0, 650.0, 590.0])

    all_ok &= check(
        abs(df["payment_to_income"].iloc[0] - expected_pto) < 1e-8,
        f"payment_to_income = {df['payment_to_income'].iloc[0]:.6f} (expected {expected_pto:.6f})",
    )
    all_ok &= check(
        abs(df["rent_to_income"].iloc[0] - expected_rto) < 1e-8,
        f"rent_to_income = {df['rent_to_income'].iloc[0]:.6f} (expected {expected_rto:.6f})",
    )
    all_ok &= check(
        abs(df["total_repayment_ratio"].iloc[0] - expected_trr) < 1e-8,
        f"total_repayment_ratio = {df['total_repayment_ratio'].iloc[0]:.6f} (expected {expected_trr:.6f})",
    )
    all_ok &= check(
        abs(df["mean_fico"].iloc[0] - expected_mfico) < 1e-8,
        f"mean_fico = {df['mean_fico'].iloc[0]:.4f} (expected {expected_mfico:.4f})",
    )
    all_ok &= check(
        list(df.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "Column order matches NUMERIC_FEATURES + CATEGORICAL_FEATURES",
        "Column order mismatch",
    )
    return all_ok


# ---------------------------------------------------------------------------
# Test 4 — Input validation (edge cases)
# ---------------------------------------------------------------------------
def test_input_validation(pipeline):
    print("\nTest 4: Edge Cases & Input Validation")
    print("-" * 50)
    all_ok = True

    # 4a: Prediction runs without error for minimum valid inputs
    raw_min = {
        "amount_requested": 250.0,
        "loan_duration": 2.0,
        "payment_amount": 35.0,
        "monthly_income_amount": 300.0,
        "monthly_rent_amount": 0.0,
        "num_payments": 3.0,
        "raw_FICO_telecom": 222.0,
        "raw_FICO_retail": 222.0,
        "raw_FICO_bank_card": 222.0,
        "raw_FICO_money": 222.0,
        "raw_l2c_score": 50.0,
        "payment_frequency": "Weekly",
        "residence_rent_or_own": False,
        "bank_account_direct_deposit": False,
        "email_duration": "3 months or less",
        "residence_duration": "6 months or less",
        "bank_account_duration": "6 months or less",
        "home_phone_type": "Home",
    }
    try:
        df = engineer_features(raw_min)
        result = predict(pipeline, df)
        ok = result["predicted_class"] in (0, 1)
        ok &= 0.0 <= result["prob_good"] <= 1.0
        ok &= 0.0 <= result["prob_bad"] <= 1.0
        ok &= abs(result["prob_good"] + result["prob_bad"] - 1.0) < 1e-6
        check(ok, "Min-value applicant produces valid prediction")
        all_ok &= ok
    except Exception as e:
        check(False, "", f"Min-value applicant raised exception: {e}")
        all_ok = False

    # 4b: Probabilities sum to 1.0
    raw_max = {
        "amount_requested": 800.0,
        "loan_duration": 8.0,
        "payment_amount": 455.0,
        "monthly_income_amount": 19392.0,
        "monthly_rent_amount": 3018.0,
        "num_payments": 34.0,
        "raw_FICO_telecom": 712.0,
        "raw_FICO_retail": 756.0,
        "raw_FICO_bank_card": 797.0,
        "raw_FICO_money": 662.0,
        "raw_l2c_score": 808.0,
        "payment_frequency": "Monthly",
        "residence_rent_or_own": True,
        "bank_account_direct_deposit": True,
        "email_duration": "1 year or more",
        "residence_duration": "3+ years",
        "bank_account_duration": "3+ years",
        "home_phone_type": "Home",
    }
    try:
        df = engineer_features(raw_max)
        result = predict(pipeline, df)
        prob_sum_ok = abs(result["prob_good"] + result["prob_bad"] - 1.0) < 1e-6
        check(prob_sum_ok, f"Probabilities sum to 1.0 (P(Good)={result['prob_good']:.4f}, P(Bad)={result['prob_bad']:.4f})")
        all_ok &= prob_sum_ok
    except Exception as e:
        check(False, "", f"Max-value applicant raised exception: {e}")
        all_ok = False

    # 4c: Missing required key raises KeyError
    raw_incomplete = raw_min.copy()
    del raw_incomplete["raw_FICO_retail"]
    try:
        engineer_features(raw_incomplete)
        check(False, "", "Incomplete inputs should raise KeyError but did not")
        all_ok = False
    except KeyError:
        check(True, "Missing key raises KeyError as expected")
    except Exception as e:
        check(False, "", f"Unexpected exception for missing key: {type(e).__name__}: {e}")
        all_ok = False

    # 4d: Zero income does not divide by zero (denominator +1.0 guard)
    raw_zero_income = raw_min.copy()
    raw_zero_income["monthly_income_amount"] = 0.0
    try:
        df = engineer_features(raw_zero_income)
        result = predict(pipeline, df)
        ok = not np.isnan(result["prob_bad"]) and not np.isinf(result["prob_bad"])
        check(ok, "Zero income does not produce NaN or Inf probability")
        all_ok &= ok
    except Exception as e:
        check(False, "", f"Zero income raised exception: {e}")
        all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Test 5 — Class ordering consistency
# ---------------------------------------------------------------------------
def test_class_ordering(pipeline):
    print("\nTest 5: Class Ordering Consistency")
    print("-" * 50)
    all_ok = True

    classes = pipeline.named_steps["model"].classes_.tolist()
    all_ok &= check(
        classes == [0, 1],
        f"classes_ = {classes} (0=Good, 1=Bad — matches save_model.py assertion)",
        f"Unexpected classes_: {classes}",
    )

    # Sanity: a very low-risk synthetic applicant (high FICO, low debt burden)
    raw_low_risk = {
        "amount_requested": 300.0,
        "loan_duration": 3.0,
        "payment_amount": 50.0,
        "monthly_income_amount": 5000.0,
        "monthly_rent_amount": 200.0,
        "num_payments": 6.0,
        "raw_FICO_telecom": 700.0,
        "raw_FICO_retail": 710.0,
        "raw_FICO_bank_card": 720.0,
        "raw_FICO_money": 690.0,
        "raw_l2c_score": 700.0,
        "payment_frequency": "Monthly",
        "residence_rent_or_own": True,
        "bank_account_direct_deposit": True,
        "email_duration": "1 year or more",
        "residence_duration": "3+ years",
        "bank_account_duration": "3+ years",
        "home_phone_type": "Home",
    }
    df = engineer_features(raw_low_risk)
    result = predict(pipeline, df)
    print(
        f"  INFO: Low-risk synthetic applicant -> "
        f"class={result['predicted_class']}, "
        f"P(Good)={result['prob_good']:.3f}, P(Bad)={result['prob_bad']:.3f}"
    )
    all_ok &= check(
        result["prob_good"] + result["prob_bad"] > 0.999,
        "Probabilities sum to ~1.0 for low-risk profile",
    )
    return all_ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  TapLend — Prediction Interface Test Suite")
    print("=" * 60)

    if not (MODELS_DIR / "rf_pipeline.joblib").exists():
        print("\nERROR: Model file not found. Run:  python save_model.py")
        sys.exit(1)

    pipeline = load_model()

    results = {
        "Metric Reproduction": test_metric_reproduction(pipeline),
        "Per-Record Agreement": test_per_record_agreement(pipeline),
        "Derived Feature Consistency": test_derived_feature_consistency(),
        "Edge Cases & Validation": test_input_validation(pipeline),
        "Class Ordering": test_class_ordering(pipeline),
    }

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(results.values())
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
    print(f"\n  {passed}/{total} test suites passed.")
    if passed < total:
        sys.exit(1)
    else:
        print("  All tests passed.")


if __name__ == "__main__":
    main()
