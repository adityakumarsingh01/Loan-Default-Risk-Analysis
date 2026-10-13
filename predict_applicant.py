"""
predict_applicant.py
====================
Interactive command-line prediction interface for the TapLend credit risk model.

The model was trained on historical TapLend India loan data (Oct 2010 - Apr 2011).
This interface is an EDUCATIONAL PROTOTYPE only — it must NOT be used as a
final lending decision tool.

Usage:
    python predict_applicant.py                 # interactive mode
    python predict_applicant.py --example       # run a worked example
    python predict_applicant.py --help          # show usage

Requirements:
    - Run  python save_model.py  first to generate models/rf_pipeline.joblib
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (must mirror save_model.py / notebook exactly)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "rf_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"

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
    "payment_to_income",   # derived
    "rent_to_income",       # derived
    "total_repayment_ratio",  # derived
    "mean_fico",            # derived
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

VALID_VALUES = {
    "payment_frequency": ["Bi-weekly", "Monthly", "Semi-monthly", "Weekly"],
    "residence_rent_or_own": [True, False],  # True = own, False = rent
    "bank_account_direct_deposit": [True, False],
    "email_duration": ["3 months or less", "4-12 months", "1 year or more"],
    "residence_duration": ["6 months or less", "7-12 months", "1-2 years", "3+ years"],
    "bank_account_duration": ["6 months or less", "7-12 months", "1-2 years", "3+ years"],
    "home_phone_type": ["Home", "Mobile", "Work"],
}

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model():
    """Load the saved Random Forest pipeline. Raises a clear error if missing."""
    if not MODEL_PATH.exists():
        print("ERROR: Trained model not found at models/rf_pipeline.joblib")
        print("       Please run:  python save_model.py")
        sys.exit(1)
    pipeline = joblib.load(MODEL_PATH)

    # Verify class ordering
    classes = pipeline.named_steps["model"].classes_.tolist()
    assert classes == [0, 1], f"Unexpected class ordering: {classes}"
    return pipeline


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def prompt_float(label, min_val=None, max_val=None):
    while True:
        raw = input(f"  {label}: ").strip()
        try:
            val = float(raw)
        except ValueError:
            print(f"    -> Please enter a number.")
            continue
        if min_val is not None and val < min_val:
            print(f"    -> Value must be >= {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"    -> Value must be <= {max_val}.")
            continue
        return val


def prompt_choice(label, choices):
    choices_lower = [str(c).lower() for c in choices]
    display = " / ".join(str(c) for c in choices)
    while True:
        raw = input(f"  {label} [{display}]: ").strip()
        if raw.lower() in choices_lower:
            # Return the original-cased value
            return choices[choices_lower.index(raw.lower())]
        print(f"    -> Please enter one of: {display}")


def prompt_bool(label):
    while True:
        raw = input(f"  {label} [yes / no]: ").strip().lower()
        if raw in ("yes", "y", "true", "1"):
            return True
        if raw in ("no", "n", "false", "0"):
            return False
        print("    -> Please enter yes or no.")


# ---------------------------------------------------------------------------
# Feature engineering (mirrors the notebook exactly)
# ---------------------------------------------------------------------------
def engineer_features(raw: dict) -> pd.DataFrame:
    """
    Compute derived features from raw applicant inputs and return a single-row
    DataFrame in the exact column order expected by the trained pipeline.
    """
    payment_amount = raw["payment_amount"]
    monthly_income = raw["monthly_income_amount"]
    monthly_rent = raw["monthly_rent_amount"]
    num_payments = raw["num_payments"]
    amount_requested = raw["amount_requested"]

    payment_to_income = payment_amount / (monthly_income + 1.0)
    rent_to_income = monthly_rent / (monthly_income + 1.0)
    total_repayment_ratio = (num_payments * payment_amount) / (amount_requested + 1.0)
    mean_fico = np.mean(
        [
            raw["raw_FICO_telecom"],
            raw["raw_FICO_retail"],
            raw["raw_FICO_bank_card"],
            raw["raw_FICO_money"],
        ]
    )

    row = {
        # Raw numeric
        "amount_requested": amount_requested,
        "loan_duration": raw["loan_duration"],
        "payment_amount": payment_amount,
        "monthly_income_amount": monthly_income,
        "monthly_rent_amount": monthly_rent,
        "num_payments": num_payments,
        "raw_FICO_telecom": raw["raw_FICO_telecom"],
        "raw_FICO_retail": raw["raw_FICO_retail"],
        "raw_FICO_bank_card": raw["raw_FICO_bank_card"],
        "raw_FICO_money": raw["raw_FICO_money"],
        "raw_l2c_score": raw["raw_l2c_score"],
        # Engineered numeric
        "payment_to_income": payment_to_income,
        "rent_to_income": rent_to_income,
        "total_repayment_ratio": total_repayment_ratio,
        "mean_fico": mean_fico,
        # Categorical
        "payment_frequency": raw["payment_frequency"],
        "residence_rent_or_own": raw["residence_rent_or_own"],
        "bank_account_direct_deposit": raw["bank_account_direct_deposit"],
        "email_duration": raw["email_duration"],
        "residence_duration": raw["residence_duration"],
        "bank_account_duration": raw["bank_account_duration"],
        "home_phone_type": raw["home_phone_type"],
    }

    return pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(pipeline, input_df: pd.DataFrame) -> dict:
    """
    Run prediction and return a results dict.
    Classes: 0 = Good (Repaid), 1 = Bad (Default)
    predict_proba columns are ordered by classes_ = [0, 1].
    """
    proba = pipeline.predict_proba(input_df)[0]  # shape (2,)
    pred_class = pipeline.predict(input_df)[0]

    # Class indices confirmed via save_model.py validation
    prob_good = float(proba[0])  # index 0 = class 0 (Good)
    prob_bad = float(proba[1])   # index 1 = class 1 (Bad)
    label = "Bad (High Default Risk)" if pred_class == 1 else "Good (Low Default Risk)"

    return {
        "predicted_class": int(pred_class),
        "predicted_label": label,
        "prob_good": prob_good,
        "prob_bad": prob_bad,
    }


def display_result(result: dict):
    print()
    print("=" * 60)
    print("  CREDIT RISK ASSESSMENT RESULT")
    print("=" * 60)
    print(f"  Predicted Class  :  {result['predicted_label']}")
    print(f"  P(Good / Repaid) :  {result['prob_good']:.1%}")
    print(f"  P(Bad / Default) :  {result['prob_bad']:.1%}")
    print()
    print("  IMPORTANT LIMITATIONS")
    print("  ─────────────────────────────────────────────────────")
    print("  1. This is an EDUCATIONAL PROTOTYPE — not a lending")
    print("     decision system.")
    print("  2. The model was trained on 524 historical TapLend")
    print("     India loans (Oct 2010 – Apr 2011). It cannot")
    print("     generalise to different populations or time periods.")
    print("  3. Probabilities are statistical estimates, not")
    print("     guarantees of individual repayment behaviour.")
    print("  4. Approved-only training data means the model does")
    print("     not account for rejected applicant risk.")
    print("  5. Age was excluded from the model to reduce fair-")
    print("     lending risk, but proxy bias may still exist.")
    print("  6. DO NOT deploy this model in automated underwriting.")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Interactive collection of applicant inputs
# ---------------------------------------------------------------------------
def collect_applicant_inputs() -> dict:
    print()
    print("=" * 60)
    print("  TapLend — Applicant Credit Risk Assessment")
    print("  (Educational prototype — not a lending decision)")
    print("=" * 60)
    print()
    print("Enter applicant details. Press Ctrl+C to cancel.")
    print()
    print("--- Loan Terms ---")
    amount_requested = prompt_float("Requested loan amount ($)", 250, 800)
    loan_duration = prompt_float("Loan duration (months)", 2, 8)
    payment_amount = prompt_float("Proposed periodic payment amount ($)", 1)
    num_payments = prompt_float("Number of payments", 1, 40)
    payment_frequency = prompt_choice(
        "Payment frequency",
        VALID_VALUES["payment_frequency"],
    )

    print()
    print("--- Financial Profile ---")
    monthly_income = prompt_float("Self-reported gross monthly income ($)", 1)
    monthly_rent = prompt_float("Monthly rent or mortgage payment ($)", 0)
    residence_rent_or_own = prompt_bool("Does applicant own (not rent) their home?")

    print()
    print("--- Credit Bureau Scores (222–850 range) ---")
    fico_telecom = prompt_float("FICO Telecom score", 222, 850)
    fico_retail = prompt_float("FICO Retail score", 222, 850)
    fico_bankcard = prompt_float("FICO Bank Card score", 222, 850)
    fico_money = prompt_float("FICO Money score", 222, 850)
    l2c_score = prompt_float("L2C (LexisNexis) score", 50, 850)

    print()
    print("--- Account History ---")
    bank_direct_deposit = prompt_bool("Bank account has direct deposit?")
    email_duration = prompt_choice(
        "How long has applicant had their email address?",
        VALID_VALUES["email_duration"],
    )
    residence_duration = prompt_choice(
        "How long has applicant lived at current address?",
        VALID_VALUES["residence_duration"],
    )
    bank_account_duration = prompt_choice(
        "How long has applicant had their bank account?",
        VALID_VALUES["bank_account_duration"],
    )
    home_phone_type = prompt_choice(
        "Primary phone type",
        VALID_VALUES["home_phone_type"],
    )

    return {
        "amount_requested": amount_requested,
        "loan_duration": loan_duration,
        "payment_amount": payment_amount,
        "monthly_income_amount": monthly_income,
        "monthly_rent_amount": monthly_rent,
        "num_payments": num_payments,
        "raw_FICO_telecom": fico_telecom,
        "raw_FICO_retail": fico_retail,
        "raw_FICO_bank_card": fico_bankcard,
        "raw_FICO_money": fico_money,
        "raw_l2c_score": l2c_score,
        "payment_frequency": payment_frequency,
        "residence_rent_or_own": residence_rent_or_own,
        "bank_account_direct_deposit": bank_direct_deposit,
        "email_duration": email_duration,
        "residence_duration": residence_duration,
        "bank_account_duration": bank_account_duration,
        "home_phone_type": home_phone_type,
    }


# ---------------------------------------------------------------------------
# Worked example (a real record from the training data)
# ---------------------------------------------------------------------------
EXAMPLE_APPLICANT = {
    "amount_requested": 500.0,
    "loan_duration": 6.0,
    "payment_amount": 87.08,
    "monthly_income_amount": 2000.0,
    "monthly_rent_amount": 600.0,
    "num_payments": 10.0,
    "raw_FICO_telecom": 600.0,
    "raw_FICO_retail": 620.0,
    "raw_FICO_bank_card": 680.0,
    "raw_FICO_money": 610.0,
    "raw_l2c_score": 600.0,
    "payment_frequency": "Bi-weekly",
    "residence_rent_or_own": False,
    "bank_account_direct_deposit": True,
    "email_duration": "1 year or more",
    "residence_duration": "1-2 years",
    "bank_account_duration": "3+ years",
    "home_phone_type": "Mobile",
}


def run_example(pipeline):
    print()
    print("=" * 60)
    print("  WORKED EXAMPLE — Hypothetical Applicant Profile")
    print("=" * 60)
    print("  Amount requested:    $500")
    print("  Loan duration:       6 months, 10 payments (bi-weekly)")
    print("  Payment amount:      $87.08")
    print("  Monthly income:      $2,000  |  Monthly rent: $600")
    print("  FICO scores:         Telecom=600, Retail=620,")
    print("                       Bank Card=680, Money=610")
    print("  L2C score:           600")
    print("  Payment frequency:   Bi-weekly")
    print("  Home ownership:      Rents")
    print("  Direct deposit:      Yes")
    print("  Email history:       1 year or more")
    print("  Residence duration:  1-2 years")
    print("  Bank account age:    3+ years")
    print("  Phone type:          Mobile")

    input_df = engineer_features(EXAMPLE_APPLICANT)
    result = predict(pipeline, input_df)
    display_result(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TapLend credit risk prediction interface (educational prototype)"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run a worked example instead of interactive input",
    )
    args = parser.parse_args()

    pipeline = load_model()

    if args.example:
        run_example(pipeline)
        return

    try:
        raw = collect_applicant_inputs()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return

    input_df = engineer_features(raw)
    result = predict(pipeline, input_df)
    display_result(result)


if __name__ == "__main__":
    main()
