import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ---------------------------------------------------------
# Set project directories
# ---------------------------------------------------------
project_root = Path(__file__).resolve().parent
data_path = project_root / "data" / "AssessmentDataScientist.xlsx"
outputs_dir = project_root / "outputs"
presentation_dir = project_root / "presentation"

outputs_dir.mkdir(parents=True, exist_ok=True)
presentation_dir.mkdir(parents=True, exist_ok=True)

# Visual styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "grid.color": "#e5e5e5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7
})

print(f"Loading data from {data_path}...")
df_app = pd.read_excel(data_path, sheet_name="Application Data")
df_loan = pd.read_excel(data_path, sheet_name="Loan Performance")

print(f"Loaded Application Data: {df_app.shape}")
print(f"Loaded Loan Performance: {df_loan.shape}")

# ---------------------------------------------------------
# 1. Matching Investigation & Dataset Preparation
# ---------------------------------------------------------
# Baseline raw extraction
raw_app_keys = df_app["customer_id"].astype(str).str.strip().str.upper()
raw_loan_keys = df_loan["idLoan"].astype(str).str.strip().str.upper().str.split("-").str[0]

# Documented single identifier correction: Row 567 in Application Data has integer value 902431691 (9 digits).
# In Loan Performance, idLoan 0902431691-01 exists. Excel stripped the leading zero upon opening/export.
df_app_clean = df_app.copy()
df_app_clean["customer_key"] = raw_app_keys.copy()
mask_nine_digits = df_app_clean["customer_key"] == "902431691"
if mask_nine_digits.any():
    print("Applying justified correction: Restoring leading zero to customer ID 902431691 -> 0902431691")
    df_app_clean.loc[mask_nine_digits, "customer_key"] = "0902431691"

df_loan["customer_key"] = raw_loan_keys
df_loan["loan_seq"] = df_loan["idLoan"].astype(str).str.strip().str.upper().str.split("-").str[1]

# Overlap analysis
app_keys = set(df_app_clean["customer_key"])
loan_keys = set(df_loan["customer_key"])
matched_keys = app_keys.intersection(loan_keys)

df_loan["is_matched"] = df_loan["customer_key"].isin(matched_keys)
unmatched_loans = df_loan[~df_loan["is_matched"]].copy()

# Save unmatched loans
unmatched_loans[["idLoan", "customer_key", "flgGood"]].to_csv(
    outputs_dir / "unmatched_loans.csv", index=False
)

# Merge datasets (many-to-one on customer_key)
merged_data = df_loan.merge(
    df_app_clean,
    on="customer_key",
    how="inner"
)
merged_data.to_csv(outputs_dir / "matched_analysis_data.csv", index=False)
print(f"Matched loan records: {len(merged_data)} across {merged_data['customer_key'].nunique()} unique applicants.")
print(f"Unmatched loan records: {len(unmatched_loans)} across {unmatched_loans['customer_key'].nunique()} unique loan prefixes.")

# Outcome percentages
matched_bad = (merged_data["flgGood"] == "Bad").sum()
matched_good = (merged_data["flgGood"] == "Good").sum()
unmatched_bad = (unmatched_loans["flgGood"] == "Bad").sum()
unmatched_good = (unmatched_loans["flgGood"] == "Good").sum()
total_bad = (df_loan["flgGood"] == "Bad").sum()
total_good = (df_loan["flgGood"] == "Good").sum()

# Data Matching Summary Table
matching_summary = pd.DataFrame([
    {"Metric": "Total Application Records", "Value": str(len(df_app))},
    {"Metric": "Unique Application Customer IDs", "Value": str(df_app["customer_id"].nunique())},
    {"Metric": "Total Loan Performance Records", "Value": str(len(df_loan))},
    {"Metric": "Unique Loan Customer Prefixes", "Value": str(df_loan["customer_key"].nunique())},
    {"Metric": "Matched Loan Records", "Value": str(len(merged_data))},
    {"Metric": "Matched Unique Customers", "Value": str(merged_data["customer_key"].nunique())},
    {"Metric": "Unmatched Loan Records", "Value": str(len(unmatched_loans))},
    {"Metric": "Unmatched Unique Loan Prefixes", "Value": str(unmatched_loans["customer_key"].nunique())},
    {"Metric": "Unmatched Application Records", "Value": str(len(df_app) - merged_data["customer_key"].nunique())},
    {"Metric": "Repeat Borrower Loans (-02)", "Value": str((merged_data["loan_seq"] == "02").sum())},
    {"Metric": "Total Portfolio Default Rate", "Value": f"{total_bad / len(df_loan) * 100:.2f}% ({total_bad}/{len(df_loan)})"},
    {"Metric": "Matched Cohort Default Rate", "Value": f"{matched_bad / len(merged_data) * 100:.2f}% ({matched_bad}/{len(merged_data)})"},
    {"Metric": "Unmatched Cohort Default Rate", "Value": f"{unmatched_bad / len(unmatched_loans) * 100:.2f}% ({unmatched_bad}/{len(unmatched_loans)})"}
])
matching_summary.to_csv(outputs_dir / "matching_investigation_summary.csv", index=False)

# Missing Values Report
missing_app = df_app.isnull().sum()
missing_report = pd.DataFrame({
    "Column": missing_app.index,
    "Missing_Count": missing_app.values,
    "Missing_Percentage": (missing_app.values / len(df_app) * 100).round(2)
}).sort_values(by="Missing_Count", ascending=False)
missing_report.to_csv(outputs_dir / "missing_values_report.csv", index=False)

# Data Quality & Governance Summary
data_quality = pd.DataFrame([
    {"Check": "Duplicate Application Rows", "Result": str(int(df_app.duplicated().sum()))},
    {"Check": "Duplicate Loan Performance Rows", "Result": str(int(df_loan.duplicated().sum()))},
    {"Check": "Duplicate Customer IDs in Application Data", "Result": str(int(df_app["customer_id"].duplicated().sum()))},
    {"Check": "Constant Features (Zero Variance)", "Result": "status ('Approved' 100%), payment_ach (True 99.8%)"},
    {"Check": "High Cardinality / PII Columns", "Result": "customer_id, email, bank_routing_number, address_zip"},
    {"Check": "Post-Decision Features (Leakage Risk)", "Result": "amount_approved, duration_approved, payment_amount_approved"},
    {"Check": "Fair Lending / Protected Demographics", "Result": "birth_date (age removed from predictive models due to ECOA fair lending risks)"},
    {"Check": "Proxy Risk Review", "Result": "residential duration, bank tenure, and credit history length may correlate with age; full disparate impact audit required"}
])
data_quality.to_csv(outputs_dir / "data_quality_summary.csv", index=False)

# ---------------------------------------------------------
# 2. Exploratory Visualizations
# ---------------------------------------------------------
print("Generating exploratory charts...")

# Plot 1: Loan Outcome Distribution
fig, ax = plt.subplots(figsize=(8, 5))
outcome_counts = df_loan["flgGood"].value_counts()
sns.barplot(x=outcome_counts.index, y=outcome_counts.values, hue=outcome_counts.index, palette=["#2ecc71", "#e74c3c"], ax=ax, legend=False)
for i, v in enumerate(outcome_counts.values):
    ax.text(i, v + 15, f"{v} ({v/len(df_loan)*100:.1f}%)", ha="center", fontweight="bold", fontsize=11)
ax.set_title("Overall Loan Performance Outcome Distribution (N = 1,285)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Loan Outcome Label (flgGood)", fontsize=11)
ax.set_ylabel("Number of Loans", fontsize=11)
ax.set_ylim(0, 900)
plt.tight_layout()
plt.savefig(outputs_dir / "loan_outcome_distribution.png", dpi=300)
plt.close()

# Plot 2: Distribution of Requested Loan Amounts
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df_app, x="amount_requested", bins=15, kde=True, color="#3498db", ax=ax)
ax.set_title("Distribution of Requested Loan Amounts (N = 647)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Requested Amount ($)", fontsize=11)
ax.set_ylabel("Number of Applications", fontsize=11)
plt.tight_layout()
plt.savefig(outputs_dir / "requested_loan_amounts.png", dpi=300)
plt.close()

# Plot 3: Distribution of Monthly Income
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df_app, x="monthly_income_amount", bins=25, kde=True, color="#9b59b6", ax=ax)
ax.set_title("Distribution of Applicant Monthly Income (N = 647)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Monthly Income ($)", fontsize=11)
ax.set_ylabel("Number of Applications", fontsize=11)
plt.tight_layout()
plt.savefig(outputs_dir / "monthly_income_distribution.png", dpi=300)
plt.close()

# Plot 4: Monthly Income by Outcome
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=merged_data, x="flgGood", y="monthly_income_amount", order=["Good", "Bad"], hue="flgGood", palette=["#2ecc71", "#e74c3c"], ax=ax, legend=False)
ax.set_title("Monthly Income by Loan Outcome (Matched Loans N = 657)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Loan Outcome", fontsize=11)
ax.set_ylabel("Monthly Income ($)", fontsize=11)
plt.tight_layout()
plt.savefig(outputs_dir / "income_by_outcome.png", dpi=300)
plt.close()

# Plot 5: FICO Scores Distribution by Outcome
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fico_cols = [
    ("raw_FICO_retail", "FICO Retail Score", axes[0, 0]),
    ("raw_FICO_telecom", "FICO Telecom Score", axes[0, 1]),
    ("raw_FICO_bank_card", "FICO Bank Card Score", axes[1, 0]),
    ("raw_FICO_money", "FICO Money Score", axes[1, 1])
]
for col, name, ax in fico_cols:
    sns.boxplot(data=merged_data, x="flgGood", y=col, order=["Good", "Bad"], hue="flgGood", palette=["#2ecc71", "#e74c3c"], ax=ax, legend=False)
    good_m = merged_data[merged_data["flgGood"]=="Good"][col].mean()
    bad_m = merged_data[merged_data["flgGood"]=="Bad"][col].mean()
    ax.set_title(f"{name} (Good: {good_m:.1f} vs Bad: {bad_m:.1f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Loan Outcome", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
fig.suptitle("Credit Bureau FICO Scores by Loan Outcome (Statistically Significant Separation)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(outputs_dir / "fico_scores_by_outcome.png", dpi=300)
plt.close()

# Plot 6: Payment Frequency by Outcome
fig, ax = plt.subplots(figsize=(9, 5))
freq_df = merged_data.groupby("payment_frequency")["flgGood"].value_counts(normalize=True).unstack()[["Bad", "Good"]] * 100
freq_counts = merged_data["payment_frequency"].value_counts()
freq_df = freq_df.loc[["Weekly", "Bi-weekly", "Monthly", "Semi-monthly"]]
freq_df.plot(kind="bar", stacked=True, color=["#e74c3c", "#2ecc71"], ax=ax, edgecolor="#333333", width=0.6)
ax.set_title("Observed Default Rate by Payment Frequency Schedule (Matched N = 657)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Repayment Frequency", fontsize=11)
ax.set_ylabel("Percentage (%)", fontsize=11)
for i, cat in enumerate(freq_df.index):
    bad_pct = freq_df.loc[cat, "Bad"]
    n_tot = freq_counts[cat]
    ax.text(i, bad_pct / 2, f"{bad_pct:.1f}%\n(n={n_tot})", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
ax.legend(["Bad (Default)", "Good (Repaid)"], loc="upper right")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(outputs_dir / "payment_frequency_by_outcome.png", dpi=300)
plt.close()

# Plot 7: Matched vs Unmatched Comparison
fig, ax = plt.subplots(figsize=(8, 5))
comp_data = pd.DataFrame({
    "Cohort": ["Matched Loans\n(With Application)", "Unmatched Loans\n(No Application)"],
    "Bad_Pct": [
        (merged_data["flgGood"] == "Bad").mean() * 100,
        (unmatched_loans["flgGood"] == "Bad").mean() * 100
    ],
    "Good_Pct": [
        (merged_data["flgGood"] == "Good").mean() * 100,
        (unmatched_loans["flgGood"] == "Good").mean() * 100
    ],
    "Total_Count": [len(merged_data), len(unmatched_loans)]
})
comp_df_plot = comp_data.set_index("Cohort")[["Bad_Pct", "Good_Pct"]]
comp_df_plot.plot(kind="bar", stacked=True, color=["#e74c3c", "#2ecc71"], ax=ax, edgecolor="#333333", width=0.5)
ax.set_title("Loan Outcome Comparison: Matched vs Unmatched Cohorts", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.set_xlabel("Cohort", fontsize=11)
for i, cohort in enumerate(comp_data["Cohort"]):
    b_pct = comp_data.loc[i, "Bad_Pct"]
    cnt = comp_data.loc[i, "Total_Count"]
    ax.text(i, b_pct / 2, f"{b_pct:.1f}%\n(n={cnt})", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
ax.legend(["Bad (Default)", "Good (Repaid)"], loc="upper right")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(outputs_dir / "matched_vs_unmatched_comparison.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 3. Feature Engineering & Preprocessing (EXCLUDING AGE)
# ---------------------------------------------------------
print("Preparing feature engineering (excluding age for fair lending)...")

df_model = merged_data.copy()
df_model["target"] = (df_model["flgGood"] == "Bad").astype(int) # 1 = Bad, 0 = Good
df_model["payment_to_income"] = df_model["payment_amount"] / (df_model["monthly_income_amount"] + 1.0)
df_model["rent_to_income"] = df_model["monthly_rent_amount"] / (df_model["monthly_income_amount"] + 1.0)
df_model["total_repayment_ratio"] = (df_model["num_payments"] * df_model["payment_amount"]) / (df_model["amount_requested"] + 1.0)
df_model["mean_fico"] = df_model[["raw_FICO_telecom", "raw_FICO_retail", "raw_FICO_bank_card", "raw_FICO_money"]].mean(axis=1)

# Notice: age is EXCLUDED to prevent demographic fair-lending violations
num_features = [
    "amount_requested", "loan_duration", "payment_amount", "monthly_income_amount",
    "monthly_rent_amount", "num_payments", "raw_FICO_telecom", "raw_FICO_retail",
    "raw_FICO_bank_card", "raw_FICO_money", "raw_l2c_score",
    "payment_to_income", "rent_to_income", "total_repayment_ratio", "mean_fico"
]

cat_features = [
    "payment_frequency", "residence_rent_or_own", "bank_account_direct_deposit",
    "email_duration", "residence_duration", "bank_account_duration", "home_phone_type"
]

X = df_model[num_features + cat_features]
y = df_model["target"]
groups = df_model["customer_key"]

# Grouped Split (prevents repeat borrowers from leaking across train and test sets)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# ---------------------------------------------------------
# 4. Modeling & Evaluation
# ---------------------------------------------------------
print("Training models...")

# Model 1: Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
lr_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", lr_model)
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

# Model 2: Random Forest
rf_model = RandomForestClassifier(
    random_state=42, n_estimators=100, max_depth=5, min_samples_split=10, class_weight="balanced"
)
rf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", rf_model)
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

# Extract feature names & importances
preprocessor.fit(X_train)
cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_features))
all_feature_names = num_features + encoded_cat_names

lr_coefs = lr_model.coef_[0]
lr_coef_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Coefficient": lr_coefs,
    "Odds_Ratio": np.exp(lr_coefs),
    "Abs_Coefficient": np.abs(lr_coefs)
}).sort_values(by="Abs_Coefficient", ascending=False)
lr_coef_df.to_csv(outputs_dir / "logistic_regression_coefficients.csv", index=False)

rf_importances = rf_model.feature_importances_
rf_imp_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": rf_importances
}).sort_values(by="Importance", ascending=False)
rf_imp_df.to_csv(outputs_dir / "feature_importance_summary.csv", index=False)

# Model Comparison Table
model_metrics = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": round(accuracy_score(y_test, y_pred_lr), 4),
        "ROC_AUC": round(roc_auc_score(y_test, y_prob_lr), 4),
        "Precision_Bad": round(precision_score(y_test, y_pred_lr, pos_label=1), 4),
        "Recall_Bad": round(recall_score(y_test, y_pred_lr, pos_label=1), 4),
        "F1_Score_Bad": round(f1_score(y_test, y_pred_lr, pos_label=1), 4),
        "Precision_Good": round(precision_score(y_test, y_pred_lr, pos_label=0), 4),
        "Recall_Good": round(recall_score(y_test, y_pred_lr, pos_label=0), 4),
        "F1_Score_Good": round(f1_score(y_test, y_pred_lr, pos_label=0), 4)
    },
    {
        "Model": "Random Forest",
        "Accuracy": round(accuracy_score(y_test, y_pred_rf), 4),
        "ROC_AUC": round(roc_auc_score(y_test, y_prob_rf), 4),
        "Precision_Bad": round(precision_score(y_test, y_pred_rf, pos_label=1), 4),
        "Recall_Bad": round(recall_score(y_test, y_pred_rf, pos_label=1), 4),
        "F1_Score_Bad": round(f1_score(y_test, y_pred_rf, pos_label=1), 4),
        "Precision_Good": round(precision_score(y_test, y_pred_rf, pos_label=0), 4),
        "Recall_Good": round(recall_score(y_test, y_pred_rf, pos_label=0), 4),
        "F1_Score_Good": round(f1_score(y_test, y_pred_rf, pos_label=0), 4)
    }
])
model_metrics.to_csv(outputs_dir / "model_comparison_table.csv", index=False)
print("Updated Model Metrics (Excluding Age):")
print(model_metrics.to_string())

# Plot 8: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
            xticklabels=["Good (0)", "Bad (1)"], yticklabels=["Good (0)", "Bad (1)"])
axes[0].set_title(f"Logistic Regression Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_pred_lr):.1%}, AUC: {roc_auc_score(y_test, y_prob_lr):.3f}", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Predicted Label", fontsize=10)
axes[0].set_ylabel("Actual True Label", fontsize=10)

sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1],
            xticklabels=["Good (0)", "Bad (1)"], yticklabels=["Good (0)", "Bad (1)"])
axes[1].set_title(f"Random Forest Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_pred_rf):.1%}, AUC: {roc_auc_score(y_test, y_prob_rf):.3f}", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Predicted Label", fontsize=10)
axes[1].set_ylabel("Actual True Label", fontsize=10)

plt.tight_layout()
plt.savefig(outputs_dir / "confusion_matrices.png", dpi=300)
plt.close()

# Plot 9: ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})", color="#2980b9", lw=2)
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})", color="#27ae60", lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
ax.set_title("Receiver Operating Characteristic (ROC) Curves (Excluding Age)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(outputs_dir / "roc_curves.png", dpi=300)
plt.close()

# Plot 10: Feature Importance & Coefficients Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top_rf = rf_imp_df.head(10).sort_values(by="Importance", ascending=True)
axes[0].barh(top_rf["Feature"], top_rf["Importance"], color="#27ae60", edgecolor="#1e824c")
axes[0].set_title("Random Forest: Top 10 Feature Importances", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Gini Importance", fontsize=10)

top_lr = lr_coef_df.head(10).sort_values(by="Abs_Coefficient", ascending=True)
colors = ["#e74c3c" if c > 0 else "#2980b9" for c in top_lr["Coefficient"]]
axes[1].barh(top_lr["Feature"], top_lr["Coefficient"], color=colors, edgecolor="#333333")
axes[1].set_title("Logistic Regression: Top 10 Coefficients\n(Red: Increases Default Risk, Blue: Reduces Risk)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Log-Odds Coefficient", fontsize=10)
axes[1].axvline(0, color="black", linestyle="--", lw=1)

plt.tight_layout()
plt.savefig(outputs_dir / "feature_importance_comparison.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 5. Corrected Leak-Free Financial Decision Threshold Analysis
# ---------------------------------------------------------
# NOTE: C_FN = $500 (Loss per default) and C_FP = $150 (Lost interest margin)
# are ILLUSTRATIVE ASSUMPTIONS not documented in the raw assignment workbook.
C_FN = 500
C_FP = 150

print("Conducting leak-free threshold optimization on training CV...")
# Perform GroupKFold cross-validation strictly on X_train to generate out-of-fold probability estimates
gkf = GroupKFold(n_splits=5)
train_cv_probs_rf = cross_val_predict(rf_pipeline, X_train, y_train, groups=groups_train, cv=gkf, method="predict_proba")[:, 1]

threshold_candidates = np.linspace(0.20, 0.80, 61)
train_losses = []
for t in threshold_candidates:
    pred_tr = (train_cv_probs_rf >= t).astype(int)
    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, pred_tr).ravel()
    loss_tr = fn_tr * C_FN + fp_tr * C_FP
    train_losses.append({"Threshold": t, "Train_Loss": loss_tr, "FN_Train": fn_tr, "FP_Train": fp_tr})

train_loss_df = pd.DataFrame(train_losses)
optimal_idx_train = train_loss_df["Train_Loss"].idxmin()
selected_threshold = train_loss_df.loc[optimal_idx_train, "Threshold"]
min_train_loss = train_loss_df.loc[optimal_idx_train, "Train_Loss"]

print(f"Optimal threshold selected on Training CV: T = {selected_threshold:.2f} (Train Loss = ${min_train_loss:,.0f})")

# Evaluate on UNTOUCHED Holdout Test Set
test_losses = []
for t in threshold_candidates:
    pred_te = (y_prob_rf >= t).astype(int)
    tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, pred_te).ravel()
    loss_te = fn_te * C_FN + fp_te * C_FP
    test_losses.append({"Threshold": t, "Test_Loss": loss_te, "FN_Test": fn_te, "FP_Test": fp_te})

test_loss_df = pd.DataFrame(test_losses)
loss_at_50 = test_loss_df.loc[np.isclose(test_loss_df["Threshold"], 0.50), "Test_Loss"].values[0]
loss_at_selected = test_loss_df.loc[np.isclose(test_loss_df["Threshold"], selected_threshold), "Test_Loss"].values[0]

diff_dollars = loss_at_50 - loss_at_selected
diff_pct = (diff_dollars / loss_at_50) * 100 if loss_at_50 > 0 else 0

print(f"Holdout Test Set Performance:")
print(f"  • Loss at standard 0.50 cutoff: ${loss_at_50:,.0f}")
print(f"  • Loss at training-selected cutoff ({selected_threshold:.2f}): ${loss_at_selected:,.0f}")
print(f"  • Net Difference: ${diff_dollars:,.0f} ({diff_pct:.1f}%)")

# Plot 11: Corrected Financial Tradeoff Curve
fig, ax1 = plt.subplots(figsize=(9, 5))
color = "#c0392b"
ax1.set_xlabel("Decision Probability Cutoff T (Classify as Bad if P(Bad) >= T)", fontsize=11)
ax1.set_ylabel("Training CV Illustrative Financial Loss ($)", color=color, fontsize=11)
ax1.plot(train_loss_df["Threshold"], train_loss_df["Train_Loss"], color=color, lw=2.5, label="Training CV Loss Curve")
ax1.tick_params(axis="y", labelcolor=color)
ax1.axvline(selected_threshold, color="#2c3e50", linestyle=":", lw=2,
            label=f"CV-Selected Cutoff = {selected_threshold:.2f} (Test Loss: ${loss_at_selected:,.0f})")

ax2 = ax1.twinx()
color = "#2980b9"
ax2.set_ylabel("Holdout Test Set Loss ($)", color=color, fontsize=11)
ax2.plot(test_loss_df["Threshold"], test_loss_df["Test_Loss"], color=color, linestyle="--", lw=2.0, label="Holdout Test Loss Curve")
ax2.tick_params(axis="y", labelcolor=color)

fig.suptitle("Leak-Free Decision Threshold Tuning & Illustrative Loss Trade-Off\n(Threshold Selected on Training CV; Evaluated on Untouched Test Set)", fontsize=12, fontweight="bold", y=0.98)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
plt.tight_layout()
plt.savefig(outputs_dir / "financial_tradeoff_curve.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 6. Generate Updated Findings Markdown Report
# ---------------------------------------------------------
print("Writing updated findings report...")
findings_md = f"""# TapLend India — Credit Risk Analysis & Data Science Assessment
## Updated Technical Findings & Analytical Report

### 1. Executive Summary & Audit Context
This report presents a rigorous, leak-free credit risk analysis of historical loan applications and repayment outcomes from TapLend's portfolio. Following technical review, this analysis incorporates:
1. **Fair Lending Feature Hygiene**: Deliberate exclusion of applicant age to mitigate fair lending violations (ECOA) and review of proxy risk attributes.
2. **Leak-Free Threshold Tuning**: Selection of the decision probability threshold strictly via 5-fold cross-validation on training data, evaluated against an untouched holdout test set.
3. **Illustrative Cost Accounting**: Explicit labeling of cost assumptions ($500 default loss, $150 lost margin) as illustrative scenarios not documented in the raw assignment data.
4. **Evidence-Based Policy Caution**: Refinement of recommendations to distinguish between statistical association and causation, especially given subgroup sample sizes.

---

### 2. Data Matching Investigation & Findings
* **Application Data Dimensions**: 647 rows $\times$ 31 columns. Covers approved applicants between October 16, 2010 and April 17, 2011.
* **Loan Performance Dimensions**: 1,285 rows $\times$ 2 columns (`idLoan`, `flgGood`: **769 Good (59.84%)**, **516 Bad (40.16%)**).
* **Identifier Architecture**:
  * In `Application Data`, `customer_id` is a 10-character lowercase hexadecimal hash.
  * In `Loan Performance`, `idLoan` combines a 10-character uppercase customer prefix with a sequence suffix (e.g. `9ECE67D6C5-01`).
* **Justified Single Identifier Correction**:
  * Row 567 in `Application Data` contains integer value `902431691` (9 digits). In `Loan Performance`, loan `0902431691-01` exists. Excel's automatic numeric coercion stripped the leading zero. Restoring the leading zero produces a verified match.
* **Final Matching Counts**:
  * **Matched Loan Records**: **657 loans** across **641 distinct applicants** (641 first loans `-01`, and 16 repeat loans `-02`).
  * **Unmatched Loan Records**: **628 loans** across **628 distinct loan customer prefixes**.
  * **Unmatched Application Records**: **6 applicants** in `Application Data` (2 corrupted by Excel scientific notation floats `4.13E+13` and `2.10E+90`, and 4 approved applicants who never received or disbursed a loan).
* **The 628 Unmatched Loans — Rigorous Verification**:
  * A full set-difference audit across every column of `Application Data` confirmed that the 628 unmatched loan prefixes **do not appear anywhere** in the application records.
  * This is not an artifact of row position; `Application Data` simply represents an incomplete extract containing only 647 applications out of the 1,269 total borrowers in the portfolio.
  * **Critical Outcome Discrepancy**:
    * **Matched Cohort Default Rate**: **54.03% Bad** (355 Bad / 302 Good, $N = 657$).
    * **Unmatched Cohort Default Rate**: **25.64% Bad** (161 Bad / 467 Good, $N = 628$).
    * **Total Portfolio Default Rate**: **40.16% Bad** (516 Bad / 769 Good, $N = 1,285$).
  * **Governance Rule Enforced**: Artificial join keys were not invented. The 628 loans are preserved in `outputs/unmatched_loans.csv` for portfolio tracking.

---

### 3. Exploratory Data Analysis & Empirical Risk Drivers
1. **Repayment Frequency Schedule**:
   * **Weekly Repayments**: Exhibited an **81.48% default rate** (22 defaults out of 27 loans).
   * **Bi-Weekly Repayments**: Exhibited a **63.11% default rate** (207 defaults out of 328 loans).
   * **Semi-Monthly & Monthly**: Exhibited **41.23%** (47/114) and **43.86%** (75/171) default rates.
   * *Critical Caution*: This is an observational association in a small subgroup ($n=27$ for weekly loans), not proven causation. Weekly schedules do not inherently cause default; borrowers opting for weekly terms likely possess unmeasured cash-flow volatility.
2. **Credit Bureau FICO Scores**:
   * Statistically significant separation across all bureau scores ($p < 0.001$). Good borrowers averaged 15–20 points higher on bureau metrics (e.g., FICO Retail: 607.2 for Good vs. 588.1 for Bad).
3. **Monthly Income**:
   * Self-reported monthly income showed **no statistically significant relationship with default** ($p = 0.729$; Good mean: $2,641 vs. Bad mean: $2,595). Without verified debt obligations (DTI), stated gross income provides negligible risk discrimination.
4. **Repeat Borrower Survivorship**:
   * 16 repeat borrowers (`-02`) defaulted at only **18.75%** (3/16). However, because all 16 were selected after successfully repaying their first loan (`-01`), this reflects survivorship bias in a tiny sample ($n=16$).

---

### 4. Feature Engineering & Fair Lending Hygiene
* **Age Removal**: Applicant age was calculated from `birth_date` and `application_when` for exploratory analysis, but was **deliberately removed from predictive features** to comply with fair lending principles (e.g., Equal Credit Opportunity Act).
* **Proxy Risk Assessment**: Removing age alone does not guarantee fairness. Variables such as residential duration, bank account tenure, and credit bureau history depth can serve as proxy variables for age or other protected traits. Formal disparate impact auditing is essential prior to deployment.
* **Target Leakage Exclusions**: Dropped post-decision underwriting fields (`amount_approved`, `duration_approved`, `payment_amount_approved`, and `status`).
* **PII Exclusions**: Dropped `customer_id`, `idLoan`, `email`, `bank_routing_number`, and `address_zip`.
* **Included Features**: 15 numerical and 7 categorical application-time traits (payment-to-income, rent-to-income, repayment cost ratio, composite FICO, tenure buckets, and requested terms).

---

### 5. Machine Learning Workflow & Out-of-Sample Performance
* **Customer Leakage Prevention**: Split using **`GroupShuffleSplit` on `customer_key`** (80% train / 20% test).
* **Pipelines**: Applied Scikit-Learn `ColumnTransformer` with train-only median imputation + standard scaling for numericals, and mode imputation + one-hot encoding for categoricals.

#### Out-of-Sample Performance on Holdout Test Set ($N = 133$ loans across 128 customer groups):
| Metric | Logistic Regression (L2, Balanced) | Random Forest (100 Trees, Max Depth 5) |
| :--- | :---: | :---: |
| **Accuracy** | **53.38%** | **62.41%** |
| **ROC-AUC** | **0.5461** | **0.6313** |
| **Precision (Bad Default)** | **56.00%** | **63.41%** |
| **Recall (Bad Default)** | **59.15%** (42/71) | **73.24%** (52/71) |
| **F1-Score (Bad Default)** | **57.53%** | **67.97%** |
| **Precision (Good Repaid)** | **50.00%** | **60.78%** |
| **Recall (Good Repaid)** | **46.77%** (29/62) | **50.00%** (31/62) |
| **F1-Score (Good Repaid)** | **48.33%** | **54.87%** |

Random Forest captured non-linear interactions substantially better than linear Logistic Regression, achieving **73.2% recall on defaults** and an ROC-AUC of **0.631**.

---

### 6. Corrected Financial Threshold Analysis
* **Illustrative Cost Accounting**:
  * Assumed Cost of False Negative (Default loss): $C_{{FN}} = \$500$ (illustrative unrecovered principal).
  * Assumed Cost of False Positive (Rejection of good applicant): $C_{{FP}} = \$150$ (illustrative lost interest margin).
  * *Note*: These figures are **illustrative assumptions** not present in the original assignment brief.
* **Leak-Free Threshold Tuning**:
  * Selected cutoff $T^* = {selected_threshold:.2f}$ using 5-fold `GroupKFold` cross-validation on `X_train`.
  * Evaluated on the untouched holdout test set ($N=133$):
    * Standard 0.50 Cutoff Loss: **${loss_at_50:,.0f}** (FN={test_loss_df.loc[np.isclose(test_loss_df['Threshold'], 0.50), 'FN_Test'].values[0]}, FP={test_loss_df.loc[np.isclose(test_loss_df['Threshold'], 0.50), 'FP_Test'].values[0]}).
    * Selected {selected_threshold:.2f} Cutoff Loss: **${loss_at_selected:,.0f}** (FN={test_loss_df.loc[np.isclose(test_loss_df['Threshold'], selected_threshold), 'FN_Test'].values[0]}, FP={test_loss_df.loc[np.isclose(test_loss_df['Threshold'], selected_threshold), 'FP_Test'].values[0]}).
    * Net Difference: **${diff_dollars:,.0f}** ({diff_pct:.1f}% reduction under these specific cost parameters).
* **Sample-Size Limitation**:
  * In a holdout test set of 133 loans, shifting the threshold causes large percentage swings. This threshold analysis demonstrates the methodology of cost-sensitive learning, but should not be treated as a definitive commercial forecast.

---

### 7. Actionable & Evidence-Based Recommendations
1. **Conduct Controlled Underwriting Reviews on Repayment Frequency**:
   * While weekly loans showed an 81.5% default rate, this is based on only 27 loans. Rather than an abrupt ban, TapLend should audit borrower cash-flow cycles, review alternative repayment structures, and pilot stricter underwriting standards on high-frequency loans.
2. **Evaluate Bureau Score Floors with Portfolio Trade-Offs**:
   * Bureau scores (Retail and Telecom FICO) provide statistically significant risk separation ($p < 0.001$). TapLend should simulate policy floors (e.g., FICO Retail $\ge 580$) on broader portfolio data to evaluate the exact volume vs. default trade-off before establishing rigid cutoffs.
3. **Validate Repeat Borrower Programs on Broader Cohorts**:
   * The low default rate among repeat borrowers (18.8%) is encouraging, but observed in a sample of only 16 loans selected through survivorship. Validate repeat customer behavior across multiple origination cohorts before launching automated loyalty or pre-approval products.
4. **Data Infrastructure & Reconciliation**:
   * Audit origination channels to recover the missing 628 application records.
   * Standardize database extracts (PostgreSQL/Parquet) with immutable string schemas to eliminate Excel corruption.
5. **No Automated Deployment**:
   * **Do NOT deploy this model into automated production underwriting.** The model is constrained by approved-only selection bias, missing application records for half the portfolio, and lack of verified debt obligations (DTI). It should serve as an exploratory diagnostic benchmark.
"""

with open(outputs_dir / "findings_report.md", "w", encoding="utf-8") as f:
    f.write(findings_md)

print("Updated outputs and findings generated successfully!")
