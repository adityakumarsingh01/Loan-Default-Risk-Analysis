# TapLend India — Data Science Assessment: Credit Risk Analysis

## Project Objective
This project analyzes historical consumer loan applications and subsequent loan performance records from TapLend. The primary analytical objective is to investigate whether applicant characteristics and credit bureau scores captured at application time can reliably predict loan repayment outcomes (`Good` vs. `Bad`), evaluate data quality and cohort matching anomalies, establish a leak-free machine learning workflow with fair-lending hygiene, and provide actionable, evidence-based underwriting recommendations.

---

## Dataset Description
The analysis is based on the workbook `data/AssessmentDataScientist.xlsx`, which contains three sheets:

1. **`Assignment`**:
   The primary task brief specifying an analysis of historical applications and loan outcomes over a 6-month period to guide future credit decisions.
2. **`Application Data` (647 rows × 31 columns)**:
   Covers applicants approved between October 16, 2010 and April 17, 2011. Includes applicant demographics, self-reported monthly income and housing expenses, requested financing terms, five specialized credit bureau scores (`raw_l2c_score`, `raw_FICO_telecom`, `raw_FICO_retail`, `raw_FICO_bank_card`, `raw_FICO_money`), and post-decision underwriter counter-offers (`amount_approved`, `duration_approved`, `payment_amount_approved`). Every applicant in this dataset has status `Approved`.
3. **`Loan Performance` (1,285 rows × 2 columns)**:
   Captures all loans issued during the observation window with their binary repayment performance (`flgGood`: `Good` vs. `Bad`). Across the entire table, there are 769 Good loans (59.84%) and 516 Bad loans (40.16%).

---

## Project Structure
```
TapLend_Data_Science_Assignment/
├── .venv/                              # Python 3.13 virtual environment
├── data/
│   └── AssessmentDataScientist.xlsx    # Original Excel workbook (preserved & unmodified)
├── notebooks/
│   └── taplend_analysis.ipynb          # Updated master Jupyter notebook (executed top-to-bottom)
├── outputs/
│   ├── confusion_matrices.png          # Holdout test set confusion heatmaps
│   ├── data_quality_summary.csv        # Hygiene, leakage, and cardinality audit
│   ├── feature_importance_comparison.png # Top features across RF and Logistic Regression
│   ├── feature_importance_summary.csv  # Gini importance scores from Random Forest
│   ├── fico_scores_by_outcome.png      # Bureau FICO distributions across Good vs. Bad loans
│   ├── financial_tradeoff_curve.png    # Training CV threshold selection and test loss curve
│   ├── findings_report.md              # Concise executive summary report
│   ├── income_by_outcome.png           # Monthly income distribution across loan outcomes
│   ├── loan_outcome_distribution.png   # Good vs. Bad loan outcome counts and percentages
│   ├── logistic_regression_coefficients.csv # Log-odds and odds-ratio coefficients
│   ├── matched_analysis_data.csv       # Verified matched analysis dataset (N = 657)
│   ├── matched_vs_unmatched_comparison.png # Outcome comparison between matched & unmatched cohorts
│   ├── matching_investigation_summary.csv # Quantitative matching investigation audit
│   ├── missing_values_report.csv       # Field-by-field missing value frequency and percentages
│   ├── model_comparison_table.csv      # Out-of-sample metrics (Accuracy, ROC-AUC, Precision, Recall, F1)
│   ├── monthly_income_distribution.png # Overall applicant monthly income distribution
│   ├── payment_frequency_by_outcome.png # Default rates across payment intervals (Weekly to Monthly)
│   ├── requested_loan_amounts.png      # Distribution of requested loan financing amounts
│   ├── roc_curves.png                  # Receiver Operating Characteristic curves (LR vs. RF)
│   └── unmatched_loans.csv             # 628 loans lacking application records (preserved)
├── presentation/
│   ├── TapLend_Credit_Risk_Analysis.pptx # 12-slide executive presentation with actual figures & notes
│   └── speaker_notes_guide.md          # Comprehensive speaker script, timing, and interview Q&A
├── build_and_execute_notebook.py       # Standalone script to regenerate and execute notebook
├── build_presentation.py               # Standalone script to regenerate presentation and notes
├── outputs_generator.py                # Standalone script to regenerate all analytical outputs
├── README.md                           # Master project documentation and replication guide
└── requirements.txt                    # Pinned Python package dependencies
```

---

## Required Python Packages
The project is built on Python 3.13 and uses standard scientific libraries:
* `pandas >= 2.2.0`
* `numpy >= 2.0.0`
* `matplotlib >= 3.9.0`
* `seaborn >= 0.13.0`
* `scipy >= 1.14.0`
* `scikit-learn >= 1.5.0`
* `openpyxl >= 3.1.5`
* `nbformat >= 5.10.0`
* `nbclient >= 0.10.0`
* `ipykernel >= 6.29.0`
* `python-pptx >= 1.0.2`

All dependencies are recorded in `requirements.txt`.

---

## Setup & Execution Instructions

### 1. Activate Virtual Environment
On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```
On macOS/Linux:
```bash
source .venv/bin/activate
```

If initializing a fresh environment:
```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Run the Master Jupyter Notebook
To open and execute the analysis interactively:
```powershell
jupyter lab notebooks/taplend_analysis.ipynb
# Or via Jupyter Notebook:
jupyter notebook notebooks/taplend_analysis.ipynb
```
The notebook uses dynamic relative path discovery based on the project root and executes cleanly from top to bottom.

### 3. Regenerate All Outputs & Visuals via CLI
You can also run the full end-to-end analytical pipeline via script:
```powershell
python outputs_generator.py
```

### 4. Regenerate PowerPoint Presentation
```powershell
python build_presentation.py
```

---

## Main Analysis Steps

### Phase 1: Audit & Hygiene Verification
* Inspected all initial notebook cells and preserved initial EDA charts.
* Replaced machine-specific absolute paths (`C:/Users/...`) with robust relative paths.
* Fixed formatting inconsistencies (e.g. print statement typos `/n` $\to$ `\n`).

### Phase 2: The Data Matching Investigation
* **Key Architecture**: `customer_id` is a 10-character hexadecimal string. `idLoan` combines the hex prefix with a sequence suffix (e.g. `9ECE67D6C5-01`).
* **Repeat Borrowers**: 16 customers took repeat loans (`-02`). All 16 customers repaid their first loan (`-01` Good). In their second loan, 13 were Good and 3 were Bad (default rate: **18.75%** vs. **54.84%** for first-time loans).
* **Justified Identifier Correction (Row 567)**:
  * Row 567 in Application Data had `customer_id = 902431691` (9 digits). Excel stripped the leading zero; restoring it matches loan `0902431691-01`.
  * Total matched loans: **657 across 641 distinct applicants**.
* **The 628 Unmatched Loan Records**:
  * An exhaustive set-difference audit confirms that the 628 unmatched loan prefixes do not appear anywhere in `Application Data`.
  * This is not an artifact of row order; `Application Data` simply covers only 647 out of the portfolio's 1,269 total borrowers in that 6-month period.
  * **Distribution Discrepancy**: Matched cohort default rate is **54.03% Bad** ($N = 657$), whereas the unmatched cohort default rate is **25.64% Bad** ($N = 628$).
  * **Governance Decision**: Under strict data governance rules, artificial join keys were not invented. The 628 loans were preserved in `outputs/unmatched_loans.csv` and flagged as an upstream data extraction omission.

### Phase 3: Exploratory Data Analysis & Risk Drivers
* **Payment Frequency**: Borrowers on **Weekly repayment schedules** experienced an **81.48% default rate** ($n=27$). Bi-weekly schedules had a **63.11% default rate** ($n=328$). Semi-monthly (41.23%) and monthly (43.86%) borrowers performed substantially better.
  * *Methodological Note*: This is an observational association in a small subgroup ($n=27$), not established causation. Weekly terms do not inherently cause default; borrowers opting for weekly terms likely possess volatile cash flows.
* **Credit Bureau Scores**: FICO Retail, Telecom, Bank Card, and Money scores demonstrated statistically significant separation ($p < 0.001$).
* **Stated Income vs. Default**: Self-reported monthly income showed no linear correlation with default risk ($p = 0.729$).

### Phase 4: Feature Engineering & Fair Lending Hygiene
* **Removal of Age**: Applicant age was calculated for EDA, but **deliberately removed from predictive model features** to comply with fair lending statutes (ECOA).
* **Proxy Risk Review**: Removing age alone does not guarantee fairness. Correlated traits (residential duration, banking tenure, credit bureau history depth) can act as proxy variables. Formal disparate impact auditing is essential prior to production use.
* **Leakage Exclusions**: Dropped post-decision underwriting fields (`amount_approved`, `duration_approved`, `payment_amount_approved`, `status`).
* **Privacy Exclusions**: Excluded PII (`customer_id`, `idLoan`, `email`, `bank_routing_number`, `address_zip`).
* **Engineered Ratios**: Payment-to-income, rent-to-income, total repayment cost ratio, and composite mean FICO score.

### Phase 5 & 6: Modeling & Evaluation
* **Customer Leakage Prevention**: Partitioned data using **`GroupShuffleSplit` on `customer_key`** (80% train / 20% test) to prevent repeat customer records from leaking across folds.
* **Pipelines**: Applied Scikit-Learn `ColumnTransformer` with train-only median imputation + standard scaling for numerical features, and mode imputation + one-hot encoding for categoricals.
* **Holdout Test Set Results ($N = 133$ loans across 128 customer groups, excluding age)**:
  * **Random Forest**: **62.41% Accuracy**, **0.6313 ROC-AUC**, **73.24% Bad Recall** (52/71 defaults caught), **62.65% Bad Precision**, **67.53% Bad F1-Score**.
  * **Logistic Regression**: **53.38% Accuracy**, **0.5461 ROC-AUC**, **59.15% Bad Recall**, **56.00% Bad Precision**, **57.53% Bad F1-Score**.

### Phase 7: Leak-Free Decision Thresholds & Illustrative Trade-Offs
* **Illustrative Cost Accounting**:
  * Assumed Cost of False Negative (Default loss): $\approx \$500$ in lost principal (illustrative assumption).
  * Assumed Cost of False Positive (Rejection of good borrower): $\approx \$150$ in lost profit margin (illustrative assumption).
  * *Note*: These figures are illustrative operational assumptions not present in the assignment workbook.
* **Leak-Free Threshold Tuning**:
  * Threshold $T^* = 0.30$ was selected strictly on training cross-validation (GroupKFold).
  * Evaluated on the untouched holdout test set ($N=133$):
    * Test loss at standard 0.50 cutoff: \$14,150 ($FN=19, FP=31$).
    * Test loss at selected 0.30 cutoff: \$9,650 ($FN=2, FP=59$).
  * *The Volume vs. Risk Trade-Off*: While lowering $T$ to 0.30 catches nearly all defaults ($FN$ drops from 19 to 2), it rejects 59 out of 62 good borrowers ($FP$ increases from 31 to 59). Approval volume collapses! In real business operations, volume constraints must be enforced.
  * *Sample Size Limitation*: With only 133 test cases, threshold tuning is sensitive to small sample noise.

---

## Important Limitations & Governance Realities
1. **Selection Bias (Survivor Bias)**: The application dataset contains *exclusively approved applications* (`status == 'Approved'` for 100%). We cannot observe repayment behavior for applicants who were denied.
2. **Missing 49% of Portfolio Application Records**: 628 loans lack application profiles. Their substantially lower default rate (25.6% vs. 54.0%) proves the missingness is non-random (MNAR).
3. **Small Sample Size ($N = 657$)**: Constrains statistical power and makes subgroup analyses ($n=27$ for weekly loans, $n=16$ for repeat loans) sensitive to sample noise.
4. **Omitted Debt Obligations**: The dataset lacks verified Debt-to-Income (DTI) and active debt tradelines.
5. **Fair Lending & Regulatory Boundaries**: Proxy risk persists even after removing age; formal disparate impact audits are required.
6. **No Production Deployment Claim**: The models developed here demonstrate empirical signal and underwriting principles; they are **not** ready for autonomous production deployment without resolving the missing 628 application records and conducting fair lending bias audits.

---

## Deliverables Summary
* **Master Notebook**: [taplend_analysis.ipynb](file:///c:/Users/adity/Downloads/TapLend_Data_Science_Assignment/notebooks/taplend_analysis.ipynb)
* **Executive Report**: [findings_report.md](file:///c:/Users/adity/Downloads/TapLend_Data_Science_Assignment/outputs/findings_report.md)
* **Presentation**: [TapLend_Credit_Risk_Analysis.pptx](file:///c:/Users/adity/Downloads/TapLend_Data_Science_Assignment/presentation/TapLend_Credit_Risk_Analysis.pptx)
* **Speaker Guide**: [speaker_notes_guide.md](file:///c:/Users/adity/Downloads/TapLend_Data_Science_Assignment/presentation/speaker_notes_guide.md)
* **Cleaned Matched Dataset**: [matched_analysis_data.csv](file:///c:/Users/adity/Downloads/TapLend_Data_Science_Assignment/outputs/matched_analysis_data.csv)
* **Preserved Unmatched Loans**: [unmatched_loans.csv](file:///c:/Users/adity/Downloads/TapLend_Data_Science_Assignment/outputs/unmatched_loans.csv)
