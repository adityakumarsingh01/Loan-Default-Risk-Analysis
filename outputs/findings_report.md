# TapLend India — Credit Risk Analysis & Data Science Assessment
## Updated Technical Findings & Analytical Report

### 1. Executive Summary & Audit Context
This report presents a rigorous, leak-free credit risk analysis of historical loan applications and repayment outcomes from TapLend's portfolio. Following technical review, this analysis incorporates:
1. **Fair Lending Feature Hygiene**: Deliberate exclusion of applicant age to mitigate fair lending violations (ECOA) and review of proxy risk attributes.
2. **Leak-Free Threshold Tuning**: Selection of the decision probability threshold strictly via 5-fold cross-validation on training data, evaluated against an untouched holdout test set.
3. **Illustrative Cost Accounting**: Explicit labeling of cost assumptions ($500 default loss, $150 lost margin) as illustrative scenarios not documented in the raw assignment data.
4. **Evidence-Based Policy Caution**: Refinement of recommendations to distinguish between statistical association and causation, especially given subgroup sample sizes.

---

### 2. Data Matching Investigation & Findings
* **Application Data Dimensions**: 647 rows $	imes$ 31 columns. Covers approved applicants between October 16, 2010 and April 17, 2011.
* **Loan Performance Dimensions**: 1,285 rows $	imes$ 2 columns (`idLoan`, `flgGood`: **769 Good (59.84%)**, **516 Bad (40.16%)**).
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
  * Assumed Cost of False Negative (Default loss): $C_{FN} = \$500$ (illustrative unrecovered principal).
  * Assumed Cost of False Positive (Rejection of good applicant): $C_{FP} = \$150$ (illustrative lost interest margin).
  * *Note*: These figures are **illustrative assumptions** not present in the original assignment brief.
* **Leak-Free Threshold Tuning**:
  * Selected cutoff $T^* = 0.30$ using 5-fold `GroupKFold` cross-validation on `X_train`.
  * Evaluated on the untouched holdout test set ($N=133$):
    * Standard 0.50 Cutoff Loss: **$14,150** (FN=19, FP=31).
    * Selected 0.30 Cutoff Loss: **$9,650** (FN=1, FP=61).
    * Net Difference: **$4,500** (31.8% reduction under these specific cost parameters).
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
