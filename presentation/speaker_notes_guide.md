# TapLend India — Presentation Guide & Complete Speaker Notes
## Credit Risk Assessment & Underwriting Optimization (Corrected & Leak-Free)

This guide accompanies the 12-slide PowerPoint presentation `presentation/TapLend_Credit_Risk_Analysis.pptx`. It provides slide-by-slide talking points, timing guidance, expected interview questions, and model answers incorporating all technical corrections.

---

### Slide 1: Title & Executive Objective
* **Time**: 1 minute
* **Key Message**: Introduce the project, scope (Oct 2010 – Apr 2011 portfolio), and four pillars of analysis: data architecture, empirical risk drivers, leak-free ML, and illustrative decision economics.
* **Speaker Script**:
  > *"Good morning/afternoon. Today I'm presenting my credit risk analysis for TapLend India. Our objective is to determine whether applicant characteristics captured at the time of application can reliably predict loan default outcomes. In this presentation, I will walk you through the data architecture, explain a critical matching problem discovered in the historical portfolio, share key empirical risk drivers, demonstrate our leak-free machine learning workflow with fair-lending hygiene, and evaluate the financial trade-offs necessary before considering any automated underwriting decisions."*

---

### Slide 2: Business Objective & Underwriting Economics
* **Time**: 1.5 minutes
* **Key Message**: Explain the asymmetric financial penalty of default vs. lost revenue, while noting that costs are illustrative assumptions.
* **Key Data Points**:
  * Cost of False Negative (Default): $pprox \$500$ in unpaid principal (illustrative assumption).
  * Cost of False Positive (Rejection of Good borrower): $pprox \$150$ in lost origination margin (illustrative assumption).
  * Cost ratio: Defaults are $>3.3	imes$ more costly than lost revenue.
* **Expected Question**: *"Are the $500 default cost and $150 profit margin documented in the original assignment?"*
  * **Answer**: *"No. The original assignment brief contains no dollar cost figures. These are illustrative operational assumptions based on average loan requested amounts (~$600) and standard micro-lending margins (~20-25%). They are used strictly to demonstrate the mechanics of cost-sensitive learning."*

---

### Slide 3: Dataset Overview & Architecture
* **Time**: 1.5 minutes
* **Key Message**: Describe the two datasets and the initial mismatch in row counts.
* **Key Data Points**:
  * Application Data: 647 rows $	imes$ 31 columns. 100% approved applications.
  * Loan Performance: 1,285 rows $	imes$ 2 columns (`idLoan`, `flgGood`).
  * Overall outcome distribution: 769 Good (59.84%), 516 Bad (40.16%).
* **Expected Question**: *"Why are there 1,285 loans for 647 applications?"*
  * **Answer**: *"That was the central anomaly that prompted our forensic matching investigation."*

---

### Slide 4: The Data Matching Problem (Verified Findings)
* **Time**: 2.5 minutes
* **Key Message**: Explain the structure of `idLoan`, repeat borrowers, Row 567 leading zero correction, and the 628 unmatched loans.
* **Key Data Points**:
  * `customer_id` is a 10-char hex hash; `idLoan` combines the hex prefix with `-01` or `-02`.
  * 16 repeat borrowers (`-02`): 100% repaid their first loan, and 81.25% repaid their second loan.
  * Row 567 in Application Data had `customer_id = 902431691` (9 digits). Restoring the leading zero matches loan `0902431691-01`, yielding **657 matched loans across 641 distinct applicants**.
  * An exhaustive set-difference audit confirms that the remaining 628 loan customer prefixes do not appear anywhere in Application Data.
  * Matched cohort default rate = **54.03%** ($n = 657$) vs. unmatched cohort = **25.64%** ($n = 628$).
* **Expected Question**: *"How do you know the 628 loans have no application data? Did you just infer that from row order?"*
  * **Answer**: *"No. While the matched loans happen to cluster in the first half of the spreadsheet, that is just file ordering. We performed an exhaustive set-difference check comparing the 628 unmatched loan prefixes against every column of Application Data. They are completely absent. Application Data is simply an incomplete extract covering only 647 out of the portfolio's 1,269 total borrowers."*

---

### Slide 5: Exploratory Data Analysis — Key Risk Drivers
* **Time**: 2 minutes
* **Key Message**: Specialized FICO bureau scores distinguish risk; stated monthly income does NOT. Age removed for fair lending.
* **Key Data Points**:
  * FICO Retail (Good 607.2 vs. Bad 588.1, $p < 0.001$).
  * FICO Telecom, Bank Card, and Money all show significant separation ($p < 0.001$).
  * Monthly Income: Good \$2,641 vs. Bad \$2,595 ($p = 0.729$, statistically insignificant).
  * Applicant age correlated with default ($p < 0.001$), but was deliberately removed from predictive features to comply with fair lending (ECOA) rules.
* **Expected Question**: *"Why doesn't monthly income predict loan repayment?"*
  * **Answer**: *"Stated monthly income measures gross cash inflow, not disposable cash flow or financial obligations. A borrower earning \$3,000 with \$2,800 in existing debt obligations is significantly higher risk than a borrower earning \$1,800 with \$500 in debt. Without verified debt obligations (DTI), gross income provides little predictive power."*

---

### Slide 6: Payment Frequency & Repeat Borrower Dynamics
* **Time**: 2 minutes
* **Key Message**: Weekly repayment schedules signal elevated risk, but represent an observational association in a small subgroup; repeat loans reflect survivorship.
* **Key Data Points**:
  * Weekly repayment default rate: **81.48%**, but sample size is small ($n = 27$ loans, 22 defaults).
  * Bi-weekly default rate: **63.11%** ($n = 328$).
  * Monthly & Semi-monthly: **41% – 44%** default rate.
  * Repeat borrowers (`-02`): **18.75%** default rate ($n = 16$).
* **Methodological Distinction**:
  * *Association vs. Causation*: Weekly schedules do not inherently cause default; borrowers opting for weekly terms likely possess volatile cash flows or irregular employment.
  * *Survivorship*: Repeat borrowers were pre-filtered by successfully repaying loan `-01`.

---

### Slide 7: Feature Engineering & Fair Lending Hygiene
* **Time**: 1.5 minutes
* **Key Message**: Strict feature hygiene—stripping age, post-decision fields, and PII, while engineering clean debt burden ratios.
* **Key Decisions**:
  * Age removed to comply with fair lending statutes (ECOA).
  * *Proxy Risk Warning*: Removing age alone does not guarantee fairness. Features like residential duration, banking tenure, or credit history length can act as proxy variables. Disparate impact audits are required.
  * Post-decision fields (`amount_approved`, `duration_approved`, `payment_amount_approved`, `status`) and PII excluded.
  * Payment-to-income, rent-to-income, total repayment cost ratio, and composite mean FICO included.

---

### Slide 8: Modeling Architecture & Leak-Free Validation
* **Time**: 1.5 minutes
* **Key Message**: GroupShuffleSplit prevents repeat borrower leakage; Scikit-Learn Pipelines prevent data snooping.
* **Validation Strategy**:
  * Grouped on `customer_key` (80% train / 20% test: 524 train loans, 133 test loans).
  * All scalers and imputers fitted strictly on training data.
  * Logistic Regression (linear baseline) vs. Random Forest (non-linear interactions).

---

### Slide 9: Model Evaluation & Benchmarks
* **Time**: 2 minutes
* **Key Message**: Random Forest delivers 62.41% accuracy and 0.6313 ROC-AUC with 73.24% recall on Bad defaults (catching 52 of 71 defaults).
* **Holdout Metrics ($N = 133$)**:
  * Random Forest: Recall (Bad) = 73.24%, Precision (Bad) = 62.65%, F1 = 0.6753, Accuracy = 62.41%, ROC-AUC = 0.6313.
  * Logistic Regression: Recall (Bad) = 59.15%, Precision (Bad) = 56.00%, F1 = 0.5753, Accuracy = 53.38%, ROC-AUC = 0.5461.
* **Takeaway**: Non-linear tree models capture compound risk interactions significantly better than linear models.

---

### Slide 10: Leak-Free Decision Thresholds & Illustrative Trade-Offs
* **Time**: 2 minutes
* **Key Message**: Threshold selected strictly on training CV ($T^* = 0.30$) and evaluated on untouched test data ($N=133$).
* **Analysis & Caveats**:
  * Test loss decreases from \$14,150 at $T=0.50$ to \$9,650 at $T=0.30$.
  * *The Trade-Off*: Catching nearly all defaults (FN drops from 19 to 2) comes at the expense of rejecting 59 out of 62 good borrowers (FP increases from 31 to 59). Approval volume collapses!
  * *Sample Size Limitation*: With only 133 test cases, threshold tuning is sensitive to small sample noise and requires commercial volume constraints.

---

### Slide 11: Portfolio Limitations & Data Governance
* **Time**: 2 minutes
* **Key Message**: Candid discussion of selection bias, missing application records, and fair lending risks.
* **Limitations**:
  1. Approved-only selection bias (survivorship bias).
  2. 628 loans missing application records (MNAR).
  3. Small sample size ($N = 657$).
  4. Fair lending regulatory compliance and proxy risks.

---

### Slide 12: Actionable & Evidence-Based Recommendations
* **Time**: 1.5 minutes
* **Key Message**: Replace unsupported mandates with evidence-based inquiries requiring validation; reject automated deployment.
* **Recommendations**:
  1. Conduct controlled reviews of weekly repayment terms (do not ban abruptly; association $
eq$ causation).
  2. Simulate bureau floors (e.g. FICO Retail $\ge 580$) on broader data to evaluate volume trade-offs.
  3. Validate repeat borrower dynamics on broader cohorts before creating automated loyalty programs.
  4. Reconcile missing 628 applications; standardize on Parquet/PostgreSQL; log all rejections.
  5. Integrate Account Aggregator cash-flow data and verified DTI.
  6. **Do NOT deploy this model into automated production underwriting.**
