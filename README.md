# TapLend India --- Credit Risk Analysis

A data science assessment project exploring historical consumer-loan
performance, investigating application-to-loan matching quality, and
evaluating machine-learning models for distinguishing **Good** and
**Bad** repayment outcomes.

> **Important:** This is an educational analysis, not a production
> lending or automated underwriting system. The dataset is historical,
> the matched sample is small, and the results require further
> validation before any business use.

## Project objectives

-   Audit the source workbook and assess data quality.
-   Investigate the relationship between application records and
    loan-performance records without inventing missing join keys.
-   Explore applicant, loan-term, and credit-bureau characteristics in
    relation to repayment outcomes.
-   Engineer features available at application time and exclude
    post-decision fields to reduce target leakage.
-   Compare Logistic Regression and Random Forest classifiers on a
    held-out test set.
-   Discuss decision-threshold trade-offs, selection bias, missing
    records, and fair-lending limitations.
-   Provide a saved model, a command-line prediction example,
    reproducible tests, a report, and a presentation.

## Repository structure

``` text
TapLend_Data_Science_Assignment/
├── data/
│   └── AssessmentDataScientist.xlsx       # Source workbook; do not publish if it contains sensitive data
├── notebooks/
│   └── taplend_analysis.ipynb             # Main exploratory analysis and modelling notebook
├── outputs/
│   ├── data_quality_summary.csv
│   ├── feature_importance_summary.csv
│   ├── logistic_regression_coefficients.csv
│   ├── matching_investigation_summary.csv
│   ├── matched_analysis_data.csv
│   ├── unmatched_loans.csv
│   ├── missing_values_report.csv
│   ├── model_comparison_table.csv
│   ├── findings_report.md
│   └── *.png                                # Generated charts and model-evaluation visuals
├── models/
│   ├── rf_pipeline.joblib                  # Saved Random Forest pipeline
│   └── test_predictions_snapshot.csv       # Test-set prediction snapshot, if generated
├── presentation/
│   ├── TapLend_Credit_Risk_Analysis.pptx
│   └── speaker_notes_guide.md
├── save_model.py                           # Saves the fitted model pipeline
├── predict_applicant.py                    # Command-line example prediction
├── test_predictions.py                     # Regression/sanity checks for predictions and features
├── outputs_generator.py                    # Regenerates analytical outputs, if included
├── build_and_execute_notebook.py            # Notebook build/execution helper, if included
├── build_presentation.py                    # Presentation-generation helper, if included
├── requirements.txt
└── README.md
```

Some generated files may not exist until the notebook or supporting
scripts have been run. The `*.py` helper scripts shown above should be
retained in the repository only if they are present in your local
project folder.

## Dataset overview

The source workbook is `data/AssessmentDataScientist.xlsx` and contains
three sheets:

-   **Assignment:** The task brief.
-   **Application Data:** 647 rows and 31 columns of approved
    applications, dated October 16, 2010 to April 17, 2011, according to
    the project audit.
-   **Loan Performance:** 1,285 loan records with the binary outcome
    `flgGood` (`Good` or `Bad`). The full sheet contains 769 Good and
    516 Bad outcomes.

The application sheet contains applicant and requested-loan
characteristics, credit-bureau scores, and some post-decision fields.
The analysis excludes post-decision fields from predictive features.

### Matching audit

The reported matching investigation found:

-   **657 matched loan records** across **641 distinct applicants**.
-   **628 loan-performance records** without a corresponding application
    record in the available application sheet.
-   One application identifier was reported to have lost a leading zero
    during spreadsheet handling; the correction was applied only because
    it produced a supported identifier match.
-   The unmatched records were preserved for audit rather than assigned
    fabricated keys or matched by row position.

The matched and unmatched groups have different observed outcome
distributions. This difference is a warning that the available
application records may not represent the complete loan portfolio. It
does **not**, by itself, prove the precise mechanism that caused records
to be missing.

## Methodology

1.  **Data quality and matching:** inspect schema, missing values,
    identifier formats, duplicates, and join coverage.
2.  **Exploratory analysis:** examine outcome distributions,
    applicant/loan characteristics, credit-bureau scores, and repayment
    frequency.
3.  **Feature engineering:** create ratios such as payment-to-income and
    rent-to-income, alongside a composite bureau-score feature where
    appropriate.
4.  **Leakage and privacy controls:** exclude post-decision variables
    and identifiers or direct personal information from model features.
    Age was excluded from the predictive feature set in this project;
    this alone does not establish that the model is fair.
5.  **Preprocessing:** use a scikit-learn pipeline for numerical
    imputation/scaling and categorical imputation/encoding.
6.  **Validation:** use customer-group-aware splitting to reduce the
    risk of repeat records from one customer appearing across train and
    test partitions.
7.  **Model comparison:** evaluate Logistic Regression and Random Forest
    using held-out metrics.
8.  **Threshold discussion:** illustrate the effect of changing the
    classification threshold under explicitly assumed costs. These
    assumptions are illustrative, not supplied business costs from the
    workbook.

## Reported model results

The final test snapshot and test script reported the following Random
Forest results on a holdout set of **133 records**:

  Metric                 Random Forest
  -------------------- ---------------
  Accuracy                      62.41%
  ROC-AUC                       0.6313
  Precision --- Bad             62.65%
  Recall --- Bad                73.24%
  F1-score --- Bad              67.53%
  Precision --- Good            62.00%
  Recall --- Good               50.00%
  F1-score --- Good             55.36%

The corresponding reported confusion-matrix counts, with `Bad` treated
as the positive class, were **TN = 31, FP = 31, FN = 19, TP = 52**.
Metrics are specific to this split and should not be interpreted as
evidence of production performance.

The Logistic Regression comparison reported in the project documentation
was 53.38% accuracy and 0.5461 ROC-AUC. Refer to
`outputs/model_comparison_table.csv` and the executed notebook for the
authoritative comparison.

### Decision-threshold analysis

The project includes an illustrative cost analysis using an assumed cost
of **\$500 per false negative** and **\$150 per false positive**. These
are hypothetical values, not verified TapLend economics.

At the 0.50 threshold, the reported test counts of FN = 19 and FP = 31
imply an illustrative loss of:

`(19 × $500) + (31 × $150) = $14,150`

A lower threshold may catch more bad outcomes but can also classify many
genuinely good borrowers as bad. Before publishing or presenting a
specific alternative-threshold loss, verify its FN/FP counts against the
final generated threshold analysis and ensure all documents use the same
figures. Threshold choice should be based on validation data and
business constraints---not tuned to optimize the final test set.

## Saved model and prediction example

The saved pipeline is expected at `models/rf_pipeline.joblib`.

From the project root in Windows PowerShell, run the example using the
project's virtual environment:

``` powershell
.\.venv\Scripts\python.exe predict_applicant.py --example
```

The previously tested hypothetical example returned:

-   Predicted class: **Bad (High Default Risk)**
-   Estimated probability of Good / repaid: **46.4%**
-   Estimated probability of Bad / default: **53.6%**

This is a demonstration using a hypothetical profile, not an actual
applicant decision. Model probabilities are estimates and are not
guarantees.

## Run the prediction checks

``` powershell
.\.venv\Scripts\python.exe test_predictions.py
```

The latest reported run passed **5 out of 5 test suites**, covering
metric consistency, prediction snapshots, engineered features, input
validation, and probability/class checks. Re-run the test script after
changing the model, features, or prediction code.

## Setup

Use Python compatible with the pinned packages in `requirements.txt`.
The original project environment was reported as Python 3.13; if
installation fails because a package does not support your Python
version, use a compatible Python version and recreate the virtual
environment.

### Windows PowerShell

``` powershell
# Create the virtual environment if it does not already exist
python -m venv .venv

# Install dependencies
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Open the notebook
.\.venv\Scripts\python.exe -m jupyter lab notebooks/taplend_analysis.ipynb

# Run the prediction example and tests
.\.venv\Scripts\python.exe predict_applicant.py --example
.\.venv\Scripts\python.exe test_predictions.py
```

### macOS / Linux

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab notebooks/taplend_analysis.ipynb
python predict_applicant.py --example
python test_predictions.py
```

If the optional generation scripts are present, the project may also
support regenerating outputs or the presentation through:

``` powershell
.\.venv\Scripts\python.exe outputs_generator.py
.\.venv\Scripts\python.exe build_presentation.py
```

Run only scripts that actually exist in your local project folder.

## Key limitations and responsible use

-   **Historical sample:** the data covers a limited historical period
    and may not represent current borrowers or economic conditions.
-   **Selection bias:** the application dataset reportedly contains
    approved applications only. Repayment outcomes for rejected
    applicants are unavailable, so this dataset cannot establish how the
    model would perform on them.
-   **Incomplete application records:** 628 loan-performance records
    have no matching application record in the available sheet. This
    limits representativeness and may bias analysis.
-   **Small holdout set:** 133 test records produce uncertain estimates;
    subgroup results are particularly sensitive to small sample sizes.
-   **Missing affordability information:** the workbook does not provide
    all verified debt obligations needed for a robust affordability or
    debt-to-income assessment.
-   **Fairness:** removing age does not guarantee fairness. Correlated
    features may act as proxies; formal subgroup and disparate-impact
    audits would be required.
-   **Threshold costs:** the financial costs used for threshold
    illustrations are assumptions and must be replaced with validated
    business estimates before operational use.
-   **No production readiness:** this is an assessment prototype. It
    must not be used for autonomous lending decisions without data
    remediation, independent validation, fairness/legal review,
    monitoring, and appropriate human oversight.

## Deliverables

-   `notebooks/taplend_analysis.ipynb` --- main analysis and modelling
    workflow.
-   `outputs/findings_report.md` --- written findings and limitations.
-   `presentation/TapLend_Credit_Risk_Analysis.pptx` --- executive
    presentation.
-   `presentation/speaker_notes_guide.md` --- speaker script and
    interview preparation.
-   `models/rf_pipeline.joblib` --- saved model pipeline, if generated.
-   `predict_applicant.py` and `test_predictions.py` --- prediction demo
    and validation checks.
-   `outputs/` --- generated tables, charts, and audit files.

## Data and repository hygiene

Before making the repository public:

-   Do not commit `.venv/`, cache folders, secrets, or machine-specific
    paths.
-   Check whether the source workbook or generated CSVs contain names,
    contact details, bank information, customer identifiers, or other
    sensitive records. Do not publish raw or row-level sensitive data
    without explicit authorization.
-   If the assignment permits public sharing of results only, publish
    the code, methodology, and appropriately aggregated outputs instead
    of the original workbook or sensitive row-level files.
-   Keep the report, presentation, README, and speaker guide aligned
    with the final executed notebook and generated outputs.

## Disclaimer

This repository was prepared for educational and assessment purposes. It
is not financial advice, a credit decision service, or a claim that the
model is suitable for production underwriting.
