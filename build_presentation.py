import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

project_root = Path(__file__).resolve().parent
outputs_dir = project_root / "outputs"
presentation_dir = project_root / "presentation"
presentation_dir.mkdir(parents=True, exist_ok=True)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color Palette
COLOR_NAVY = RGBColor(26, 43, 76)      # Primary header
COLOR_SLATE = RGBColor(70, 80, 95)     # Subtitle
COLOR_ACCENT = RGBColor(217, 83, 79)   # Risk/Bad accent (coral red)
COLOR_SUCCESS = RGBColor(46, 125, 50)  # Good/Success accent (forest green)
COLOR_CARD_BG = RGBColor(245, 247, 250)# Light card fill
COLOR_BORDER = RGBColor(220, 224, 230) # Border
COLOR_TEXT_DARK = RGBColor(33, 37, 41) # Dark charcoal body text
COLOR_WHITE = RGBColor(255, 255, 255)

def add_header(slide, title_text, category_text="TAPLEND INDIA • CREDIT RISK ASSESSMENT"):
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.35))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = RGBColor(120, 130, 145)
    
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_NAVY

def add_card(slide, left, top, width, height, title, items, bg_color=COLOR_CARD_BG, border_color=COLOR_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1)
    
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_right = Inches(0.25)
    tf.margin_top = Inches(0.2)
    tf.margin_bottom = Inches(0.2)
    
    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_NAVY
    p0.space_after = Pt(8)
    
    for item in items:
        p = tf.add_paragraph()
        p.text = f"• {item}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_DARK
        p.space_after = Pt(4)

blank_layout = prs.slide_layouts[6]

# ==============================================================================
# SLIDE 1: TITLE SLIDE
# ==============================================================================
slide1 = prs.slides.add_slide(blank_layout)

bg = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
bg.fill.solid()
bg.fill.fore_color.rgb = COLOR_NAVY
bg.line.fill.background()

title_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.5), Inches(3.2))
tf = title_box.text_frame
tf.word_wrap = True

p_sub = tf.paragraphs[0]
p_sub.text = "TAPLEND INDIA — DATA SCIENCE INTERN ASSESSMENT"
p_sub.font.size = Pt(13)
p_sub.font.bold = True
p_sub.font.color.rgb = RGBColor(160, 190, 230)
p_sub.space_after = Pt(14)

p_main = tf.add_paragraph()
p_main.text = "Credit Risk Analysis & Underwriting Optimization"
p_main.font.size = Pt(32)
p_main.font.bold = True
p_main.font.color.rgb = COLOR_WHITE
p_main.space_after = Pt(16)

p_desc = tf.add_paragraph()
p_desc.text = "A rigorous, leak-free evaluation of applicant risk signals, verified data matching corrections, fair lending feature hygiene, and illustrative decision economics."
p_desc.font.size = Pt(14)
p_desc.font.color.rgb = RGBColor(220, 230, 245)

meta_box = slide1.shapes.add_textbox(Inches(1.2), Inches(5.6), Inches(10.5), Inches(1.0))
tf_meta = meta_box.text_frame
p_meta = tf_meta.paragraphs[0]
p_meta.text = "Author: Data Science Candidate  |  Context: 15–20 Minute Executive Presentation  |  Dataset: Oct 2010 – Apr 2011 Portfolio"
p_meta.font.size = Pt(11)
p_meta.font.color.rgb = RGBColor(180, 200, 220)

slide1.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 1):
Good morning/afternoon everyone. Today I'm presenting my credit risk analysis for TapLend India. 
Our objective is to evaluate whether applicant characteristics captured at application time can reliably predict loan repayment outcomes—distinguishing Good loans from Bad loans.
Throughout this project, I audited the data architecture, verified identifier corrections, enforced fair-lending feature hygiene, built a leak-free machine learning workflow, and evaluated the economic trade-offs required before considering any automated underwriting. Let's begin."""

# ==============================================================================
# SLIDE 2: THE BUSINESS PROBLEM & UNDERWRITING ECONOMICS
# ==============================================================================
slide2 = prs.slides.add_slide(blank_layout)
add_header(slide2, "Business Objective & Underwriting Economics")

add_card(slide2, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0),
         "The Lending Challenge",
         [
             "TapLend provides consumer credit to subprime and near-prime borrowers.",
             "Core dilemma: Balancing loan origination volume against portfolio default losses.",
             "Objective: Identify which applicant characteristics predict repayment to guide future credit decisions.",
             "Regulatory boundary: Decisions must be transparent, non-discriminatory, and legally compliant."
         ])

add_card(slide2, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0),
         "Illustrative Economic Costs",
         [
             "Asymmetric Cost Framework:",
             "  • Cost of False Negative (Default loss): Lost principal (~$500 illustrative loss).",
             "  • Cost of False Positive (Good rejection): Lost profit margin (~$150 illustrative profit).",
             "Illustrative Nature:",
             "  • Neither $500 nor $150 is specified in the raw workbook; they represent illustrative operational assumptions.",
             "  • Because default costs exceed margin by >3:1, naive 0.50 cutoffs are economically misaligned."
         ])

add_card(slide2, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
         "Analytical Evaluation Pillars",
         [
             "1. Architectural Audit: Verify customer and loan identifier linkages.",
             "2. Empirical Risk Drivers: Statistically evaluate bureau scores, cash flow, and repayment terms.",
             "3. Fair Lending Hygiene: Deliberately exclude age; review proxy risk attributes.",
             "4. Leak-Free ML: Model defaults with grouped customer cross-validation.",
             "5. Governance Caution: Document limitations and reject premature automation."
         ])

slide2.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 2):
In digital consumer lending, underwriting is inherently an asymmetric economic problem. 
If we approve a borrower who defaults, we lose unrecovered principal—illustrated here as roughly $500 based on average loan sizes. If we reject a creditworthy borrower, we lose the interest margin, illustrated as roughly $150.
I want to emphasize upfront that these dollar amounts are illustrative business assumptions, not numbers found in the raw assignment brief.
Because defaults are over three times as costly as missed loan opportunities, our threshold strategy cannot treat errors symmetrically. However, we must evaluate thresholds rigorously on holdout data to avoid overfitting."""

# ==============================================================================
# SLIDE 3: DATASET OVERVIEW & ARCHITECTURE
# ==============================================================================
slide3 = prs.slides.add_slide(blank_layout)
add_header(slide3, "Dataset Overview: Two Siloed Tables Across 6 Months")

add_card(slide3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
         "Application Data (N = 647)",
         [
             "Observation Window: October 16, 2010 to April 17, 2011 (exact 6 months).",
             "Dimensions: 647 rows × 31 columns. 100% unique customer IDs.",
             "Applicant Demographics: Birth date, housing tenure, residence rent vs. own.",
             "Financial Attributes: Self-reported monthly income ($2,618 avg), monthly rent.",
             "Credit Bureau Scores: raw_l2c_score, plus four specialized FICO scores (Retail, Telecom, Bank Card, Money).",
             "Requested Terms: Amount requested ($300–$1,000), duration, payment frequency (weekly to monthly).",
             "Post-Decision Fields: amount_approved, duration_approved (must be excluded)."
         ])

add_card(slide3, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
         "Loan Performance (N = 1,285)",
         [
             "Portfolio Coverage: All loans issued during the same 6-month period.",
             "Dimensions: 1,285 rows × 2 columns (idLoan, flgGood).",
             "Outcome Label: Binary repayment performance (Good vs. Bad).",
             "Overall Outcome Distribution: 769 Good loans (59.84%) vs. 516 Bad loans (40.16%).",
             "No Date Stamps: Loan Performance lacks disbursement or maturity timestamps.",
             "Identifier Structure: idLoan combines customer hex prefix with loan sequence (e.g. 9ECE67D6C5-01).",
             "Core Investigation: Why are there 1,285 loans for only 647 application records?"
         ])

slide3.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 3):
Here is an overview of the two tables provided in the assignment workbook:
Application Data has 647 rows and 31 columns. Every applicant in this dataset has status 'Approved'. It captures customer traits, requested loan terms, and five specialized credit bureau scores.
Loan Performance has 1,285 rows and two columns: idLoan and flgGood. 60% of loans are Good and 40% are Bad.
Immediately, an apparent discrepancy appears: there are 1,285 disbursed loans, but only 647 application records. Auditing this relationship was our first task."""

# ==============================================================================
# SLIDE 4: THE MATCHING INVESTIGATION (VERIFIED FINDINGS)
# ==============================================================================
slide4 = prs.slides.add_slide(blank_layout)
add_header(slide4, "The Data Matching Problem: Investigation & Verified Findings")

add_card(slide4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
         "Forensic Matching Investigation",
         [
             "Identifier Architecture: customer_id is a 10-char hex string; idLoan prefixes match these 10-char hex hashes.",
             "Repeat Borrowers (16 Loans): 16 customers took a second loan (-02). All 16 were Good on loan -01, and 13 were Good on loan -02 (18.75% default rate).",
             "Justified Correction (Row 567): 902431691 lost its leading zero in Excel; restoring it matches loan 0902431691-01, yielding 657 matched loans across 641 applicants.",
             "The 628 Unmatched Loans:",
             "  • An exhaustive check confirms that these 628 prefixes do NOT appear anywhere in Application Data.",
             "  • Missingness is not inferred from row order: Application Data is simply an incomplete extract covering only 647 of the portfolio's 1,269 total borrowers.",
             "  • Matched default rate = 54.03% vs. Unmatched = 25.64% (Non-random missingness).",
             "Governance Rule: We strictly do NOT invent artificial join keys. The 628 loans lack feature data and cannot be used for predictive training."
         ])

# Add Image
if (outputs_dir / "matched_vs_unmatched_comparison.png").exists():
    slide4.shapes.add_picture(str(outputs_dir / "matched_vs_unmatched_comparison.png"),
                              Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))

slide4.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 4):
Let's address the data matching problem directly with verified facts:
1. In the baseline raw join, 656 loans matched. When we audited Row 567, we found an integer value 902431691. Excel stripped the leading zero; restoring it matches loan 0902431691-01, bringing total matched loans to 657 across 641 unique applicants.
2. 16 customers took repeat loans. All 16 had paid off their first loan successfully.
3. The remaining 628 loans in Loan Performance belong to customer prefixes that do not appear anywhere in Application Data. This is not inferred from row position—Application Data simply covers only 647 of the 1,269 total borrowers in that 6-month period.
4. Crucially, as shown on the chart, the matched cohort had a 54% default rate, while the unmatched cohort had only a 26% default rate.
Per strict data governance, we did not fabricate join keys. We preserved all 628 unmatched loans in unmatched_loans.csv, trained our models strictly on verified matched records, and documented this data pipeline gap."""

# ==============================================================================
# SLIDE 5: EXPLORATORY DATA ANALYSIS — RISK DRIVERS
# ==============================================================================
slide5 = prs.slides.add_slide(blank_layout)
add_header(slide5, "Exploratory Data Analysis: Empirical Risk Drivers & Surprises")

add_card(slide5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
         "Statistical Findings Across Features",
         [
             "Credit Bureau Scores Separate Risk (p < 0.001):",
             "  • FICO Retail: Good borrowers avg 607.2 vs. Bad avg 588.1.",
             "  • FICO Telecom: Good avg 576.2 vs. Bad avg 562.6.",
             "  • FICO Bank Card: Good avg 671.8 vs. Bad avg 659.4.",
             "  • FICO Money: Good avg 607.5 vs. Bad avg 599.3.",
             "Monthly Income Does NOT Predict Default (p = 0.729):",
             "  • Good borrowers avg $2,641 vs. Bad avg $2,595.",
             "  • Zero statistical correlation with default risk.",
             "  • Takeaway: Stated gross income without verified debt service obligations (DTI) is uninformative.",
             "Applicant Age Correlation (Removed from Model):",
             "  • Good borrowers were slightly older (44.2 vs. 40.3 years, p < 0.001).",
             "  • Regulatory Notice: Age was deliberately removed from predictive features to comply with fair lending (ECOA) rules."
         ])

# Add Image
if (outputs_dir / "fico_scores_by_outcome.png").exists():
    slide5.shapes.add_picture(str(outputs_dir / "fico_scores_by_outcome.png"),
                              Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))

slide5.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 5):
Looking at our exploratory data analysis:
First, all four specialized FICO scores—Retail, Telecom, Bank Card, and Money—demonstrate statistically significant separation between Good and Bad loans (p < 0.001). Good borrowers averaged 15 to 20 points higher on bureau scores.
Second, a major surprise: applicant monthly income has no statistically significant correlation with loan default (p = 0.729). Higher self-reported gross income did not prevent defaults. This confirms that gross income without verified active debt obligations provides little risk signal.
Third, while applicant age showed an observational correlation with default, we deliberately removed age from predictive features due to fair lending compliance."""

# ==============================================================================
# SLIDE 6: PAYMENT FREQUENCY & REPEAT BORROWER DYNAMICS
# ==============================================================================
slide6 = prs.slides.add_slide(blank_layout)
add_header(slide6, "Payment Frequency & Repeat Borrower Dynamics")

add_card(slide6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
         "Underwriting Schedule Observations",
         [
             "Weekly Payment Frequency Risk (Association ≠ Causation):",
             "  • Borrowers on weekly schedules had an 81.5% default rate.",
             "  • Critical Subgroup Limitation: Only 27 total loans were weekly (22 defaults).",
             "  • Interpretation: Weekly schedules do not inherently cause default; borrowers selecting weekly terms likely possess volatile cash flows or irregular wages.",
             "Bi-Weekly Repayments (63.1% Bad):",
             "  • Half the portfolio (n = 328). Still exhibits elevated default rates.",
             "Monthly & Semi-Monthly Schedules:",
             "  • Monthly: 43.9% Bad (n = 171) | Semi-Monthly: 41.2% Bad (n = 114).",
             "Repeat Borrower Survivorship Bias:",
             "  • Repeat loans (-02) defaulted at only 18.75% (3/16).",
             "  • Limitation: Sample is tiny (n = 16) and reflects pre-filtered borrowers who already proved creditworthiness on loan -01."
         ])

# Add Image
if (outputs_dir / "payment_frequency_by_outcome.png").exists():
    slide6.shapes.add_picture(str(outputs_dir / "payment_frequency_by_outcome.png"),
                              Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))

slide6.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 6):
Examining repayment frequency reveals critical operational insights and methodological caveats:
As shown on the chart, applicants on weekly repayment schedules experienced an 81.5% default rate. However, we must be careful: this is an observational association in a small subgroup of only 27 loans, not established causation. Weekly terms do not necessarily cause default; rather, borrowers requiring weekly installments likely face severe cash-flow volatility.
Similarly, repeat borrowers defaulted at only 18.75% on loan -02 compared to 55% for first-time loans. But this reflects survivorship bias in a tiny sample of 16 borrowers who were selected precisely because they repaid their first loan.
Both findings warrant controlled operational review rather than definitive, unverified policy mandates."""

# ==============================================================================
# SLIDE 7: FEATURE ENGINEERING & FAIR LENDING HYGIENE
# ==============================================================================
slide7 = prs.slides.add_slide(blank_layout)
add_header(slide7, "Feature Engineering & Fair Lending Hygiene")

add_card(slide7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
         "Fair Lending Hygiene & Exclusions",
         [
             "Removal of Age (Fair Lending Compliance):",
             "  • Removed from predictive features to comply with Equal Credit Opportunity Act (ECOA) standards.",
             "Proxy Risk Review:",
             "  • Removing age alone does NOT guarantee fairness.",
             "  • Variables like residential duration, bank account tenure, or credit bureau history length can serve as proxy variables for age or other protected traits.",
             "  • Formal Disparate Impact testing required before any commercial deployment.",
             "Target Leakage / Post-Decision Fields (EXCLUDED):",
             "  • amount_approved, duration_approved, payment_amount_approved, status.",
             "PII & Arbitrary Identifiers (EXCLUDED):",
             "  • customer_id, idLoan, customer_key, email, bank_routing_number, address_zip."
         ])

add_card(slide7, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
         "Engineered Application-Time Features",
         [
             "Financial & Debt Burden Ratios (INCLUDED):",
             "  • Payment-to-Income: payment_amount / monthly_income_amount. Measures installment burden relative to cash flow.",
             "  • Rent-to-Income: monthly_rent_amount / monthly_income_amount. Captures fixed housing commitments.",
             "  • Total Repayment Cost Ratio: (num_payments * payment_amount) / amount_requested. Measures effective financing cost multiple.",
             "Credit Bureau Aggregations (INCLUDED):",
             "  • mean_fico: Composite average across Retail, Telecom, Bank Card, and Money FICO scores.",
             "Account & Residence Stability:",
             "  • Tenure Buckets: residence_duration, bank_account_duration, email_duration.",
             "  • Underwriting Schedule: payment_frequency, residence_rent_or_own, direct deposit."
         ])

slide7.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 7):
Before building our models, strict feature engineering and fair lending hygiene were enforced.
First, applicant age was deliberately excluded from all predictive models to comply with ECOA fair lending regulations. 
Importantly, we do NOT claim that removing age alone guarantees fairness. Other features—such as banking tenure, residential stability, or credit history depth—can act as proxy variables for age or socioeconomic status. Any real-world deployment requires formal Disparate Impact auditing.
Second, all post-decision underwriting fields—amount approved, duration approved, and payment approved—were stripped to prevent target leakage.
We then engineered clean financial ratios: payment-to-income, rent-to-income, total repayment cost ratio, and composite FICO."""

# ==============================================================================
# SLIDE 8: MODELING ARCHITECTURE & LEAK-FREE VALIDATION
# ==============================================================================
slide8 = prs.slides.add_slide(blank_layout)
add_header(slide8, "Modeling Architecture & Leak-Free Validation")

add_card(slide8, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0),
         "Preventing Customer Leakage",
         [
             "16 repeat borrowers have multiple loans (-01 and -02).",
             "Random row-level splitting would leak the same customer's traits into both train and test.",
             "Solution: GroupShuffleSplit grouped strictly on customer_key.",
             "Train: 80% of customer groups (n = 524 loans).",
             "Test: 20% of customer groups (n = 133 loans).",
             "Guarantees holdout independence."
         ])

add_card(slide8, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0),
         "Scikit-Learn Preprocessing",
         [
             "Strict Featurization Order:",
             "Pipelines fitted ONLY on training split.",
             "Numerical Pipeline (15 Features):",
             "  • Median imputation (robust to outliers).",
             "  • StandardScaler (zero mean, unit variance).",
             "Categorical Pipeline (7 Features):",
             "  • Mode imputation for missing values.",
             "  • OneHotEncoder with handle_unknown='ignore' and drop='first'.",
             "ColumnTransformer packaging."
         ])

add_card(slide8, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
         "Candidate Algorithms",
         [
             "Model 1: Logistic Regression",
             "  • Linear classifier with L2 regularization.",
             "  • class_weight='balanced' to handle class proportions.",
             "  • Provides interpretable odds ratios for regulatory compliance.",
             "Model 2: Random Forest Classifier",
             "  • 100 trees, max_depth=5, min_samples_split=10.",
             "  • class_weight='balanced'.",
             "  • Captures non-linear thresholds and interactions without overfitting."
         ])

slide8.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 8):
To ensure complete technical rigor, we solved a critical validation problem: repeat borrower data leakage.
Because 16 customers took both a first and second loan, standard random splitting would place one loan from a customer in training and their second loan in testing. To eliminate this leak, we implemented GroupShuffleSplit on customer_key, ensuring that all loans for any given borrower were kept strictly on one side of the split.
Furthermore, all imputation and scaling parameters were learned exclusively from training folds through Scikit-Learn Pipelines.
We evaluated two complementary models: Logistic Regression for linear interpretability, and Random Forest to capture non-linear risk interactions."""

# ==============================================================================
# SLIDE 9: MODEL EVALUATION & BENCHMARKS
# ==============================================================================
slide9 = prs.slides.add_slide(blank_layout)
add_header(slide9, "Model Evaluation & Out-of-Sample Benchmarks (Excl. Age)")

add_card(slide9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
         "Model Evaluation (Holdout Test N = 133)",
         [
             "Target Definition: Positive Class = Bad Loan (Default Event = 1).",
             "Model Comparison Table (Excluding Age):",
             "  • Random Forest: Accuracy = 62.41%, ROC-AUC = 0.6313",
             "    - Bad Precision = 62.65%, Bad Recall = 73.24%, Bad F1 = 0.6753",
             "    - Good Precision = 62.00%, Good Recall = 50.00%, Good F1 = 0.5536",
             "  • Logistic Regression: Accuracy = 53.38%, ROC-AUC = 0.5461",
             "    - Bad Precision = 56.00%, Bad Recall = 59.15%, Bad F1 = 0.5753",
             "Key Model Evaluation Takeaways:",
             "  • Random Forest achieved 73.2% recall on defaults, successfully identifying 52 out of 71 defaults in the holdout test set.",
             "  • Overall discrimination is moderate (~0.63 AUC), reflecting sample size and the absence of verified debt obligations.",
             "  • Random Forest consistently out-performed linear Logistic Regression."
         ])

# Add Image
if (outputs_dir / "confusion_matrices.png").exists():
    slide9.shapes.add_picture(str(outputs_dir / "confusion_matrices.png"),
                              Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))

slide9.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 9):
Here are our actual out-of-sample evaluation results on the holdout test set of 133 loans, trained without age.
Remember, our positive class is defined as a 'Bad Loan'—because detecting credit defaults is our core underwriting objective.
The Random Forest classifier demonstrated superior performance, achieving an ROC-AUC of 0.631 and a 73.2% recall on Bad loans. In practical terms, it successfully identified 52 out of the 71 defaulting borrowers in the holdout set.
Logistic Regression achieved 53.4% accuracy and 0.546 AUC.
Both models demonstrate meaningful signal, but overall discrimination remains moderate, primarily constrained by the sample size and the lack of verified debt obligation data."""

# ==============================================================================
# SLIDE 10: LEAK-FREE DECISION THRESHOLDS & TRADE-OFFS
# ==============================================================================
slide10 = prs.slides.add_slide(blank_layout)
add_header(slide10, "Leak-Free Decision Thresholds & Illustrative Trade-Offs")

add_card(slide10, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
         "Corrected Threshold Methodology",
         [
             "Leak-Free Selection Protocol:",
             "  • Threshold selected strictly on Training Data via 5-fold GroupKFold cross-validation.",
             "  • Optimal Training Threshold: T* = 0.30.",
             "Evaluation on Untouched Holdout Test Set (N = 133):",
             "  • Default 0.50 Cutoff Loss: $14,150 (FN = 19, FP = 31).",
             "  • Selected 0.30 Cutoff Loss: $9,650 (FN = 2, FP = 59).",
             "  • Difference: $4,500 reduction under illustrative assumptions.",
             "The Volume vs. Risk Trade-Off:",
             "  • While lowering T to 0.30 catches nearly all defaults (FN drops from 19 to 2), it rejects 59 out of 62 good borrowers (FP increases from 31 to 59).",
             "  • Approval volume collapses! Real businesses must impose approval volume constraints.",
             "Sample-Size Limitation: In a test set of 133 loans, threshold tuning is sensitive to small sample noise."
         ])

# Add Image
if (outputs_dir / "financial_tradeoff_curve.png").exists():
    slide10.shapes.add_picture(str(outputs_dir / "financial_tradeoff_curve.png"),
                              Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0))

slide10.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 10):
A key methodological correction in this revised analysis is eliminating test-set tuning. 
We selected our optimal threshold strictly on training cross-validation data, finding T* = 0.30, and then evaluated it on the untouched holdout test set.
Under our illustrative cost assumptions ($500 default loss vs. $150 lost margin), setting T* = 0.30 reduces test loss from $14,150 to $9,650.
However, look at the trade-off: false negatives drop to 2, but false positives jump to 59 out of 62 good borrowers! That means 95% of good applicants are rejected, and loan origination volume collapses.
In a real lending business, underwriting teams cannot operate without loan volume constraints. Furthermore, with only 133 test cases, threshold optimization is highly sensitive to sample noise."""

# ==============================================================================
# SLIDE 11: PORTFOLIO LIMITATIONS & DATA GOVERNANCE
# ==============================================================================
slide11 = prs.slides.add_slide(blank_layout)
add_header(slide11, "Portfolio Limitations & Data Governance Risks")

add_card(slide11, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0),
         "1. Selection Bias",
         [
             "100% Approved Applications: Data contains zero rejected applicants.",
             "Survivor Bias: We cannot observe repayment behavior for denied applicants.",
             "Model Distortion: Scoring models trained only on approved loans degrade on general applicant flow.",
             "Remediation: Reject inference modeling or randomized control tranches."
         ])

add_card(slide11, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0),
         "2. Missing Portfolio Half",
         [
             "628 Loans Lack Application Records: Nearly half of all loans cannot be joined.",
             "Non-Random Missingness (MNAR): Unmatched loans default at 25.6% vs. 54.0% for matched loans.",
             "Set-Difference Proof: Prefixes do not exist anywhere in Application Data.",
             "Remediation: Ingestion pipeline reconciliation before production deployment."
         ])

add_card(slide11, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
         "3. Data Quality & Fairness",
         [
             "Excel Identifier Corruption: Hex hashes coerced to scientific floats and stripped zeros.",
             "Omitted Variable Bias: Missing verified Debt-to-Income (DTI) and active debt tradelines.",
             "Regulatory Fair Lending: Proxy risk persists even after removing age; formal disparate impact audits required.",
             "Sample Size: N = 657 is too small for automated underwriting."
         ])

slide11.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 11):
Every honest data science assessment must articulate its limitations clearly:
First, selection bias: our application dataset contains only approved applicants. We have no data on rejected borrowers.
Second, missing portfolio data: 628 loans in Loan Performance have no application records. Because their default rate is 26% versus 54% for matched loans, this missingness is clearly non-random.
Third, proxy risk and omitted variables: while age was removed, correlated features like banking tenure or credit history depth can proxy for protected demographics. Moreover, we lack active debt obligation (DTI) data.
Because of these structural limitations, we strongly advise against deploying this model into automated production."""

# ==============================================================================
# SLIDE 12: STRATEGIC RECOMMENDATIONS & ROADMAP
# ==============================================================================
slide12 = prs.slides.add_slide(blank_layout)
add_header(slide12, "Actionable & Evidence-Based Recommendations")

add_card(slide12, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0),
         "Underwriting Policy Reviews",
         [
             "1. Controlled Review of Weekly Repayments: Weekly schedule borrowers defaulted at 81.5% (n=27). Audit borrower cash-flow cycles and pilot stricter terms (association ≠ causation).",
             "2. Simulate Bureau Floors: Bureau scores show strong risk separation. Evaluate volume vs. default trade-offs for cutoffs (e.g. FICO Retail ≥ 580) on broader data.",
             "3. Validate Repeat Borrower Dynamics: 18.8% default rate on loan -02 is encouraging, but reflects survivorship in a sample of 16. Validate on broader cohorts before creating VIP programs."
         ])

add_card(slide12, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0),
         "Data Infrastructure Fixes",
         [
             "1. Reconcile Unmatched Records: Audit origination databases to recover application profiles for the 628 unmatched loans.",
             "2. Modernize Pipeline Hygiene: Replace manual Excel extracts with automated database pipelines (Parquet / PostgreSQL) with immutable string schemas.",
             "3. Log Rejected Applications: Capture and store all incoming applications—including rejections—to enable reject inference."
         ])

add_card(slide12, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
         "Essential Next Steps",
         [
             "1. Verified Debt-to-Income (DTI): Capture existing active monthly debt service to evaluate true repayment capacity.",
             "2. Cash-Flow Banking Analytics: Integrate Account Aggregator data (average balance, payroll regularity, overdraft frequency).",
             "3. Fair Lending Disparate Impact Audits: Conduct formal Disparate Impact testing across protected classes.",
             "4. Do NOT deploy automatically without addressing selection bias and data gaps."
         ])

slide12.notes_slide.notes_text_frame.text = """SPEAKER NOTES (SLIDE 12):
To conclude, I recommend three strategic pillars for TapLend:
First, evidence-based underwriting inquiries: conduct controlled reviews of weekly repayment terms, evaluate bureau score floors against portfolio volume trade-offs, and validate repeat borrower loyalty programs across broader cohorts.
Second, fix data infrastructure: eliminate manual spreadsheet exports, recover the missing 628 application records, and begin logging all applicant rejections for future training.
Third, enrich applicant data: integrate verified debt-to-income and real-time bank transaction analytics through the Account Aggregator network, while conducting formal fair lending audits.
We explicitly advise against deploying this model into automated production at this stage.
Thank you for your time. I am now very happy to answer any questions."""

# Save Presentation
pptx_path = presentation_dir / "TapLend_Credit_Risk_Analysis.pptx"
prs.save(pptx_path)
print(f"Presentation saved successfully to {pptx_path}")

# Write standalone speaker notes guide
guide_content = """# TapLend India — Presentation Guide & Complete Speaker Notes
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
  * Cost of False Negative (Default): $\approx \$500$ in unpaid principal (illustrative assumption).
  * Cost of False Positive (Rejection of Good borrower): $\approx \$150$ in lost origination margin (illustrative assumption).
  * Cost ratio: Defaults are $>3.3\times$ more costly than lost revenue.
* **Expected Question**: *"Are the $500 default cost and $150 profit margin documented in the original assignment?"*
  * **Answer**: *"No. The original assignment brief contains no dollar cost figures. These are illustrative operational assumptions based on average loan requested amounts (~$600) and standard micro-lending margins (~20-25%). They are used strictly to demonstrate the mechanics of cost-sensitive learning."*

---

### Slide 3: Dataset Overview & Architecture
* **Time**: 1.5 minutes
* **Key Message**: Describe the two datasets and the initial mismatch in row counts.
* **Key Data Points**:
  * Application Data: 647 rows $\times$ 31 columns. 100% approved applications.
  * Loan Performance: 1,285 rows $\times$ 2 columns (`idLoan`, `flgGood`).
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
  1. Conduct controlled reviews of weekly repayment terms (do not ban abruptly; association $\neq$ causation).
  2. Simulate bureau floors (e.g. FICO Retail $\ge 580$) on broader data to evaluate volume trade-offs.
  3. Validate repeat borrower dynamics on broader cohorts before creating automated loyalty programs.
  4. Reconcile missing 628 applications; standardize on Parquet/PostgreSQL; log all rejections.
  5. Integrate Account Aggregator cash-flow data and verified DTI.
  6. **Do NOT deploy this model into automated production underwriting.**
"""

with open(presentation_dir / "speaker_notes_guide.md", "w", encoding="utf-8") as f:
    f.write(guide_content)

print(f"Speaker notes guide written successfully to {presentation_dir / 'speaker_notes_guide.md'}")
