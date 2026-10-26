# Steam Games (2021–2025) Metrics Analysis
### CSE558 Data Science — Monsoon 2026, IIIT-Delhi | Midterm Course Project

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Course](https://img.shields.io/badge/Course-CSE558%20Data%20Science-orange.svg)]()
[![Status](https://img.shields.io/badge/Audit-Passed%20100%25-brightgreen.svg)]()
[![Reproducible](https://img.shields.io/badge/Reproducibility-Verified%20(Seed%20558)-success.svg)]()

### Team Members
| # | Name | Roll Number | Institute Email | Contribution |
|---|------|-------------|-----------------|--------------|
| 1 | **Atharv Goel** | `MT26155` | `atharv26155@iiitd.ac.in` | Dataset ingestion, cleaning, scale verification (1, 2, 3), EDA (4), and primary hypothesis tests (5) |
| 2 | **Ishwaani Batra** | `MT26166` | `ishwaani26166@iiitd.ac.in` | Pre-test specifications, statistical comparisons, Holm-Bonferroni correction, limitations, report & README (5, 6) |

---

## 1. Executive Overview

This repository contains the complete dataset, analysis pipeline, and confirmatory statistical inference framework for our midterm project in **CSE558: Data Science**.

We investigate the commercial and storefront metrics of **65,521 Steam titles published between 2021 and 2025**. Specifically, we examine how **game pricing**, **platform technical feature adoption** (e.g., Steam Cloud), and **release-year cohorts** are associated with achieving public storefront traction (positive public recommendation counts) and reported volume.

All analysis adheres strictly to pre-registered confirmatory protocols:
* **Strict Methodological Partitioning:** A 70/30 stratified split (`SEED = 558`) divides the data into an **Exploration Partition** ($N = 45,814$, 70%) for exploratory data analysis (EDA) and a held-out **Confirmation Partition** ($N = 19,635$, 30%) used exclusively to evaluate our 5 primary hypothesis tests.
* **Scale Requirement:** Programmatically evaluated at $65,449 \text{ rows} \times 76 \text{ encoded features} = \mathbf{4,974,124} \ge \mathbf{3,000,000}$ matrix entries, exceeding course specifications.
* **Family-Wise Error Control:** Multiplicity is formally corrected across all five primary tests using the **Holm-Bonferroni** step-down procedure and Bonferroni adjustments ($\alpha = 0.05$).

---

## 2. Repository Structure

```text
.
├── steam_metrics_analysis_final.ipynb   # Main reproducible Jupyter notebook (EDA + Tests)
├── verify_final_notebook.py             # Automated TA verification & reproducibility test suite
├── CSE558_Project_Report_Template.tex   # LaTeX source code for the 8-page project report
├── a_steam_data_2021_2025.csv           # Raw Steam games catalog dataset (65,521 records)
├── steam_data_2021_2025.csv             # Byte-identical backup of raw dataset
└── README.md                            # Project documentation, methodology, and execution guide
```

---

## 3. Dataset Description & Provenance

* **Source:** Kaggle — [*Steam Games Dataset 2021–2025 (65K)*](https://www.kaggle.com/datasets/jypenpen54534/steam-games-dataset-2021-2025-65k/data) by jypenpen54534.
* **Raw Dimensions:** 65,521 rows $\times$ 10 columns.
* **Cleaned Dimensions:** 65,449 rows $\times$ 90 columns (including 76 binary model features, `has_rec`, `log_rec`, `price_band`, and `parsed_date`).
* **Scale Verification Arithmetic:**
  $$\text{\#instances} \times \text{\#attributes} = 65,449 \times 76 = 4,974,124 \ge 3,000,000$$

### Data Dictionary

| # | Attribute Name | Statistical Type | Unit | Range / Allowed Domain | Missingness | Description & Real-World Meaning |
|---|----------------|------------------|------|------------------------|-------------|----------------------------------|
| 1 | `appid` | Nominal (Key) | Identifier | `[1,200,220, 4,243,930]` | 0 (0.00%) | Unique Steam numerical application key. |
| 2 | `name` | Nominal | Text | Text string | 0 (0.00%) | Commercial title as displayed on the Steam store. |
| 3 | `release_year` | Discrete (Ordinal)| Year | `[2021, 2025]` | 0 (0.00%) | Calendar year the game was published. |
| 4 | `release_date` | Nominal / Date | String | `[2021-01-01, 2025-12-31]` | 0 (0.00%) | Formatted date string (1,400 unparsed irregular strings). |
| 5 | `genres` | Nominal (Multi-label)| Text | Semicolon tags (22 distinct) | 66 (0.10%) | Developer-assigned genres (e.g., Action;Indie). |
| 6 | `categories` | Nominal (Multi-label)| Text | Semicolon tags (54 distinct) | 7 (0.01%) | Platform/storefront features (e.g., Steam Cloud). |
| 7 | `price` | Continuous | USD ($) | `[$0.00, $1,900.00]` | 0 (0.00%) | Purchase price ($0.00 denotes free-to-play). |
| 8 | `recommendations`| Discrete | Counts | `[0, 862,487]` | 0 (0.00%) | Stored count of public user recommendations. |
| 9 | `developer` | Nominal | Text | Studio Name | 53 (0.08%) | Credited developer studio (imputed with `""`). |
| 10 | `publisher` | Nominal | Text | Publisher Name | 183 (0.28%) | Commercial publishing entity (imputed with `""`). |

---

## 4. Data Preparation Summary

| Preprocessing Step | Issue Identified | Action Taken | Methodological Justification |
| :--- | :--- | :--- | :--- |
| **Missing Tag Audit** | 72 rows lacked genre or category tags | Dropped (65,521 $\to$ 65,449 rows, retaining 99.89%) | Synthesizing developer tags injects arbitrary bias into feature hypothesis tests. |
| **Metadata Imputation** | 45 missing `developer`, 174 `publisher` | Imputed with empty string `""` | Preserves consistent string dtypes without inventing studio identities. |
| **Duplicate Title Audit** | 737 rows share titles across 344 names | Retained as distinct `appid`s | Verified as legitimate distinct editions, demos, prologues, or remakes. |
| **Outlier Handling** | 165 games priced $> \$100$ (up to $\$1,900$) | Retained; binned into price tiers | Legitimate commercial assets/bundles; grouping avoids straight-line leverage. |
| **Multi-label Encoding** | Semicolon-delimited tags | Expanded to 76 binary indicator columns | Provides full model feature space (22 genres + 54 categories). |
| **Target Variables** | Heavy zero mass in recommendations | Created `has_rec = (rec > 0)` and `log_rec = log10(1 + rec)` | Captures binary traction and log-transformed volume without assuming suppression. |

---

## 5. Research Questions & Formal Problem Statements

### Basic Exploratory Questions
1. What is the price distribution of games released on Steam?
2. Are there extreme high-priced outliers in the commercial catalog?
3. What are the most common genre tags among published games?

### Our Confirmatory Problem Statements
* **P1 (primary):** *Among mid-tier paid indie games ($5 < \text{price} \le \$20$), does Steam Cloud integration increase the proportion of games with positive recommendations, or is there no observable difference?*
* **P2:** *Across commercial price bands, is there a positive linear trend component in positive recommendation rates, or do rates remain invariant across price tiers?*
* **P3:** *Among paid titles with positive recommendation counts, does higher price correlate with higher reported recommendation volume, or is there no monotonic rank association?*

### Why They Are Novel (Not on the Obvious List)
The basic questions inspect univariate catalog distributions in isolation (overall price spread, tag frequencies, and extreme outliers). In contrast, our problem statements investigate **multivariate conditional relationships** and platform features—specifically isolating mid-tier commercial indie titles to evaluate technical feature adoption, evaluating ordered price-tier trend dynamics, and analyzing volume associations conditional on positive counts. These cannot be answered by surface-level summaries and require subgroup segmentation, ordinal scoring, and formal confirmatory testing.

---

## 6. Confirmatory Hypothesis Testing Results

All 5 primary tests were evaluated strictly on the **Confirmation Partition ($N = 19,635$)** with prespecified effect thresholds ($\delta$) and Holm-Bonferroni correction ($\alpha = 0.05$):

| # | Test Name | Population & Sample ($N$) | Statistical Method | Test Statistic ($df$) | Point Estimate [95% CI] | Raw $p$-value | Holm-Adjusted $p$ | Prespecified $\delta$ | Practical Verdict | Final Decision |
|---|:---|:---|:---|:---|:---|:---|:---|:---|:---:|:---:|
| **1** | **Steam Cloud Adoption (P1 Primary)** | Mid-tier paid ($5 < \text{price} \le \$20$) ($N = 6,767$) | Two-Proportion $z$-test (one-sided) | $z = 21.2321$ (N/A) | $+0.2199$ $[+0.1984, +0.2414]$ | $2.41 \times 10^{-100}$ | $7.24 \times 10^{-100}$ | $\delta = +0.050$ (+5 pp) | **Confirmed** (CI lower $> \delta$) | **Reject $H_0$** |
| **2** | **Free vs. Paid Recommendation Volume** | All confirmation titles ($N = 19,635$) | Welch unequal-variance $t$-test | $t = -36.9038$ ($df = 6,252$) | $-0.3551$ $[-0.3740, -0.3363]$ log units | $7.74 \times 10^{-274}$ | $3.10 \times 10^{-273}$ | $\delta = -0.10$ log units | **Confirmed** (CI upper $< -\delta$) | **Reject $H_0$** |
| **3** | **Release-Year Cohorts (2021–2024)** | Paid titles in 2021–2024 ($N = 11,386$) | $\chi^2$ Test of Independence | $\chi^2 = 1.3411$ ($df = 3$) | Cramer's $V = 0.0109$ $[0.0039, 0.0210]$ | $0.7194$ | $0.7194$ | $\delta = 1.0\text{ pp}$; TOST margin $\pm 0.010$ | **Not Confirmed** (TOST $p = 0.285$) | **Fail to Reject $H_0$** |
| **4** | **Price vs. Volume (Positive-Count Paid)** | Paid games with $\text{rec} > 0$ ($N = 2,363$) | Spearman Rank Correlation | $\rho_s = 0.2613$ ($df = 2,361$) | $\rho_s = 0.2613$ $[0.2237, 0.3011]$ | $1.68 \times 10^{-38}$ | $3.36 \times 10^{-38}$ | $\delta = 0.10$ | **Confirmed** (CI lower $> \delta$) | **Reject $H_0$** |
| **5** | **Price Tier Trend Component** | All confirmation titles ($N = 19,635$) | Cochran-Armitage Trend Test | $z = 46.5938$ (N/A) | Unadjusted $\text{OR} = 2.1478$ $[2.0727, 2.2256]$ per tier | $< 1 \times 10^{-300}$ | $< 1 \times 10^{-300}$ | $\delta: \text{OR} = 1.10$ per tier | **Confirmed** (OR lower $> \delta$) | **Reject $H_0$** |

### Multiple Testing Conclusion
Under both Bonferroni ($\alpha_{\text{Bonf}} = 0.010$) and Holm-Bonferroni corrections at $\alpha = 0.05$:
* **Tests 1, 2, 4, and 5** reject their respective null hypotheses with overwhelming significance and confirm practical importance against their prespecified thresholds.
* **Test 3** fails to reject $H_0$ ($p = 0.7194$), and TOST equivalence analysis confirms that cohort differences across 2021–2024 remain minimal.

---

## 7. Methodological Limitations & Causal Caveats

1. **Association, Not Causation:** Platform features (e.g., Steam Cloud) and pricing tiers are not randomly assigned. Studios integrating Steam Cloud may possess higher development budgets, marketing support, or engineering experience.
2. **Zero-Count Ambiguity:** The Steam API and available collection scripts do not document whether stored zeros represent true zeros or suppressed/unavailable counts. The metric `has_rec` is strictly interpreted as an indicator of a positive stored value.
3. **Selection in Test 4:** Test 4 conditions on games having positive counts ($n = 2,363$), which represents a selected commercial subset and cannot be generalized to the entire paid catalog.
4. **Unknown Snapshot Windows:** The dataset lacks an explicit scrape timestamp; older titles have longer observation windows, which must be interpreted observationally.

---

## 8. Reproducibility & How to Run

### Prerequisites
* Python 3.10+ (tested on Python 3.12.15)
* Virtual environment recommended

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/cse558-steam-metrics.git
cd cse558-steam-metrics
pip install -r requirements.txt
```

*Required Libraries:*
```text
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
nbformat>=5.9.0
```

### Running the Analysis
1. **Interactive Jupyter Session:**
   ```bash
   jupyter notebook steam_metrics_analysis_final.ipynb
   ```
2. **Headless Execution (Command Line):**
   ```bash
   jupyter nbconvert --to notebook --execute steam_metrics_analysis_final.ipynb --output steam_metrics_executed.ipynb
   ```
3. **Run the Automated Verification Audit:**
   To verify that all 48 cells execute with zero errors, confirm sample sizes, and check statistical assertions:
   ```bash
   python verify_final_notebook.py
   ```
   *Expected terminal output:*
   ```text
   === AUDIT OF steam_metrics_analysis_final.ipynb ===
   Total cells: 48
   Code cells: 27 | Empty outputs: 0 | Execution errors: 0
   Syntax warnings: 0 | Windows paths in warnings: 0
   ...
   === AUDIT COMPLETE: ALL ASSERTIONS PASSED PERFECTLY ===
   ```

---

## 9. Academic Integrity & Course Declaration

This work was produced for the midterm evaluation in **CSE558: Data Science (Monsoon 2026)** at **IIIT-Delhi**. All code, hypothesis definitions, and statistical interpretations represent original work adhering to course guidelines.
