# Steam Games Metrics Analysis (2021–2025)
**CSE558 Data Science — Monsoon 2026**

## Team Members
* **[Ishwaani Batra]** — [MT26166] — [ishwaani26166@iiitd.ac.in]
* **[Atharv Goel]** — [MT26155] — [atharv26155@iiitd.ac.in]

---

## Repository Overview
This repository contains the data pipeline, exploratory data analysis, and formal hypothesis testing for our CSE558 Data Science course project. We analyze modern Steam catalog releases (2021–2025) to investigate market visibility hurdles, pricing tiers, platform feature adoption, and cohort stability.

* `steam_data_2021_2025.csv`: The primary raw dataset retrieved from Kaggle.
* `collect_data_v2.py`: The asynchronous data collection and API verification script.
* `steam_metrics_analysis_final.ipynb`: The complete end-to-end Jupyter notebook containing EDA, data preparation, splitting protocols, and all five formal hypothesis tests.
* `report.pdf`: The final compiled 8-page project report in LaTeX.

---

## Execution Instructions
1. **Prerequisites:** Ensure Python 3.8+ is installed with the required libraries:
   ```bash
   pip install pandas numpy matplotlib scipy statsmodels scikit-learn tqdm aiohttp

---

## Key Findings & Hypothesis Tests Summary
Our analysis investigated modern Steam releases (2021–2025; $N = 65,449$) using rigorous parametric and non-parametric testing on a held-out confirmation partition ($N = 19,635$):
1. **Steam Cloud Integration (Test 1):** Mid-tier indie games supporting Steam Cloud achieve a significantly higher hurdle clearance rate ($35.31\%$ vs. $13.32\%$, $z = 21.23, p = 2.41 \times 10^{-100}$), remaining robust in multivariable logistic regression ($\text{OR} = 2.54$).
2. **Free-to-Play vs. Paid Review Volume (Test 2):** Free-to-play titles exhibit substantially lower transformed review volume than paid games (Welch’s $t = -41.13, p < 1 \times 10^{-300}$), reflecting differing development intent and customer friction.
3. **Cohort Stability (Test 3):** Among mature release cohorts (2021–2024), annual hurdle clearance rates remained remarkably stable between $14.5\%$ and $16.1\%$ ($\chi^2 = 4.63, p = 0.2010$), indicating steady long-term catalog dynamics.
4. **Price vs. Review Volume (Test 4):** Conditional on crossing the 100+ review visibility hurdle, purchase price correlates positively with review volume among paid titles ($\rho_s = 0.2613, p = 1.68 \times 10^{-38}$).
5. **Cochran-Armitage Trend (Test 5):** The probability of crossing the recommendation threshold increases monotonically across ordered commercial price tiers ($z = 46.59, p < 1 \times 10^{-300}$).

---

## File Structure & Artifacts
* `steam_data_2021_2025.csv`: Raw source dataset (65,521 rows).
* `collect_data_v2.py`: Asynchronous script querying Steam's `IStoreService` and `appdetails` APIs.
* `steam_metrics_analysis_final.ipynb`: Executable Jupyter notebook covering data cleaning, multi-label one-hot encoding ($76$ binary model features yielding $>4.97\text{M}$ matrix entries), stratified splitting ($70/30$, seed 558), EDA visualizations, and confirmatory tests.
* `CSE558_Project_Report_Template.tex`: LaTeX source code for the final 8-page report.
* `report.pdf`: Compiled PDF report submitted for the midterm milestone.
