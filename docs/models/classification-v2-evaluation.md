# ODIN Financial Classification Framework, Data Synthesis, and Evaluation Methodology

## 1. Emergency Savings Classification

Emergency Fund Coverage (EFC) measures how many months of recurring essential expenses can be covered by available liquid emergency savings. Bhutta et al. (2023) use three months of non-discretionary expenses as an emergency-savings adequacy benchmark. The Board of Governors of the Federal Reserve System (2024) also uses three months of expenses as a measure of financial resilience.

$$
\text{EFC} =
\frac{\text{Liquid Emergency Savings}}
{\text{Average Monthly Recurring Essential Expenses}}
$$

*Formula adapted from the emergency-savings coverage measure discussed by Bhutta et al. (2023).*

| Classification | Quantifiable Rule |
|---|---|
| **Emergency Fund Inadequate** | $EFC < 3$ months |
| **Emergency Fund Adequate** | $EFC \geq 3$ months |

*Classification threshold based on Bhutta et al. (2023) and Board of Governors of the Federal Reserve System (2024).*

---

## 2. Debt Burden (DSTI) Classification

The Debt-Service-to-Income Ratio (DSTI) measures the proportion of disposable income committed to debt payments. He and Zhou (2022) discuss DSTI as a household financial-vulnerability indicator and report prior research using a 40% threshold to identify financially vulnerable households.

$$
\text{DSTI} =
\frac{\text{Required Monthly Debt Payments}}
{\text{Monthly Disposable Income}}
\times 100
$$

*Formula based on the DSTI measure discussed by He and Zhou (2022). The 40% classification boundary is adopted from prior research discussed in that study.*

| Classification | Quantifiable Rule |
|---|---|
| **Not Financially Overburdened** | $DSTI \leq 40\%$ |
| **Financially Overburdened** | $DSTI > 40\%$ |

*Classification threshold based on He and Zhou (2022); the labels are ODIN's operational classifications of debt burden.*

---

## 3. Financial Margin Classification

Financial Margin (FM) measures the disposable income remaining after required debt payments and basic living costs. He and Zhou (2022) discuss negative financial margin as an indicator used in household financial-vulnerability research.

$$
\begin{aligned}
\text{FM} ={}& \text{Monthly Disposable Income} \\
&- \text{Required Monthly Debt Payments} \\
&- \text{Monthly Basic Living Costs}
\end{aligned}
$$

*Formula based on the financial-margin measure discussed by He and Zhou (2022).*

| Classification | Quantifiable Rule |
|---|---|
| **Positive Financial Margin** | $FM \geq 0$ |
| **Negative Financial Margin** | $FM < 0$ |

*The classifications apply the sign of the financial margin discussed by He and Zhou (2022). For ODIN, these categories describe the financial-margin state rather than overall financial vulnerability.*

---

## 4. Credit Card Behavior Classification

Credit-card behavior can be classified according to the number of months an account maintains a revolving balance during the preceding 12 months. Adams et al. (2022) classify accounts as transactors, light revolvers, and heavy revolvers based on their revolving-balance history.

$$
R = \sum_{m=1}^{12} I(\text{Revolving Balance}_m > 0)
$$

Where:

- $R$ = Number of months with a revolving balance during the preceding 12 months.
- $I(\cdot)$ = Indicator function that returns 1 when a revolving balance exists and 0 otherwise.

*The formula is ODIN's mathematical representation of the 12-month revolving-frequency classification described by Adams et al. (2022); it is not presented as a formula printed verbatim in that publication.*

| Classification | Quantifiable Rule |
|---|---|
| **Transactor** | $R = 0$ |
| **Light Revolver** | $1 \leq R \leq 11$ |
| **Heavy Revolver** | $R = 12$ |

*Classification rules based on Adams et al. (2022), who distinguish the three account types using the number of months with revolving balances during the preceding 12 months.*


---

## 5. Purpose and Scope of the Evaluation

ODIN will evaluate the implementation of the four documented rule-based classifications using a labeled dataset. Accuracy, precision, recall, and F1-score measure agreement between ODIN's output and an established reference label; they do not, by themselves, establish that the financial thresholds are suitable for Filipino working adults. No machine-learning classifier needs to be trained for this evaluation.

The proposed dataset will reuse ODIN's V3 forecasting inputs for income and category-level expenditures, then add *controlled synthetic* emergency-fund, loan, and credit-card records. V3's exact fields and preprocessing requirements must be checked when its implementation is finalized. V3 will not be assumed to contain actual savings balances, loan schedules, or card statements merely because it contains expenditure data.

### 5.1 Data sources and limitations

| Input | Source or generation method | What it does **not** establish |
|---|---|---|
| Income and expenditure | ODIN V3, using available 2023 PSA FIES information and HFCE temporal indicators as specified in the forecasting methodology | An individual's real savings, debt balances, or payment behavior |
| Emergency-fund balance | Controlled opening-balance scenarios plus simulated deposits and withdrawals | The prevalence of particular savings balances among NCR working adults |
| Loan principal, terms, and payments | Documented lending terms, amortization mathematics, and controlled scenarios; BSP survey summaries may contextualize loan ownership and loan types (BSP, 2025) | The joint distribution of loan principal, terms, payments, and income without suitable microdata |
| Credit-card cycles | Controlled purchase, statement, payment, and interest records; repayment categories informed by Adams et al. (2022) | Actual card-level payment history or its distribution among target users |

The BSP survey can inform *which scenarios to consider* but aggregate percentages must not be treated as evidence for how debt or savings jointly vary with a particular person's income. Do not use the old persona/archetype labels as ground truth for V3.

## 6. Synthetic Financial Data Generation

### 6.1 Link each observation to V3 financial inputs

For each synthetic financial record, retain a `record_id`, a `source_household_id` (if present), an `evaluation_month`, monthly disposable income, and category-level expenses. Map the relevant expenditure categories to **recurring essential expenses** and **basic living costs**, recording which categories were included. These two cost definitions should be fixed before running evaluation; they need not be identical.

Use consistent monthly units and a documented definition of **disposable income** (rather than silently substituting gross income). Any division by zero or missing income/expense data must lead to an explicitly defined unavailable/invalid-input outcome, not an invented numerical classification.

### 6.2 Emergency-fund synthesis

Choose controlled starting coverage levels such as 0, 1, 2, 2.99, 3.00, 3.01, and 6 months. These are *test design values*, not estimates of actual Filipino savings behavior. Convert the desired coverage to an opening fund balance:

$$
E_0 = C_0 \times M_0
$$

Where $E_0$ is opening liquid emergency savings, $C_0$ is the selected number of coverage months, and $M_0$ is the initial average monthly recurring essential expense figure.

For later months, simulate deposits and withdrawals and update the fund consistently:

$$
E_t = E_{t-1} + D_t - W_t
$$

Where $D_t$ and $W_t$ are emergency-fund deposits and withdrawals in month $t$. Do not count a savings transfer itself as a consumption expense. Prevent a negative liquid balance unless an explicitly defined separate borrowing mechanism is modeled. Recompute EFC using the actual balance and appropriate average recurring essential expenses at each observation. This permits scenarios in which a user crosses the three-month boundary.

### 6.3 Loan and debt-payment synthesis

Create separate loan accounts with `loan_id`, principal, documented rate type, periodic interest rate, term, due dates, scheduled payments, actual payments, and outstanding balance. Construct scenarios for no loans, one loan, and multiple simultaneous loans. Loan ownership proportions from aggregate surveys, where used, should be described as contextual inputs rather than an empirically established distribution for ODIN's narrower target population (BSP, 2025).

For a standard fixed-rate amortizing loan with equal monthly payments and monthly rate $r$:

$$
M = P\frac{r(1+r)^n}{(1+r)^n-1}
$$

Where $P$ is principal, $n$ is the number of monthly installments, and $M$ is the scheduled installment. If $r=0$, use:

$$
M = \frac{P}{n}
$$

Use different, documented calculations for flat-rate or other loan products. Record interest and principal separately, and update outstanding principal after each payment. Sum **required** monthly repayments for DSTI and financial margin; actual payments may be recorded separately to test missed-payment handling. Generate cases just below, exactly at, and just above 40% DSTI. Per the uploaded framework, **40% is classified as Not Financially Overburdened** and only values above 40% are classified as Financially Overburdened (He & Zhou, 2022).

Credit-card required payments may contribute to total required debt service, but must not be counted twice if card payments are also stored in the loan/debt module.

### 6.4 Financial-margin synthesis

Reuse V3 monthly disposable income and the mapped basic living costs, then subtract *required* monthly debt payments according to Section 3. Do not subtract a loan payment again if a V3 expenditure column already includes it. Generate records with negative, zero, and positive financial margin. Under the uploaded framework, **zero belongs to Positive Financial Margin** (He & Zhou, 2022).

### 6.5 Credit-card behavior synthesis

Create account-level billing records for **12 consecutive completed cycles** per evaluable account. Store `account_id`, cycle start/end, statement balance, due date, payments with timestamps, applicable interest/charges, and remaining balance. Assign selected V3 purchases to card payment methods if useful, but do not count card payments as new consumption on top of the original purchases.

Use the actual statement-and-payment ledger to determine whether an interest-bearing revolving balance exists in a cycle, rather than interpreting every positive purchase or statement balance as revolving debt. Apply the relevant card terms when calculating interest and due dates. Make three controlled groups with the following reference outcomes (Adams et al., 2022):

| Generated 12-cycle history | Expected label |
|---|---|
| No months with revolving balances | Transactor ($R=0$) |
| 1 through 11 months with revolving balances | Light Revolver ($1\leq R\leq 11$) |
| All 12 months with revolving balances | Heavy Revolver ($R=12$) |

Generate cases at $R=0,1,11,12$, plus examples involving on-time full payment, partial payment, payment after the due date, and zero spending. If fewer than 12 completed billing cycles are available, mark the classification **INSUFFICIENT_HISTORY** unless ODIN's documented production specification expressly defines a shorter-history rule. Do not silently label a newly opened card a Transactor.

## 7. Test-Case Generation and Synthetic Dataset Checks

### 7.1 Test-case design

Apply *boundary-value analysis* to numeric thresholds; *equivalence partitioning* to select representatives from every output class; and *constraint-based/scenario-based generation* to create internally consistent combinations and changes over time. Guo et al. (2024) investigate boundary-focused test-input generation; these techniques test rule implementation and are not evidence that a simulated financial profile is population-representative.

| Dimension | Key boundary and coverage tests |
|---|---|
| Emergency Savings | EFC 2.99, 3.00, 3.01; zero fund; missing or zero essential expenses |
| DSTI | 39.99%, 40.00%, 40.01%; multiple loans; zero/missing disposable income |
| Financial Margin | -₱1, ₱0, +₱1; debt payments counted exactly once |
| Credit Card Behavior | $R=0,1,11,12$; late/partial payments; incomplete billing history |

Choose the number of cases from required rule and boundary coverage rather than asserting that a fixed count is scientifically mandated. Balance the evaluation classes when useful to expose errors in rare classes, but do not claim that this balanced test set reproduces real-world class prevalence.

### 7.2 Dataset consistency and provenance

Before testing the classifier, check that: (1) monthly expense categories reconcile with source totals where the V3 methodology requires it; (2) fund balances follow deposits and withdrawals; (3) loan outstanding balances reconcile with interest/principal and repayments; (4) credit-card statements and payments reconcile with their cycle ledgers; (5) source expenditure and credit-card repayments are not double counted; and (6) the full 12-cycle history exists for each evaluated credit-card account.

Record the dataset version, random seed, source-statistic vintage, formula assumptions, and whether each variable is **source-derived**, **calculated**, or **scenario-assumed**. Compare source-derived distributions with their corresponding reference statistics. Synthetic-data evaluation literature distinguishes resemblance to reference data from downstream utility; neither guarantees the validity of a rule-based financial threshold (Hernandez et al., 2023).

## 8. Reference Labels and Classification Evaluation

### 8.1 Establish the answer key

Create a separate reference-label calculator from the *written formulas and thresholds in Sections 1–4*. It must not call ODIN's classification function or copy its predicted labels. Independently review representative calculations and edge cases. For every valid record, produce four expected labels where the necessary inputs exist. Keep missing/invalid data and incomplete credit-card history in a separately reported input-handling test set, unless ODIN explicitly treats them as classification outputs.

These reference labels show whether ODIN correctly implements its own documented rules. Because the same financial definitions determine the answer key and ODIN's target behavior, this evaluation **does not independently establish the financial validity of those definitions**.

### 8.2 Execute and score

Run the stored inputs through ODIN and record `expected_label` and `predicted_label` for **each dimension separately**. Construct a confusion matrix and count true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) for a chosen positive class. For example, select **Emergency Fund Inadequate** as the positive class and state that choice in the report.

$$
\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\text{Precision} = \frac{TP}{TP+FP}
$$

$$
\text{Recall} = \frac{TP}{TP+FN}
$$

$$
\text{F1-score} = \frac{2\,\text{Precision}\,\text{Recall}}{\text{Precision}+\text{Recall}}
$$

For Credit Card Behavior (three classes), compute one-vs-rest precision, recall, and F1-score for **each** class; report macro-averaged values and overall accuracy as well. Where a denominator is zero, use a predefined `undefined` or zero-division reporting convention and disclose it. Report separate results for synthetic rule tests, invalid-input handling, and any independently assessed profiles; do not pool them without explanation.

### 8.3 Interpretation and independent validation

A high score on synthetic cases with reference labels calculated from ODIN's stated rules demonstrates **implementation correctness on the tested cases**. It does not demonstrate that the underlying thresholds independently reflect the financial vulnerability or behavior of actual Filipino working adults. That would require a separately collected, appropriately sampled or independently assessed reference dataset, with its own documented labeling and evaluation procedure.

## 9. References (APA 7th Edition)

Adams, R. M., Bord, V. M., & Katcher, B. (2022). Credit card profitability. *FEDS Notes*. Board of Governors of the Federal Reserve System. https://doi.org/10.17016/2380-7172.3100

Bangko Sentral ng Pilipinas. (2025). *2025 consumer finance and inclusion survey*. https://www.bsp.gov.ph/Inclusive%20Finance/Financial%20Inclusion%20Reports%20and%20Publications/2025/2025CFISreport.pdf

Bhutta, N., Blair, J., & Dettling, L. J. (2023). The smart money is in cash? Financial literacy and liquid savings among U.S. families. *Journal of Accounting and Public Policy, 42*(2), 107000. https://doi.org/10.1016/j.jaccpubpol.2022.107000

Board of Governors of the Federal Reserve System. (2024). *Report on the economic well-being of U.S. households in 2023*. https://www.federalreserve.gov/publications/2024-economic-well-being-of-us-households-in-2023.htm

Guo, X., Okamura, H., & Dohi, T. (2024). Optimal test case generation for boundary value analysis. *Software Quality Journal, 32*, 543–566. https://doi.org/10.1007/s11219-023-09659-9

He, L., & Zhou, S. (2022). Household financial vulnerability to income and medical expenditure shocks: Measurement and determinants. *International Journal of Environmental Research and Public Health, 19*(8), 4480. https://doi.org/10.3390/ijerph19084480

Hernandez, M., Epelde, G., Alberdi, A., Cilla, R., & Rankin, D. (2023). Synthetic tabular data evaluation in the health domain covering resemblance, utility, and privacy dimensions. *Methods of Information in Medicine, 62*(S01), e19–e38. https://doi.org/10.1055/s-0042-1760247
