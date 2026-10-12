# FIES-HFCE v4 Cleaning and Preprocessing Pipeline

```json
{
  "document-type": "workflow",
  "version": "1.1.0",
  "date": "2026.09.27",
  "authors": ["Group 4, III-DCSAD"]
}
```

## Overview

The v4 work contains two independent data pipelines that share a 12-category vocabulary:

| Source | Frequency | Role |
| :--- | :--- | :--- |
| Family Income and Expenditure Survey (FIES) public-use files | Annual survey anchors in selected survey years | Optional observed NCR household category expenditure, income, and household-size controls. |
| Philippine Statistics Authority (PSA) Household Final Consumption Expenditure (HFCE) by purpose | Quarterly | Required observed aggregate category series for quarterly forecasting. |

Quarterly SARIMA uses only PSA HFCE. It does not read `RAW/cleaned/`, FIES anchors, or temporally disaggregated household values. The optional FIES pipeline supports household-level analysis and personalization work; its temporally disaggregated values are estimates, not observed quarterly household spending.

---

## Minimal Quarterly Forecast Pipeline

For the v4 aggregate quarterly forecast scope, run only:

```bash
uv run python training/scripts/data/build_hfce_category_forecast_dataset_v4.py
uv run python training/scripts/modeling/train_hfce_sarima_v4.py
```

The resulting data flow is:

```text
RAW/HFCE/hfce_by_purpose.csv
  -> training/datasets/hfce_v4/hfce_category_forecast_2000_2026.csv
  -> training/models/hfce_sarima_v4/<category>.joblib
```

This pipeline contains the observed PSA category-quarter values from `2000-Q1` through `2026-Q2`. It has no FIES dependency.

---

## Survey Coverage

The all-years FIES cleaner processes these survey years:

```text
2000, 2003, 2006, 2009, 2012, 2015, 2018, 2021, 2023, 2025
```

No household data are fabricated for intervening years such as 2001, 2002, 2004, or 2005. PSA HFCE supplies the observed quarterly aggregate series for those calendar years.

---

## Category Contract

All FIES and HFCE processing uses these 12 aligned categories:

```text
food
alcohol_tobacco
clothing_footwear
housing_water_utilities
furnishings_household
health
transport
communication
recreation_culture
education
restaurants_hotels
miscellaneous_goods_services
```

The FIES source-variable mappings, their direct/proxy status, and reconciliation rule are defined in [`training/config/fies_hfce_category_mapping_v4.json`](../../config/fies_hfce_category_mapping_v4.json). The cleaner retains an unmapped expenditure residual instead of silently assigning it to a category.

---

## Optional Household Data Flow

```text
FIES raw public-use files + dictionaries
  -> header, key, amount, and relationship validation
  -> NCR-only annual household category anchors
  -> long annual_anchors data (12 rows per household)
  -> household_categories wide data (1 row per household)
  -> HFCE-weighted quarterly household estimates (4 rows per household)

PSA HFCE by purpose, current prices
  -> within-year category allocation weights
  -> household-quarter estimates for profile analysis
```

---

## 1. Clean Raw FIES Releases

Run the all-years cleaner:

```bash
uv run python training/scripts/data/clean_raw_fies.py
```

The script uses `training/scripts/data/clean_raw_fies.py` and reads FIES files and data dictionaries from `RAW/uncleaned/`.

For every supported survey year, it performs the following controls:

1. Verifies that the CSV header exactly matches the published data-dictionary field order.
2. Requires household keys, region, income, total expenditure, household size, category source fields, and year-specific validation fields.
3. Converts documented numeric fields to numeric values and rejects missing or negative required amounts.
4. Verifies unique, complete household keys.
5. Applies documented source identities, including food-total and total-expenditure checks where the release supports them.
6. Restricts output to National Capital Region (NCR) households.
7. Quarantines invalid NCR rows with a reason code instead of imputing their values.
8. Replaces raw household identifiers with a stable HMAC pseudonym scoped to the survey year.

The cleaner also validates documented one-to-one supplementary source joins where a release provides supplementary files. Supplementary files are validation inputs only; they do not add household observations.

### Raw-Cleaning Outputs

| File | Grain | Purpose |
| :--- | :--- | :--- |
| `RAW/cleaned/annual_anchors.csv` | Household, survey year, category | Long annual category anchors. |
| `RAW/cleaned/annual_anchors.parquet` | Household, survey year, category | Parquet equivalent used by the pivot step. |
| `RAW/cleaned/households.csv` | Household, survey year | Income, total expenditure, household size, region, and residual controls. |
| `RAW/cleaned/quarantined_households.csv` | Rejected NCR source row | Audit record with the quarantine reason. |
| `RAW/cleaned/run_manifest.json` | Cleaning run | Input fingerprints, validation results, provenance, and counts. |

`annual_anchors` uses one row for each household-category pair. A valid household therefore has 12 anchor rows.

---

## 2. Produce Annual Household Category Rows

Run the annual pivot:

```bash
uv run python training/scripts/data/minimize_fies_anchors.py
```

`training/scripts/data/minimize_fies_anchors.py` reads `annual_anchors.parquet`. It requires one non-missing amount for every category and fails if a household is incomplete or repeats a category. It writes:

```text
RAW/cleaned/household_categories.csv
RAW/cleaned/household_categories.parquet
```

The table has one observed annual FIES row per household:

```text
household_id,year,food,alcohol_tobacco,...,miscellaneous_goods_services
```

The current cleaning run contains 107,659 household rows. The `year` column is the FIES survey year, not a generated calendar year.

---

## 3. Build the Quarterly HFCE Forecast Dataset

Run the PSA HFCE transformation:

```bash
uv run python training/scripts/data/build_hfce_category_forecast_dataset_v4.py
```

`training/scripts/data/build_hfce_category_forecast_dataset_v4.py` reads `RAW/HFCE/hfce_by_purpose.csv`. It maps the PSA purposes to the same 12 categories and retains both published price bases.

Before writing its output, `validate_dataset()` verifies the required schema, observed-quarter frequency, valid period labels, unique category-quarter rows, category contract, non-negative price-basis values, category-share totals, identical timelines across categories, complete-year flags, and a continuous quarterly timeline with an optional partial final year.

The default output is:

```text
training/datasets/hfce_v4/hfce_category_forecast_2000_2026.csv
```

Each row represents an observed aggregate category-quarter:

```text
year_quarter,year,quarter,category,
current_price_million_php,constant_2018_price_million_php,
current_price_category_share,is_complete_year
```

Use `current_price_million_php` for nominal PHP expenditure forecasting. The constant-price series is available for real-volume analysis, not as a substitute for nominal targets.

For a category-specific quarterly SARIMA model, the input sequence is:

```text
2000-Q1 food
2000-Q2 food
2000-Q3 food
2000-Q4 food
2001-Q1 food
...
```

The annual seasonal period is `s=4` because the observed model input has four quarters per year. A three-month quarter does not make `s=3`.

---

## 4. Temporally Disaggregate Household Categories to Quarters

Run the household quarterly allocation:

```bash
uv run python training/scripts/data/disaggregate_household_categories_quarterly.py
```

The script reads:

```text
RAW/cleaned/household_categories.csv
RAW/HFCE/hfce_by_purpose.csv
```

It writes:

```text
RAW/cleaned/household_categories_quarterly.csv
```

For household `h`, category `c`, survey year `y`, and quarter `q`:

```text
quarter_weight[c, y, q] =
  HFCE_current_price[c, y, q]
  / sum(HFCE_current_price[c, y, Q1...Q4])

quarter_amount[h, c, y, q] =
  annual_fies_amount[h, c, y]
  × quarter_weight[c, y, q]
```

The allocation is category-specific. Food uses food's quarterly HFCE profile; transport uses transport's profile; no whole-household or general inflation multiplier is applied.

### Quarterly Household Output Contract

| Column | Meaning |
| :--- | :--- |
| `household_id` | Pseudonymous FIES household identifier. |
| `anchor_survey_year` | Observed FIES survey year supplying the annual household amount. |
| `year`, `quarter`, `year_quarter` | Calendar period of the allocation. |
| `observation_status` | Always `hfce_temporally_disaggregated` for this output. |
| `allocation_method` | Always `within_year_hfce_current_price_share`. |
| 12 category columns | HFCE-weighted quarterly estimates in PHP. |

For each household, category, and survey year, the following invariant must hold, apart from floating-point representation:

```text
Q1 + Q2 + Q3 + Q4 = annual FIES category amount
```

The current output contains 430,636 rows:

```text
107,659 annual households × 4 quarters = 430,636 household-quarter rows
```

It contains quarters only for the ten observed FIES survey years. It does not create household records in intervening years.

---

## Household-Level Capability

FIES is not required for aggregate SARIMA. Retain the optional FIES outputs only when BUDI needs one or more of these household-level capabilities:

| Capability | FIES contribution |
| :--- | :--- |
| Personalized budget calibration | Supplies observed NCR household category distributions that can contextualize a user's category budget against comparable household profiles. |
| Income and household-size segmentation | Supplies annual income, household size, and category expenditure controls unavailable in aggregate HFCE. |
| Equity and affordability analysis | Supports comparisons of category burden across household income and size groups instead of using one national aggregate trend for all users. |
| Household-category feature research | Supplies household-level category combinations for future classification, anomaly, or recommendation research. |
| Aggregate-to-user allocation research | Allows a forecasted national HFCE category trend to be translated into a clearly labeled household-profile estimate. |

FIES remains repeated cross-sectional survey data. It cannot establish a particular user's future spending, and its HFCE-weighted quarterly allocations are not observed household transactions.

---

## Intended Uses and Non-Claims

| Dataset | Appropriate use | Not appropriate for |
| :--- | :--- | :--- |
| `annual_anchors` and `household_categories` | Household distribution analysis, segmentation, calibration, and annual FIES benchmarking. | A regular quarterly or monthly time-series target. |
| `hfce_v4/hfce_category_forecast_2000_2026.csv` | Quarterly aggregate category SARIMA fitting, validation, and forecasting. This is the required v4 forecast input. | Household-level inference. |
| `household_categories_quarterly.csv` | Household-profile allocation, feature construction, and reconciliation checks. | Claims of observed quarterly household expenditure or SARIMA ground truth. |

The PSA HFCE source is aggregate Philippine expenditure. Its within-year proportions provide a transparent allocation signal, but they do not establish that a particular FIES household spent the estimated amount in a particular quarter.

---

## 5. Train Quarterly SARIMA Candidates

Run the category-wise trainer:

```bash
uv run python training/scripts/modeling/train_hfce_sarima_v4.py
```

The trainer uses `current_price_million_php` and fits one model per category. It keeps the series in chronological order and uses an annual quarterly seasonal period of `s=4`.

The 106 observed quarters from `2000-Q1` through `2026-Q2` are split without shuffling:

| Partition | Rows | Period |
| :--- | :---: | :--- |
| Train | 74 | `2000-Q1` through `2018-Q2` |
| Validation | 21 | `2018-Q3` through `2023-Q3` |
| Test | 11 | `2023-Q4` through `2026-Q2` |

The test partition includes the published 2026 Q1 and Q2 observations. Candidate selection uses expanding-window walk-forward validation over the 21-quarter validation window: five four-quarter forecast folds and one final one-quarter fold. Every validation and test forecast reports MAE, RMSE, MAPE, SMAPE, and MDA. MDA compares each forecasted quarter's direction with the direction from the preceding observed quarter.

The selected default is baseline SARIMA with no COVID intervention variables. COVID observations remain in the series and are not removed. SARIMAX with a temporary `covid` shock from `2020-Q1` through `2021-Q4` and a persistent `post_covid` level shift from `2022-Q1` onward remains available through `--model-family sarimax_intervention` for documented comparison experiments. These boundaries are command-line configuration and are recorded in `evaluation.json` whenever SARIMAX is run. Raw and `log1p` target transformations are compared for each selected family; forecasts are transformed back to current-price PHP before metrics are calculated. The selected candidate is refit on train plus validation for test evaluation, then refit on all available observations for its saved artifact. A candidate that does not converge during validation, test refit, or final refit is rejected; the next-best converged validation candidate is used instead.

Generated training artifacts are written to:

```text
training/models/hfce_sarima_v4/
```

The directory contains one `.joblib` artifact per category and `evaluation.json` with input fingerprint, split dates, candidate validation results, selected orders, and held-out test metrics. `metric_averages` reports unweighted macro averages across the 12 selected category models for both validation and test metrics. These are training artifacts, not deployed serving artifacts.

---

## Verification

Run the focused tests after changing the pipeline:

```bash
uv run pytest tests/test_clean_raw_fies_region.py \
  tests/test_fies_category_anchors.py \
  tests/test_minimize_fies_anchors.py \
  tests/test_hfce_category_forecast_dataset.py \
  tests/test_disaggregate_household_categories_quarterly.py
```

Run style checks:

```bash
uv run ruff check training/scripts/data/clean_raw_fies.py \
  training/scripts/data/minimize_fies_anchors.py \
  training/scripts/data/build_hfce_category_forecast_dataset_v4.py \
  training/scripts/data/disaggregate_household_categories_quarterly.py \
  training/scripts/modeling/train_hfce_sarima_v4.py
```

Manual checks:

1. Inspect `run_manifest.json` for input fingerprints and zero unexpected validation failures.
2. Confirm that every wide annual household row has 12 category values.
3. Confirm that every household-quarter row has four quarters for its `anchor_survey_year`.
4. Sum each household-category's Q1-Q4 values and confirm that it equals the matching annual value.
5. Confirm that SARIMA inputs use the observed quarterly HFCE dataset rather than disaggregated household estimates.
