# FIES Column Dependency Report (v2 -- sparse subset selection)

Data: `training/datasets/raw/puf.csv`  
Rows: 41380, Numeric columns: 90, Excluded as ordinal: 5

## High-Confidence Identities (R² ≥ 0.99, clean 0/1 coefficients, minimal subset)

- **`WAGES`** = REG_SAL + SEASON_SAL  
  (section: HHLD_SUMMARY, R²=1.0000, subset size=2, n=41380)
- **`TOTEX`** = NFOOD + FOOD  
  (section: HHLD_SUMMARY, R²=1.0000, subset size=2, n=41380)
- **`FOOD`** = FOOD_HOME + FOOD_OUTSIDE  
  (section: HHLD_SUMMARY, R²=1.0000, subset size=2, n=41380)
- **`TOINC`** = TOREC - OTHREC  
  (section: HHLD_SUMMARY, R²=1.0000, subset size=2, n=41380)
- **`RENTVAL`** = IMPUTED_RENT + ACTRENT + BIMPUTED_RENT  
  (section: HHLD_SUMMARY, R²=1.0000, subset size=4, n=41380)
- **`TOTDIS`** = NFOOD + OTHER_DISBURSEMENT + FOOD_HOME + FOOD_OUTSIDE  
  (section: HHLD_SUMMARY, R²=1.0000, subset size=5, n=41380)
- **`NFOOD`** = HOUSING_WATER + TRANSPORT + COMMUNICATION + OTHER_EXPENDITURE + INSURANCE + EDUCATION + FURNISHING + HEALTH + MISCELLANEOUS + CLOTH + DURABLE + RECREATION + OCCASION  
  (section: HHLD_SUMMARY, R²=0.9988, subset size=14, n=41380)
- **`MEAT`** = FOOD_HOME - FOOD_NEC - MILK - VEG - BREAD - FRUIT - FRUIT_VEG - COFFEE - SUGAR - TEA - COCOA - FISH - WATER - OIL  
  (section: HHLD_SUMMARY, R²=0.9944, subset size=14, n=41380)
<!-- ADD: High-confidence identity of MEAT is rejected. -->

## Ambiguous / Partial Relationships (R² ≥ 0.90, not clean)

These likely indicate a related-but-not-identical relationship and should be reviewed manually.

- `TOREC` (section: HHLD_SUMMARY, R²=0.9784, subset size=20): PERCAPITA=0.16, REG_SAL=0.74, HOUSING_WATER=0.27, OTHER_DISBURSEMENT=0.21, TRANSPORT=0.31, FOOD_HOME=0.47, EAINC=0.76, INSURANCE=0.41, FURNISHING=0.34, MISCELLANEOUS=0.57, OTHER_EXPENDITURE=0.26, OTHREC=0.81, EDUCATION=0.51, HEALTH=0.69, FOOD_OUTSIDE=0.40, IMPUTED_RENT=0.75, DIVIDENDS=0.73, CASH_ABROAD=0.72, PENSION=0.76
- `FOOD_HOME` (section: HHLD_SUMMARY, R²=0.9549, subset size=20): VEG=1.05, MILK=1.05, FOOD_NEC=1.04, COFFEE=1.08, FISH=1.53, BREAD=1.16, FRUIT=1.18, OIL=1.74, SUGAR=1.19, FSIZE=921.42, SOFTDRINKS=1.92, WATER=1.16, FRUIT_VEG=1.12, COCOA=1.14
- `EAINC` (section: HHLD_SUMMARY, R²=0.9413, subset size=3): NET_RET=1.01, NET_NEC_A8=1.01, NET_TRANS=1.00
- `IMPUTED_RENT` (section: HHLD_SUMMARY, R²=0.9224, subset size=20): HOUSING_WATER=0.74, ACTRENT=-0.87, COMMUNICATION=-0.33, RENTALS_REC=-0.10, OIL=-0.75, SUGAR=-0.13
- `HOUSING_WATER` (section: HHLD_SUMMARY, R²=0.9114, subset size=3): IMPUTED_RENT=1.07, ACTRENT=1.09

