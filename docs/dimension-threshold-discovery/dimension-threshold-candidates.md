# Dimension & Threshold Candidates

**Phase:** 4.5 — Dimension & Threshold Discovery
**Status:** Provisional — ready for SME sanity-check; numeric values pending real user data

---

## Current Thresholds (SME Draft)

| Dimension | Threshold | Rationale |
|-----------|-----------|-----------|
| Stability | CV < 0.5 = Stable | SME draft: moderate income consistency cutoff |
| Obligation | ratio > 0.6 = Obligated | SME draft: essential obligations exceed 60% of expenses |
| Tolerance | runway >= 3 months = Tolerant | SME draft: 3-month buffer before depletion |

---

## Dimension Distributions (Synthetic Data)

### income_stability_cv

- Mean: 0.3874
- Std: 0.2888
- Median: 0.3595
- IQR: [0.1050, 0.6630]
- Range: [0.0000, 0.9200]

### obligation_ratio

- Mean: 0.8800
- Std: 0.0397
- Median: 0.8810
- IQR: [0.8550, 0.9070]
- Range: [0.5870, 1.0280]

### runway_months

- Mean: 0.7884
- Std: 1.2686
- Median: 0.2200
- IQR: [0.0000, 1.0325]
- Range: [0.0000, 9.7200]

---

## Clustering Analysis

**Binary split holds (k=2 sufficient):** True

**K-Means silhouette scores:**

- k=2: silhouette=0.3815, inertia=7381.9
- k=3: silhouette=0.4214, inertia=5237.6
- k=4: silhouette=0.3914, inertia=4157.0
- k=5: silhouette=0.3744, inertia=3241.0
- k=6: silhouette=0.3876, inertia=2844.9

**GMM BIC scores:**

- k=2: BIC=14635.3, AIC=14517.8
- k=3: BIC=12644.2, AIC=12464.7
- k=4: BIC=11506.5, AIC=11265.1
- k=5: BIC=10578.9, AIC=10275.6
- k=6: BIC=6722.4, AIC=6357.2

---

## Recommendations

1. **Binary splits are sufficient** if clustering silhouette for k=2 is competitive with higher k values.
2. **Threshold values need validation** against real user transaction data — current values are SME-informed estimates.
3. **Financial Trajectory and Financial Margin** are overlay indicators, not classifying dimensions.
4. **Circularity caveat:** All patterns here reflect the synthetic generation parameters, not real-world distributions.

---

*This document is provisional. Final thresholds require SME review + real user data.*