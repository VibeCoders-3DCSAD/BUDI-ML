# RAW Pre-Cleaning Rules — Final Audit

**Date:** 2026-09-24
**Scope:** `RAW/uncleaned/*/CLEANING_RULES.md` (2000–2025), the 2018/2021/2023/2025 relationship matrices, and the source/output conventions documented there. This audits the rules before actual data cleaning; it is not a validation run against every source row.

## Overall Score

| Category | ERRORs | WARNINGs |
|---|---:|---:|
| Data Integrity | 0 | 1 |
| Code Quality / Reproducibility | 0 | 0 |
| **Total** | **0** | **1** |

## MEDIUM — Preserve This Limitation in Downstream Use

### 1. FIES 2018 `TFOOD` and `TRESTAURANT` are retained published anchors, but not a proven additive partition

**Severity:** WARNING — Data Integrity
**Files:** `RAW/uncleaned/2018/CLEANING_RULES.md:21-27,36-47,61`; `RAW/uncleaned/2018/RELATIONSHIP_MATRIX.md:18-21,43-45`

The year-specific rules now use the requested published anchors: `food = TFOOD` and `restaurants_hotels = TRESTAURANT`. They retain `TFOODOUTSIDE` for audit and explicitly do not claim that these two anchors are mutually exclusive. This resolves the contradictory subtraction rule, while leaving a known limitation: the two values must not be summed or presented as a reconciled partition of `TTOTEX` without additional evidence.

**Handling:** Keep both anchors as specified. Retain the unresolved coverage/status in the output contract. Do not sum them for a total-expenditure check or describe their combined value as a mutually exclusive category total.

## LOW — Improve Reproducibility

### 6. Rules report prior validation results but do not require the cleaning run to reproduce them

**Severity:** WARNING — Code Quality / Reproducibility
**Files:** All `RAW/uncleaned/<year>/CLEANING_RULES.md`; relationship matrices for 2018, 2021, 2023, and 2025

The documents include useful header counts, row counts, joins, and relationship counts, but these are recorded as prose and the cleaning rules do not consistently require re-running those checks against the precise files actually consumed. Source directories also contain duplicate archive copies and inconsistent filenames (for example 2006 and 2009 duplicate extracted directories, and the 2000 Volume 2 CSV carries a 2002 name). A later replacement or accidental file selection could invalidate the reported evidence without being detected.

**Fix:** Make the run verify exact selected input paths, byte sizes and SHA-256 hashes; check header-to-dictionary identity, row counts, key uniqueness, duplicate-source identity, join cardinality, and documented field relationships. Fail on unexpected input changes and save a machine-readable run manifest with the outputs. Treat existing validation counts as expected baselines, not substitutes for these checks.

## Files With Zero Violations

The shared contract is in `docs/data/RAW-CLEANING-STANDARD.md` and is linked from each year-specific rule file. It defines year-scoped IDs, output/provenance fields, failure/quarantine/flag dispositions, synthetic-data constraints, and a reproducibility preflight. The 2018 overlap caveat remains a limitation for additive analysis, not a contradictory cleaning rule.

## Things that we're done correctly

- Every year preserves the key distinction between annual household FIES observations and observed monthly/quarterly history, and the rules generally label HFCE-based temporal shaping as synthetic.
- The rules consistently warn against converting missing values to zero, adding parent totals to their detail/cash/in-kind components, or using HFCE million-PHP levels as household PHP amounts.
- The 2018, 2021, 2023, and 2025 relationship matrices document year-specific evidence and uncertainty instead of assuming mappings transfer unchanged between survey releases.
- The 2021 and 2023 rules identify duplicate summary files and prevent them from becoming duplicate household observations; the 2015 and 2018 evidence likewise avoids row-order joins and records cross-volume discrepancies.
- The 2025 rules correctly withhold claims that unavailable detailed data can establish a strict restaurant/hotel or lodging amount.
- Household IDs are now explicitly year-scoped; no cross-year household matching is intended.
- A shared standard now makes output provenance, row disposition, synthetic-data handling, and later script preflight requirements consistent across all ten survey-year specifications.
- The source region field is preserved as text, and the verified code `13` is normalized to `NCR` across the 2000–2025 household summaries.

## Priority Fix Roadmap

### P0 — Before writing/using the cleaning script

| # | Action | Files |
|---|---|---|
| 1 | Implement the shared preflight, HMAC ID generation, output contract, quarantine/flag reports, and machine-readable manifest. | `docs/data/RAW-CLEANING-STANDARD.md`, future cleaning script |
| 2 | Ensure the 2018 output preserves unresolved food/restaurant coverage and never asserts the anchors form an additive partition. | `RAW/uncleaned/2018/CLEANING_RULES.md`, future cleaning script |

### P1 — Before training/evaluation

| # | Action | Files |
|---|---|---|
| 3 | Recompute documented validations from exact hashed inputs; keep synthetic allocations separate and use leakage-safe splits. | All year rule docs / future cleaning script and downstream evaluation |

## Final Note

Audit date: 2026-09-24; updated after rule decisions on 2026-09-24. Heuristic N+1 analysis was not applicable; this audit covers data-cleaning rules and source-data contracts, not runtime database or request code. The cleaning script is intentionally deferred until the rules are finalized.
