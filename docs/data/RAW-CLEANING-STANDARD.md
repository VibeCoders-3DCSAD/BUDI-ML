# FIES RAW Cleaning Standard

This standard applies to each year-specific cleaning specification in `RAW/uncleaned/<year>/CLEANING_RULES.md`. Year-specific source definitions and mappings remain authoritative for that survey year. Apply this shared contract in addition to, not instead of, each year's rules.

## Household identity and privacy

- Treat each survey year as an independent household sample. Never link or infer that households in different years are the same household.
- Keep original survey keys only in the local restricted intermediate needed for joins and validation. Do not publish raw keys.
- Create output IDs that are year-scoped and unlinkable across years. Use HMAC-SHA-256 with a secret supplied outside the repository/environment; serialize fields in a documented stable order and include the survey year. Do not use an unkeyed hash of survey identifiers.
- Within a year, use the exact validated key documented for that source release. Do not join by row order. Validate uniqueness and join cardinality before merging.

## Region-code normalization

- Preserve the source region field and value for traceability. Read region codes as strings so formatting such as `01` is not lost through numeric conversion.
- For every checked FIES household-summary dataset from 2000 through 2025, region code `13` identifies the National Capital Region. Normalize both source value forms (`13` and zero-padded/string representations where present) to canonical code `13` and canonical label `NCR`.
- Keep separate fields for the normalized code and label (for example `region_code="13"`, `region="NCR"`). Do not confuse NCR's code `13` with any province or subregion field; normalize only the documented region field (`REGN`, `REGION`, or `W_REGN` as appropriate to the release).
- Retain only National Capital Region households in the normalized anchor and household outputs. After the region field is normalized to a two-digit code, keep rows whose code is `13` and drop every other region. Record the excluded count in the run manifest. This is a sample cut, not a quarantine: non-NCR rows are valid source households outside the retained geography.
- The DCF dictionaries use labels including `NCR`, `Region XIII - NCR`, and `National Capital Region`; retain each release's original label in provenance if labels are loaded from dictionaries.

## Shared normalized output contract

The normalized annual anchor table must include:

```text
household_id                 # year-scoped pseudonymous ID
survey_year
source_release
source_file
period                       # annual for observed FIES anchors
category
amount_php
observation_status           # observed | synthetic
coverage_status              # direct | proxy | partial | unresolved, as applicable
allocation_method            # null for direct observed annual anchors
cleaning_spec_version
```

The linked run manifest/reconciliation table must preserve exact source variable(s), source key scope (never raw keys in exported artifacts), input file size and SHA-256, source dictionary, cleaning-spec version, row-level validation flags/reason codes, reconciliation values/tolerances, and summary counts. Keep source values immutable; normalized values must be traceable to source fields and transformations.

`amount_php` values for source FIES anchors are annual household PHP. Do not imply that category rows are mutually exclusive where year-specific evidence does not establish non-overlap. `TOTEX`/equivalent remains a control, not a category.

## Run disposition and validation policy

- **Fail the run** for unexpected input path/hash/size, header or dictionary mismatch, malformed structure, missing/duplicate required household keys, or invalid join cardinality. Do not silently choose another file or fall back to row-order matching.
- **Quarantine a row** when a required anchor is missing or non-finite, an amount violates a source/year-specific impossibility rule, or a required relationship fails beyond its documented tolerance. Store a reason code and source-row locator in the restricted audit output; do not silently drop it.
- **Retain with a flag** when a documented nonfatal reconciliation differs within its documented tolerance or the source rule explicitly allows unresolved coverage. Preserve original values; tolerances are for validation only and never authorize edits.
- Do not impute missing category amounts or replace them with zero unless a field-specific source convention is documented and verified.
- Every run must report input, retained, quarantined, and output row counts plus counts for each validation rule. Report exact PHP tolerances for the relevant year/relationship. Keep exception records available for review.

## Observed and synthetic temporal data

- Observed FIES household anchors are annual. Keep them as the authoritative source table.
- Any quarter/month derived from annual FIES using HFCE profiles is synthetic. Store it separately from annual observed anchors and set `observation_status=synthetic`, the period, exact allocation method, and HFCE source release/version/fingerprint.
- Never describe allocated values as observed household histories. Do not use these allocations as supervised forecasting labels unless a separate documented decision approves the use and defines leakage-safe evaluation. All synthetic periods derived from a given household-year must remain in the same train/validation/test partition.

## Reproducibility preflight

Before transforming data, verify exact selected input paths, byte sizes, SHA-256 hashes, expected headers against the exact volume dictionary, expected row counts where specified, key uniqueness, duplicate-source identity, join cardinality, and documented year-specific identities/relationships. Existing prose validation results are expected baselines, not substitutes for re-running checks against actual inputs. Fail on unexpected source changes and write a machine-readable run manifest with each output.

The cleaning script and manifest are implementation deliverables to create after the year-specific mappings and this shared standard are finalized.
