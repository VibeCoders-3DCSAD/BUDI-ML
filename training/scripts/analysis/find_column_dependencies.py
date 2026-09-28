"""
find_column_dependencies.py

Detects sum/identity relationships between numeric columns in the FIES dataset
(e.g. TOINC = REG_SAL + SEASON_SAL + ... + OTHER_SOURCE), using the FIES
metadata dictionary (and, when available, the valueset) to restrict the
search space and to exclude ordinal columns (deciles) that look numeric but
aren't additive.

APPROACH (v2)
-------------
v1 of this script fit each candidate target column via ordinary least squares
against *every other* numeric column at once. That works fine when a target's
components are otherwise unrelated to everything else -- but real survey data
has *nested* totals (e.g. 16 line items sum to FOOD_HOME, FOOD_HOME +
FOOD_OUTSIDE sum to FOOD, FOOD + NFOOD sum to TOTEX). When a total and its own
components (or another overlapping total) are all thrown into one regression
as predictors simultaneously, the system becomes exactly multicollinear, and
plain least-squares (`lstsq`) returns an arbitrary *minimum-norm* solution
that smears credit across many correlated columns instead of cleanly
identifying the true small subset. On the real FIES data this turned nearly
every food/non-food category into an "ambiguous" mess even though the true
relationships were simple and clean.

v2 fixes this with three changes:

  1. **Decile/ordinal exclusion.** Any column whose dictionary label (or
     valueset label) contains "Decile" or similar rank-like language is
     excluded entirely -- these are ordinal categories stored as numbers
     (1-10), not additive quantities, and were never valid targets for a
     sum-search.

  2. **Sparse subset selection instead of full-predictor OLS.** For each
     candidate target, this version uses the LARS path (least-angle
     regression) to find the order in which predictors would enter a sparse
     model, then tests increasingly larger prefixes of that path with a
     plain OLS refit (no regularization bias) until R² clears the threshold.

  3. **Greedy round-by-round resolution.** Each round, only the single most
     confident (smallest-subset) result is locked in and then excluded from
     every other column's predictor pool. This stops a genuine multi-item
     sum (e.g. FOOD_HOME = 16 line items) from getting "resolved" as a
     shortcut through some other overlapping total (e.g. FOOD - FOOD_OUTSIDE)
     just because the shortcut happened to be a smaller subset.

DICTIONARY / VALUESET INPUT
----------------------------
The --dictionary argument accepts either:
  - a .csv export of just the dictionary sheet (valueset labels are usually
    mangled or lost when exporting FIES workbooks to CSV, so decile
    detection relies on dictionary labels alone in this mode), or
  - a .xlsx workbook containing both a dictionary sheet and a valueset sheet
    (sheet names auto-detected by matching "dictionary"/"valueset",
    case-insensitive). In this mode, both the dictionary labels *and* the
    valueset entry labels are checked for ordinal/decile language, which is
    the more reliable option -- use this when you have it.

Only numeric, non-decile columns are considered.

USAGE
-----
    python find_column_dependencies.py \
        --data /path/to/Family_Income_and_Expenditure.csv \
        --dictionary /path/to/fies_2023_dictionary_and_valueset.xlsx \
        --output dependency_report.md

If --dictionary is omitted, section grouping and decile exclusion are both
skipped (all numeric columns are treated as one section, and none are
excluded as ordinal) -- only recommended for a small number of columns, or
when you've already pre-filtered the data yourself.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import lars_path
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

ORDINAL_KEYWORDS = ("decile", "rank", "quintile", "quartile", "percentile")


def _find_sheet(sheet_names, hint):
    """Return the first sheet name containing `hint` (case-insensitive)."""
    for name in sheet_names:
        if hint.lower() in name.lower():
            return name
    return None


def _rows_from_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [row for row in csv.reader(f)]


def _rows_from_xlsx_sheet(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=str)
    df = df.fillna("")
    return df.to_numpy().tolist()


def _parse_dictionary_rows(rows):
    """Shared parser for the 6-column hierarchical dictionary layout
    (section code/label in cols 2/3, variable code/label in cols 4/5),
    whether it came from a CSV or an Excel sheet."""
    sections = defaultdict(list)
    decile_columns = set()
    current_section = "UNSECTIONED"

    for row in rows:
        row = list(row) + [""] * (6 - len(row))  # pad short rows
        section_code = str(row[2]).strip()
        var_code = str(row[4]).strip()
        label = str(row[5]).strip()

        if section_code and section_code.lower() != "nan":
            current_section = section_code
            continue

        if var_code and var_code.lower() != "nan":
            sections[current_section].append(var_code)
            if any(kw in label.lower() for kw in ORDINAL_KEYWORDS):
                decile_columns.add(var_code)

    return dict(sections), decile_columns


def _parse_valueset_rows(rows):
    """
    Parse the FIES valueset sheet/CSV as a second signal for ordinal
    columns. Header rows look like:
        col0 = "<VARCODE>_VS1", col1 = "<label>"
    followed by value rows (col2=value label, col3=value code). A variable
    is flagged ordinal here if its valueset label matches an ordinal
    keyword.
    """
    decile_columns = set()
    for row in rows:
        row = list(row) + [""] * (2 - len(row)) if len(row) < 2 else row
        code_field = str(row[0]).strip() if len(row) > 0 else ""
        label_field = str(row[1]).strip() if len(row) > 1 else ""
        if code_field and code_field.lower() != "nan" and code_field.endswith("_VS1"):
            var_code = code_field[:-len("_VS1")]
            if any(kw in label_field.lower() for kw in ORDINAL_KEYWORDS):
                decile_columns.add(var_code)
    return decile_columns


def parse_dictionary_sections(dict_path, valueset_path=None,
                               dictionary_sheet_hint="dictionary",
                               valueset_sheet_hint="valueset"):
    """
    Parse the FIES metadata dictionary and return (sections, decile_columns).

    Accepts either:
      - a .csv path for the dictionary (valueset info unavailable, since the
        valueset historically gets mangled when exported to CSV), or
      - a .xlsx path containing both a dictionary sheet and a valueset
        sheet (auto-detected by name), which additionally cross-checks
        ordinal/decile columns against the valueset labels.

    If dict_path is an .xlsx and valueset_path is not given separately, the
    valueset sheet is read from the same workbook automatically.
    """
    ext = os.path.splitext(dict_path)[1].lower()

    if ext in (".xlsx", ".xls"):
        xls = pd.ExcelFile(dict_path)
        dict_sheet = _find_sheet(xls.sheet_names, dictionary_sheet_hint) or xls.sheet_names[0]
        dict_rows = _rows_from_xlsx_sheet(dict_path, dict_sheet)
        sections, decile_columns = _parse_dictionary_rows(dict_rows)

        valueset_sheet = _find_sheet(xls.sheet_names, valueset_sheet_hint)
        if valueset_sheet:
            valueset_rows = _rows_from_xlsx_sheet(dict_path, valueset_sheet)
            decile_columns |= _parse_valueset_rows(valueset_rows)
        return sections, decile_columns

    # CSV path (dictionary only; valueset typically not usable from CSV)
    dict_rows = _rows_from_csv(dict_path)
    sections, decile_columns = _parse_dictionary_rows(dict_rows)

    if valueset_path:
        vs_ext = os.path.splitext(valueset_path)[1].lower()
        if vs_ext in (".xlsx", ".xls"):
            xls = pd.ExcelFile(valueset_path)
            valueset_sheet = _find_sheet(xls.sheet_names, valueset_sheet_hint) or xls.sheet_names[0]
            valueset_rows = _rows_from_xlsx_sheet(valueset_path, valueset_sheet)
        else:
            valueset_rows = _rows_from_csv(valueset_path)
        decile_columns |= _parse_valueset_rows(valueset_rows)

    return sections, decile_columns


def fit_sparse_subset(df, target_col, predictor_cols, r2_threshold=0.99,
                       include_coef=(0.9, 1.1), exclude_coef=(-0.1, 0.1),
                       max_subset_size=20, corr_threshold=0.1):
    """
    Find the smallest subset of predictor_cols whose OLS (no-intercept) fit
    to target_col clears r2_threshold, using the LARS path to order
    candidates instead of dumping everything into one regression at once.

    Returns a dict describing the best subset found, or None.
    """
    if not predictor_cols or not HAVE_SKLEARN:
        return None

    y_raw = df[target_col].to_numpy(dtype=float)
    y_valid = ~np.isnan(y_raw)

    # Pre-filter: drop predictors whose absolute correlation with the
    # target is below corr_threshold.  A sum identity T = A + B + …
    # requires each component to share some linear signal with T, so
    # near-zero-correlation predictors cannot participate and only slow
    # LARS down.
    if len(predictor_cols) > 30:
        corrs = np.array([
            np.abs(np.corrcoef(df[c].to_numpy(dtype=float)[y_valid],
                               y_raw[y_valid])[0, 1])
            if y_valid.sum() > 2 else 0.0
            for c in predictor_cols
        ])
        keep = corrs >= corr_threshold
        predictor_cols = [c for c, k in zip(predictor_cols, keep) if k]
        if not predictor_cols:
            return None

    X = df[predictor_cols].to_numpy(dtype=float)
    y = y_raw

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]
    if len(y) < max(30, len(predictor_cols) * 3):
        return None

    # Standardize predictors for a stable LARS path (coefficients are
    # re-fit via plain OLS afterward, so this doesn't affect final values).
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X.mean(axis=0)) / X_std

    try:
        _, _, coef_path = lars_path(X_norm, y - y.mean(), method="lasso",
                                     max_iter=max_subset_size)
    except Exception:
        return None

    # Order in which each predictor first becomes non-zero along the path
    entry_order = []
    seen = set()
    for step in range(coef_path.shape[1]):
        nz = np.nonzero(coef_path[:, step])[0]
        for idx in nz:
            if idx not in seen:
                seen.add(idx)
                entry_order.append(idx)

    best = None
    for k in range(1, min(len(entry_order), max_subset_size) + 1):
        idxs = entry_order[:k]
        sub_cols = [predictor_cols[i] for i in idxs]
        Xk = df[sub_cols].to_numpy(dtype=float)[mask]
        coefs, _, _, _ = np.linalg.lstsq(Xk, y, rcond=None)
        y_pred = Xk @ coefs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if r2 >= r2_threshold:
            included, excluded_, ambiguous = [], [], []
            for col, c in zip(sub_cols, coefs):
                abs_c = abs(c)
                if include_coef[0] <= abs_c <= include_coef[1]:
                    included.append((col, c))
                elif exclude_coef[0] <= abs_c <= exclude_coef[1]:
                    excluded_.append((col, c))
                else:
                    ambiguous.append((col, c))
            is_clean = len(ambiguous) == 0 and len(included) >= 1
            best = {
                "target": target_col,
                "r2": r2,
                "included": included,
                "ambiguous": ambiguous,
                "n_rows": int(mask.sum()),
                "subset_size": k,
                "is_clean_identity": is_clean,
            }
            if is_clean:
                break  # smallest clean subset that clears the threshold
            # Otherwise keep growing — a larger subset may resolve cleanly.

    return best


def analyze_section(df, section_name, columns, decile_columns,
                     min_predictors=2, max_predictors=60, max_rounds=200):
    """
    Test every numeric, non-decile column in a section as a candidate
    'total' of the rest, resolving one column per round -- always the
    most confident (smallest subset) candidate available -- and excluding
    it from every other column's predictor pool afterward.

    This ordering matters: without it, a column that's part of a genuine
    multi-item sum can get 'resolved' via a shortcut through some other
    total sitting in the same section (e.g. PARENT = GRANDPARENT -
    OTHER_PARENT) instead of its real components, simply because the
    shortcut happened to be tested/found first. Always locking in the
    single smallest-subset result each round guarantees higher-level
    totals (which tend to resolve via very small subsets of other totals)
    get excluded before their components are tested, so the components
    are then forced to resolve via their real (larger) subset instead.
    """
    numeric_cols = [c for c in columns if c in df.columns
                     and pd.api.types.is_numeric_dtype(df[c])
                     and c not in decile_columns]
    if len(numeric_cols) < min_predictors + 1:
        return []

    if len(numeric_cols) > max_predictors:
        print(f"  Warning: section '{section_name}' has {len(numeric_cols)} "
              f"eligible numeric columns (> {max_predictors}); analysis may "
              f"be slow. Consider raising --max-predictors.", file=sys.stderr)

    resolved = {}
    excluded_predictors = set()
    unresolved = set(numeric_cols)

    for rnd in range(max_rounds):
        if not unresolved:
            break
        candidates = {}
        for target in unresolved:
            predictor_pool = [c for c in numeric_cols
                               if c != target and c not in excluded_predictors]
            result = fit_sparse_subset(df, target, predictor_pool)
            if result and result["is_clean_identity"]:
                candidates[target] = result

        if not candidates:
            break  # nothing left resolves cleanly against the reduced pool

        best_target = min(candidates, key=lambda t: candidates[t]["subset_size"])
        best_result = candidates[best_target]
        best_result["section"] = section_name
        resolved[best_target] = best_result
        unresolved.discard(best_target)
        excluded_predictors.add(best_target)
        print(f"  Round {rnd+1}: locked {best_target} "
              f"(subset={best_result['subset_size']}, "
              f"R²={best_result['r2']:.4f}), "
              f"{len(unresolved)} unresolved, "
              f"{len(candidates)} candidates")

    # Best-effort visibility pass for anything left unresolved. These never
    # counted as clean during the main rounds (that's why they're still
    # unresolved) -- force is_clean_identity=False regardless of what the
    # lower internal threshold found, so they're always reported as
    # ambiguous, never misfiled into the high-confidence section.
    leftover_results = []
    for target in unresolved:
        predictor_pool = [c for c in numeric_cols
                           if c != target and c not in excluded_predictors]
        result = fit_sparse_subset(df, target, predictor_pool, r2_threshold=0.90)
        if result:
            result["section"] = section_name
            result["is_clean_identity"] = False
            leftover_results.append(result)

    return list(resolved.values()) + leftover_results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to the FIES data CSV")
    ap.add_argument("--dictionary",
                     help="Path to the FIES dictionary (.csv), or a .xlsx "
                          "workbook containing both the dictionary sheet and "
                          "the valueset sheet (recommended -- CSV exports "
                          "tend to lose/mangle the valueset)")
    ap.add_argument("--valueset",
                     help="Optional separate path to the valueset (.csv or "
                          ".xlsx), if --dictionary is a plain CSV and you "
                          "have the valueset as a separate file")
    ap.add_argument("--output", default="dependency_report.md",
                     help="Output markdown report path")
    ap.add_argument("--r2-threshold", type=float, default=0.99)
    ap.add_argument("--max-predictors", type=int, default=60,
                     help="Max numeric columns per section before skipping "
                          "that section's search")
    ap.add_argument("--max-subset-size", type=int, default=20,
                     help="Largest subset size to test per target column")
    args = ap.parse_args()

    if not HAVE_SKLEARN:
        print("This version requires scikit-learn (pip install scikit-learn "
              "--break-system-packages). Falling back is not supported.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.data, low_memory=False)
    numeric_df_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    print(f"Loaded {len(df)} rows, {len(numeric_df_cols)} numeric columns "
          f"(of {len(df.columns)} total).")

    if args.dictionary:
        sections, decile_columns = parse_dictionary_sections(
            args.dictionary, valueset_path=args.valueset)
        sections = {s: cols for s, cols in sections.items()
                    if any(c in numeric_df_cols for c in cols)}
        excluded_present = decile_columns & set(numeric_df_cols)
        print(f"Parsed {len(sections)} sections from dictionary.")
        print(f"Excluding {len(excluded_present)} ordinal/decile columns: "
              f"{sorted(excluded_present)}")
    else:
        print("No dictionary provided; treating all numeric columns as one "
              "section and excluding nothing as ordinal.")
        sections = {"ALL": numeric_df_cols}
        decile_columns = set()

    all_results = []
    for section_name, cols in sections.items():
        all_results.extend(analyze_section(df, section_name, cols, decile_columns,
                                            max_predictors=args.max_predictors))

    clean = [r for r in all_results if r.get("is_clean_identity")]
    ambiguous = [r for r in all_results
                 if r.get("target") and not r.get("is_clean_identity")
                 and r["r2"] >= 0.90]
    skipped = [r for r in all_results if r.get("target") is None]

    with open(args.output, "w") as f:
        f.write("# FIES Column Dependency Report (v2 -- sparse subset selection)\n\n")
        f.write(f"Data: `{args.data}`  \n")
        f.write(f"Rows: {len(df)}, Numeric columns: {len(numeric_df_cols)}, "
                 f"Excluded as ordinal: {len(decile_columns & set(numeric_df_cols))}\n\n")

        f.write("## High-Confidence Identities (R² ≥ {:.2f}, clean 0/1 coefficients, "
                 "minimal subset)\n\n".format(args.r2_threshold))
        if not clean:
            f.write("_None found._\n\n")
        for r in sorted(clean, key=lambda x: (x["subset_size"], -x["r2"])):
            parts = [f"- {c}" if coef < 0 else c for c, coef in r["included"]]
            included_str = " + ".join(parts).replace(" + - ", " - ").lstrip("+ ")
            f.write(f"- **`{r['target']}`** = {included_str}  \n"
                     f"  (section: {r['section']}, R²={r['r2']:.4f}, "
                     f"subset size={r['subset_size']}, n={r['n_rows']})\n")
        f.write("\n")

        f.write("## Ambiguous / Partial Relationships (R² ≥ 0.90, not clean)\n\n")
        f.write("These likely indicate a related-but-not-identical relationship "
                 "and should be reviewed manually.\n\n")
        if not ambiguous:
            f.write("_None found._\n\n")
        for r in sorted(ambiguous, key=lambda x: -x["r2"]):
            parts = [f"{c}={c_val:.2f}" for c, c_val in r["included"] + r["ambiguous"]]
            f.write(f"- `{r['target']}` (section: {r['section']}, R²={r['r2']:.4f}, "
                     f"subset size={r['subset_size']}): {', '.join(parts)}\n")
        f.write("\n")

        if skipped:
            f.write("## Skipped Sections\n\n")
            for r in skipped:
                f.write(f"- {r['note']}\n")
            f.write("\n")

    print(f"\nReport written to {args.output}")
    print(f"  {len(clean)} clean identities, {len(ambiguous)} ambiguous relationships, "
          f"{len(skipped)} sections skipped.")


if __name__ == "__main__":
    main()
