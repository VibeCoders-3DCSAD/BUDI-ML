#!/usr/bin/env python3
"""Odin ML — Exploratory Data Analysis Script.

Generates a comprehensive EDA report with static plots and a markdown summary.
Covers: distributions, correlations, class balance, temporal patterns, anomaly
analysis, and data quality for the engineered feature matrices.

Usage:
    python scripts/eda.py --input datasets/engineered/ --output figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9)

ENGINEERED_FEATURES = [
    "income_stability_cv", "obligation_ratio", "savings_rate", "debt_to_income",
    "discretionary_ratio", "income_trend", "expense_trend", "volatility_index",
    "category_entropy", "transaction_frequency", "avg_transaction_size",
    "income_regularity", "expense_regularity", "income_expense_gap",
    "essential_income_ratio",
]

RAW_FEATURES = [
    "total_income", "total_expenses", "food_expense", "housing_expense",
    "transport_expense", "health_expense", "education_expense", "other_expense",
    "savings", "debt_payment", "transaction_count",
]

META_COLUMNS = ["user_id", "month", "pfp_label", "is_anomalous", "anomaly_type"]

PFP_CLASSES = [
    "Stable/Flexible/Tolerant", "Stable/Flexible/At-Risk",
    "Stable/Obligated/Tolerant", "Stable/Obligated/At-Risk",
    "Variable/Flexible/Tolerant", "Variable/Flexible/At-Risk",
    "Variable/Obligated/Tolerant", "Variable/Obligated/At-Risk",
]

PFP_PALETTE = {
    "Stable/Flexible/Tolerant": "#2ecc71",
    "Stable/Flexible/At-Risk": "#1f618d",
    "Stable/Obligated/Tolerant": "#3498db",
    "Stable/Obligated/At-Risk": "#8e44ad",
    "Variable/Flexible/Tolerant": "#f1c40f",
    "Variable/Flexible/At-Risk": "#e67e22",
    "Variable/Obligated/Tolerant": "#e74c3c",
    "Variable/Obligated/At-Risk": "#7f8c8d",
}

DPI = 100


def _available(df, features):
    """Return only the features that exist as columns in df."""
    return [f for f in features if f in df.columns]


def load_data(input_dir: str) -> dict[str, pd.DataFrame]:
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(input_dir, f"{name}.parquet")
        if os.path.exists(path):
            splits[name] = pd.read_parquet(path)
    return splits


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_feature_schema(input_dir: str) -> tuple[list[str], list[str]]:
    path = os.path.join(input_dir, "feature_columns.json")
    if os.path.exists(path):
        try:
            meta = load_json(path)
            feats = meta.get("feature_columns")
            raws = meta.get("raw_columns")
            if feats and raws:
                return list(feats), list(raws)
        except (json.JSONDecodeError, TypeError):
            pass
    return ENGINEERED_FEATURES, RAW_FEATURES


def load_embargo_months(input_dir: str) -> list[int] | None:
    path = os.path.join(input_dir, "temporal_folds.json")
    if not os.path.exists(path):
        return None
    try:
        folds = load_json(path)
        if not isinstance(folds, list):
            return None
        months = sorted({m for fold in folds for m in fold.get("embargo_months", [])})
        return months or None
    except (json.JSONDecodeError, TypeError):
        return None


def savefig(fig: plt.Figure, figdir: str, name: str):
    path = os.path.join(figdir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


# ---------------------------------------------------------------------------
# Section 1: Dataset Overview
# ---------------------------------------------------------------------------

def section_overview(splits: dict[str, pd.DataFrame],
                     eng_feats: list[str], raw_feats: list[str]) -> str:
    lines = ["## 1. Dataset Overview\n"]

    rows = []
    for name, df in splits.items():
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        n_personas = df["user_id"].nunique()
        n_months = df["month"].nunique()
        rows.append({
            "Split": name.capitalize(),
            "Rows": f"{len(df):,}",
            "Personas": f"{n_personas:,}",
            "Months": n_months,
            "Memory (MB)": f"{mem_mb:.1f}",
        })
    rows.append({
        "Split": "Total",
        "Rows": f"{sum(len(d) for d in splits.values()):,}",
        "Personas": f"{sum(d['user_id'].nunique() for d in splits.values()):,}",
        "Months": max((d["month"].max() for d in splits.values()), default=0),
        "Memory (MB)": f"{sum(d.memory_usage(deep=True).sum() for d in splits.values()) / 1e6:.1f}",
    })

    lines.append("| Split | Rows | Personas | Months | Memory (MB) |")
    lines.append("|-------|------|----------|--------|-------------|")
    for r in rows:
        lines.append(f"| {r['Split']} | {r['Rows']} | {r['Personas']} | {r['Months']} | {r['Memory (MB)']} |")

    train = splits.get("train", list(splits.values())[0])
    avail_eng = _available(train, eng_feats)
    avail_raw = _available(train, raw_feats)
    lines.append(f"\n**Columns:** {len(avail_eng)} engineered + {len(avail_raw)} raw + {len(META_COLUMNS)} metadata = {len(avail_eng) + len(avail_raw) + len(META_COLUMNS)} total\n")

    all_feats = list(set(avail_eng + avail_raw))
    missing = train[all_feats].isnull().sum() if all_feats else pd.Series(dtype=int)
    missing = missing[missing > 0]
    if len(missing) > 0:
        lines.append("### Missing Values (Training Set)\n")
        lines.append("| Feature | Missing Count |")
        lines.append("|---------|---------------|")
        for feat, count in missing.items():
            lines.append(f"| {feat} | {count:,} |")
    else:
        lines.append("**Missing values:** None detected in training set.\n")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 2: Feature Distributions
# ---------------------------------------------------------------------------

def section_distributions(splits: dict[str, pd.DataFrame], figdir: str,
                          eng_feats: list[str]) -> str:
    lines = ["## 2. Feature Distributions\n"]
    train = splits.get("train", list(splits.values())[0])

    def _stats_table(features: pd.DataFrame, name: str) -> str:
        desc = features.describe().T
        desc["skewness"] = features.skew()
        desc["kurtosis"] = features.kurtosis()
        t = ["| Feature | Mean | Std | Min | Max | Skew | Kurt |"]
        t.append("|---------|------|-----|-----|-----|------|------|")
        for feat in desc.index:
            r = desc.loc[feat]
            t.append(
                f"| {feat} | {r['mean']:.3f} | {r['std']:.3f} "
                f"| {r['min']:.3f} | {r['max']:.3f} "
                f"| {r['skewness']:.2f} | {r['kurtosis']:.2f} |"
            )
        return "\n".join(t)

    lines.append("### 2.1 Engineered Features\n")
    avail_eng = _available(train, eng_feats)
    if eng_feats:
        lines.append(_stats_table(train[eng_feats], "engineered"))
        lines.append("")

        ncols = 5
        nrows = (len(eng_feats) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 4))
        axes = axes.flatten()
        test = splits.get("test")
        for i, feat in enumerate(eng_feats):
            ax = axes[i]
            ax.hist(train[feat], bins=40, alpha=0.6, label="train", density=True, color="#3498db")
            if test is not None:
                ax.hist(test[feat], bins=40, alpha=0.4, label="test", density=True, color="#e74c3c")
            ax.set_title(feat, fontsize=9)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Engineered Feature Distributions (Train vs Test)", fontsize=14, y=1.01)
        fig.tight_layout()
        savefig(fig, figdir, "feature_distributions_engineered.png")
        lines.append("![Engineered Feature Distributions](feature_distributions_engineered.png)\n")
    else:
        lines.append("*No engineered features present in data. Skipping.*\n")

    raw_feats = _available(train, RAW_FEATURES)
    if raw_feats:
        lines.append("### 2.2 Raw Cumulative Features\n")
        lines.append(_stats_table(train[raw_feats], "raw"))
        lines.append("")

        ncols = 4
        nrows = (len(raw_feats) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
        axes = axes.flatten()
        for i, feat in enumerate(raw_feats):
            ax = axes[i]
            data = train[feat]
            ax.hist(data, bins=40, alpha=0.7, color="#2ecc71")
            ax.set_title(feat, fontsize=9)
            ax.tick_params(labelsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Raw Cumulative Feature Distributions (Training Set)", fontsize=14, y=1.01)
        fig.tight_layout()
        savefig(fig, figdir, "feature_distributions_raw.png")
        lines.append("![Raw Feature Distributions](feature_distributions_raw.png)\n")
    else:
        lines.append("### 2.2 Raw Cumulative Features\n")
        lines.append("*No raw features present in data. Skipping.*\n")

    skewed = []
    for feat in eng_feats:
        s = abs(train[feat].skew())
        if s > 2:
            skewed.append((feat, train[feat].skew()))
    if skewed:
        lines.append("### 2.3 High-Skewness Features\n")
        lines.append("Features with |skewness| > 2 may benefit from transformation:\n")
        lines.append("| Feature | Skewness |")
        lines.append("|---------|----------|")
        for feat, s in sorted(skewed, key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"| {feat} | {s:.2f} |")
        lines.append("")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 3: Class Balance
# ---------------------------------------------------------------------------

def section_class_balance(splits: dict[str, pd.DataFrame], figdir: str) -> str:
    lines = ["## 3. Class Balance Analysis\n"]
    train = splits.get("train", list(splits.values())[0])

    lines.append("### 3.1 PFP Class Distribution\n")
    lines.append("| Class | Train | Val | Test | Overall |")
    lines.append("|-------|-------|-----|------|---------|")
    for cls in PFP_CLASSES:
        counts = []
        for name in ["train", "val", "test"]:
            df = splits.get(name)
            if df is not None:
                counts.append(int((df["pfp_label"] == cls).sum()))
            else:
                counts.append(0)
        overall = sum(counts)
        lines.append(f"| {cls} | {counts[0]:,} | {counts[1]:,} | {counts[2]:,} | {overall:,} |")
    lines.append("")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (name, df) in enumerate(splits.items()):
        ax = axes[idx]
        counts = df["pfp_label"].value_counts().reindex(PFP_CLASSES, fill_value=0)
        bars = ax.bar(range(len(PFP_CLASSES)), counts.values,
                      color=[PFP_PALETTE[c] for c in PFP_CLASSES])
        ax.set_xticks(range(len(PFP_CLASSES)))
        ax.set_xticklabels([c.replace("/", "\n") for c in PFP_CLASSES], fontsize=8)
        ax.set_title(f"{name.capitalize()} Split", fontsize=12)
        ax.set_ylabel("Count")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{val:,}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("PFP Class Distribution by Split", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, figdir, "class_distribution.png")
    lines.append("![Class Distribution](class_distribution.png)\n")

    lines.append("### 3.2 Class Proportion Comparison\n")
    rows = []
    for cls in PFP_CLASSES:
        row = {"Class": cls}
        for name, df in splits.items():
            total = len(df)
            count = (df["pfp_label"] == cls).sum()
            row[name] = f"{count / total * 100:.1f}%"
        rows.append(row)
    lines.append("| Class | Train | Val | Test |")
    lines.append("|-------|-------|-----|------|")
    for r in rows:
        lines.append(f"| {r['Class']} | {r.get('train', '-')} | {r.get('val', '-')} | {r.get('test', '-')} |")
    lines.append("")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(PFP_CLASSES))
    width = 0.25
    for i, (name, df) in enumerate(splits.items()):
        props = [(df["pfp_label"] == cls).sum() / len(df) * 100 for cls in PFP_CLASSES]
        ax.bar(x + i * width, props, width, label=name.capitalize(), alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("/", "\n") for c in PFP_CLASSES], fontsize=9)
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Class Proportion Comparison Across Splits", fontsize=13)
    ax.legend()
    fig.tight_layout()
    savefig(fig, figdir, "class_balance_by_split.png")
    lines.append("![Class Balance by Split](class_balance_by_split.png)\n")

    top_features = _available(train, [
        "income_stability_cv", "obligation_ratio", "savings_rate",
        "debt_to_income", "income_expense_gap", "essential_income_ratio",
    ])
    if top_features:
        ncols = 3
        nrows = (len(top_features) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
        if nrows * ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, feat in enumerate(top_features):
            ax = axes[i]
            data = []
            labels = []
            for cls in PFP_CLASSES:
                vals = train.loc[train["pfp_label"] == cls, feat].dropna()
                data.append(vals)
                labels.append(cls.replace("/", "\n"))
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for patch, cls in zip(bp["boxes"], PFP_CLASSES):
                patch.set_facecolor(PFP_PALETTE[cls])
                patch.set_alpha(0.7)
            ax.set_title(feat, fontsize=10)
            ax.tick_params(labelsize=8)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Top Discriminative Features by PFP Class", fontsize=14, y=1.01)
        fig.tight_layout()
        savefig(fig, figdir, "feature_by_class_boxplots.png")
        lines.append("### 3.3 Feature Distributions by Class\n")
        lines.append("![Feature by Class](feature_by_class_boxplots.png)\n")
    else:
        lines.append("### 3.3 Feature Distributions by Class\n")
        lines.append("*No engineered features available for class comparison. Skipping.*\n")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 4: Correlation Analysis
# ---------------------------------------------------------------------------

def section_correlations(splits: dict[str, pd.DataFrame], figdir: str,
                         eng_feats: list[str]) -> str:
    lines = ["## 4. Correlation Analysis\n"]
    train = splits.get("train", list(splits.values())[0])
    numeric_cols = _available(train, eng_feats + RAW_FEATURES)

    if not numeric_cols:
        lines.append("*No numeric features available for correlation analysis.*\n")
        return "\n".join(lines)

    corr_pearson = train[numeric_cols].corr(method="pearson")
    corr_spearman = train[numeric_cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
    sns.heatmap(corr_pearson, mask=mask, annot=False, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, ax=ax, linewidths=0.5)
    ax.set_title("Pearson Correlation Matrix", fontsize=14)
    fig.tight_layout()
    savefig(fig, figdir, "correlation_pearson.png")
    lines.append("### 4.1 Pearson Correlation\n")
    lines.append("![Pearson Correlation](correlation_pearson.png)\n")

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_spearman, mask=mask, annot=False, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, ax=ax, linewidths=0.5)
    ax.set_title("Spearman Correlation Matrix", fontsize=14)
    fig.tight_layout()
    savefig(fig, figdir, "correlation_spearman.png")
    lines.append("### 4.2 Spearman Correlation\n")
    lines.append("![Spearman Correlation](correlation_spearman.png)\n")

    high_corr = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            r = corr_pearson.iloc[i, j]
            if abs(r) > 0.8:
                high_corr.append((numeric_cols[i], numeric_cols[j], r))
    if high_corr:
        lines.append("### 4.3 High Correlation Pairs (|r| > 0.8)\n")
        lines.append("**Multicollinearity warning:** These feature pairs are highly correlated and may cause instability in linear models.\n")
        lines.append("| Feature 1 | Feature 2 | Pearson r |")
        lines.append("|-----------|-----------|-----------|")
        for f1, f2, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            lines.append(f"| {f1} | {f2} | {r:.3f} |")
        lines.append("")
    else:
        lines.append("### 4.3 High Correlation Pairs\n")
        lines.append("No feature pairs with |r| > 0.8 found.\n")

    avail_eng = _available(train, eng_feats)
    if avail_eng:
        lines.append("### 4.4 Correlation with PFP Target\n")
        lines.append("Point-biserial correlation: each class encoded as 1-vs-rest.\n")
        header_cols = " | ".join(c.replace("/", "-") for c in PFP_CLASSES)
        lines.append(f"| Feature | {header_cols} |")
        lines.append("|---------|" + "-----------------|" * len(PFP_CLASSES))
        for feat in avail_eng:
            corrs = []
            for cls in PFP_CLASSES:
                binary = (train["pfp_label"] == cls).astype(int)
                r = train[feat].corr(binary, method="pearson")
                corrs.append(f"{r:.3f}")
            lines.append(f"| {feat} | {' | '.join(corrs)} |")
        lines.append("")

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(avail_eng))
        width = 0.2
        for i, cls in enumerate(PFP_CLASSES):
            binary = (train["pfp_label"] == cls).astype(int)
            corrs = [train[feat].corr(binary, method="pearson") for feat in avail_eng]
            ax.bar(x + i * width, corrs, width, label=cls.replace("/", "/"), alpha=0.8)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f.replace("_", "\n") for f in avail_eng], fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Point-Biserial Correlation")
        ax.set_title("Feature Correlation with PFP Classes (1-vs-Rest)", fontsize=13)
        ax.legend(fontsize=8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        fig.tight_layout()
        savefig(fig, figdir, "correlation_with_target.png")
        lines.append("![Correlation with Target](correlation_with_target.png)\n")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 5: Temporal Patterns
# ---------------------------------------------------------------------------

def section_temporal(splits: dict[str, pd.DataFrame], figdir: str,
                     eng_feats: list[str], input_dir: str) -> str:
    lines = ["## 5. Temporal Patterns\n"]
    train = splits.get("train", list(splits.values())[0])

    embargo_months = load_embargo_months(input_dir)

    temporal_feats = _available(train, [
        "income_stability_cv", "obligation_ratio", "savings_rate",
        "income_expense_gap", "volatility_index", "transaction_frequency",
    ])
    if temporal_feats:
        ncols = min(3, len(temporal_feats))
        nrows = (len(temporal_feats) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
        if nrows * ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, feat in enumerate(temporal_feats):
            ax = axes[i]
            monthly = train.groupby("month")[feat].agg(["mean", "std"]).reset_index()
            ax.plot(monthly["month"], monthly["mean"], marker="o", linewidth=2, color="#3498db")
            ax.fill_between(monthly["month"],
                            monthly["mean"] - monthly["std"],
                            monthly["mean"] + monthly["std"],
                            alpha=0.2, color="#3498db")
            ax.set_title(feat, fontsize=10)
            ax.set_xlabel("Month")
            ax.set_ylabel("Mean ± Std")
            ax.set_xticks(range(1, 13))
            if embargo_months:
                ax.axvspan(min(embargo_months) - 0.5, max(embargo_months) + 0.5,
                           alpha=0.1, color="red",
                           label="Embargo" if i == 0 else "")
            if i == 0:
                ax.legend(fontsize=8)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Mean Feature Values by Month (Training Set)", fontsize=14, y=1.01)
        fig.tight_layout()
        savefig(fig, figdir, "temporal_feature_means.png")
        lines.append("### 5.1 Feature Trajectories by Month\n")
        if embargo_months:
            emb = ", ".join(str(m) for m in embargo_months)
            lines.append(f"The red band marks embargo months ({emb}) defined in temporal_folds.json, held out between training and test periods.\n")
        else:
            lines.append("No temporal_folds.json found; embargo band not shown.\n")
        lines.append("![Temporal Feature Means](temporal_feature_means.png)\n")
    else:
        lines.append("### 5.1 Feature Trajectories by Month\n")
        lines.append("*No engineered features available for temporal analysis. Skipping.*\n")

    lines.append("### 5.2 Month-1 Feature Degeneracy\n")
    month1 = train[train["month"] == 1]
    lines.append(f"Month 1 rows: {len(month1):,} ({len(month1) / len(train) * 100:.1f}% of training data)\n")
    degenerate = []
    avail_eng = _available(train, eng_feats)
    for feat in avail_eng:
        unique_vals = month1[feat].nunique()
        if unique_vals <= 3:
            degenerate.append((feat, unique_vals))
    if degenerate:
        lines.append("**Degenerate features at month 1** (≤3 unique values — early months have limited signal):\n")
        lines.append("| Feature | Unique Values |")
        lines.append("|---------|---------------|")
        for feat, n in degenerate:
            lines.append(f"| {feat} | {n} |")
    else:
        lines.append("No fully degenerate features detected at month 1, though values are expected to be noisier.\n")
    lines.append("")

    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_income = train.groupby(["month", "pfp_label"])["total_income"].mean().reset_index()
    monthly_expense = train.groupby(["month", "pfp_label"])["total_expenses"].mean().reset_index()
    for cls in PFP_CLASSES:
        inc = monthly_income[monthly_income["pfp_label"] == cls]
        exp = monthly_expense[monthly_expense["pfp_label"] == cls]
        ax.plot(inc["month"], inc["total_income"], marker="o", linewidth=2,
                color=PFP_PALETTE[cls], label=f"{cls} (income)")
        ax.plot(exp["month"], exp["total_expenses"], marker="s", linewidth=2,
                color=PFP_PALETTE[cls], linestyle="--", label=f"{cls} (expense)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Cumulative Amount")
    ax.set_title("Income vs Expense Trajectories by PFP Class", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    savefig(fig, figdir, "temporal_income_expense_by_class.png")
    lines.append("### 5.3 Income vs Expense Trajectories\n")
    lines.append("![Income vs Expense](temporal_income_expense_by_class.png)\n")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 6: Anomaly Analysis
# ---------------------------------------------------------------------------

def section_anomalies(splits: dict[str, pd.DataFrame], figdir: str) -> str:
    lines = ["## 6. Anomaly Analysis\n"]
    train = splits.get("train", list(splits.values())[0])

    total_rows = len(train)
    anomalous_rows = train["is_anomalous"].sum()
    lines.append(f"**Anomaly rate (training set):** {anomalous_rows:,} / {total_rows:,} = {anomalous_rows / total_rows * 100:.2f}%\n")

    lines.append("### 6.1 Anomaly Rate by Month\n")
    monthly = train.groupby("month").agg(
        total=("is_anomalous", "count"),
        anomalous=("is_anomalous", "sum"),
    ).reset_index()
    monthly["rate"] = monthly["anomalous"] / monthly["total"] * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly["month"], monthly["rate"], marker="o", linewidth=2, color="#e74c3c")
    ax.fill_between(monthly["month"], 0, monthly["rate"], alpha=0.2, color="#e74c3c")
    ax.set_xlabel("Month")
    ax.set_ylabel("Anomaly Rate (%)")
    ax.set_title("Anomaly Rate by Month (Training Set)", fontsize=13)
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    savefig(fig, figdir, "anomaly_rate_by_month.png")
    lines.append("![Anomaly Rate by Month](anomaly_rate_by_month.png)\n")

    lines.append("### 6.2 Anomaly Rate by PFP Class\n")
    class_anom = train.groupby("pfp_label").agg(
        total=("is_anomalous", "count"),
        anomalous=("is_anomalous", "sum"),
    ).reset_index()
    class_anom["rate"] = class_anom["anomalous"] / class_anom["total"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(PFP_CLASSES)),
                  [class_anom.loc[class_anom["pfp_label"] == c, "rate"].values[0]
                   if c in class_anom["pfp_label"].values else 0 for c in PFP_CLASSES],
                  color=[PFP_PALETTE[c] for c in PFP_CLASSES])
    ax.set_xticks(range(len(PFP_CLASSES)))
    ax.set_xticklabels([c.replace("/", "\n") for c in PFP_CLASSES])
    ax.set_ylabel("Anomaly Rate (%)")
    ax.set_title("Anomaly Rate by PFP Class", fontsize=13)
    for bar, val in zip(bars, [class_anom.loc[class_anom["pfp_label"] == c, "rate"].values[0]
                                if c in class_anom["pfp_label"].values else 0 for c in PFP_CLASSES]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    savefig(fig, figdir, "anomaly_rate_by_class.png")
    lines.append("![Anomaly Rate by Class](anomaly_rate_by_class.png)\n")

    lines.append("### 6.3 Feature Distributions: Anomalous vs Normal\n")
    compare_feats = _available(train, [
        "income_stability_cv", "volatility_index", "category_entropy",
        "transaction_frequency", "avg_transaction_size", "savings_rate",
    ])
    if compare_feats:
        ncols = min(3, len(compare_feats))
        nrows = (len(compare_feats) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
        if nrows * ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, feat in enumerate(compare_feats):
            ax = axes[i]
            normal = train.loc[~train["is_anomalous"], feat]
            anomalous = train.loc[train["is_anomalous"], feat]
            ax.hist(normal, bins=40, alpha=0.6, label="Normal", density=True, color="#3498db")
            ax.hist(anomalous, bins=40, alpha=0.6, label="Anomalous", density=True, color="#e74c3c")
            ax.set_title(feat, fontsize=10)
            ax.legend(fontsize=8)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Feature Distributions: Anomalous vs Normal", fontsize=14, y=1.01)
        fig.tight_layout()
        savefig(fig, figdir, "anomaly_feature_comparison.png")
        lines.append("![Anomaly Feature Comparison](anomaly_feature_comparison.png)\n")
    else:
        lines.append("*No engineered features available for anomaly comparison. Skipping.*\n")

    lines.append("### 6.4 Anomaly Type Breakdown\n")
    anom_data = train[train["is_anomalous"] & (train["anomaly_type"] != "")]
    if len(anom_data) > 0:
        type_counts = anom_data["anomaly_type"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(type_counts.index, type_counts.values, color="#e67e22")
        ax.set_xlabel("Count")
        ax.set_title("Anomaly Type Distribution", fontsize=13)
        for bar, val in zip(bars, type_counts.values):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9)
        fig.tight_layout()
        savefig(fig, figdir, "anomaly_type_breakdown.png")
        lines.append("![Anomaly Type Breakdown](anomaly_type_breakdown.png)\n")
    else:
        lines.append("No typed anomalies found in training set.\n")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 7: Data Quality Report
# ---------------------------------------------------------------------------

def section_quality(splits: dict[str, pd.DataFrame], figdir: str,
                    pipeline_report: dict | None = None,
                    eng_feats: list[str] | None = None) -> str:
    lines = ["## 7. Data Quality Report\n"]
    train = splits.get("train", list(splits.values())[0])
    if eng_feats is None:
        eng_feats = ENGINEERED_FEATURES

    sec = 0
    sec += 1
    lines.append(f"### 7.{sec} Zero-Income Rows\n")
    zero_income = (train["total_income"] == 0).sum()
    zero_pct = zero_income / len(train) * 100
    lines.append(f"**{zero_income:,} rows** ({zero_pct:.1f}%) have zero total income.\n")
    lines.append("These rows will produce extreme values in all income-ratio features "
                 "(savings_rate, debt_to_income, essential_income_ratio).\n")

    zero_ratio_feats = _available(train, ["savings_rate", "debt_to_income", "essential_income_ratio"])
    if zero_ratio_feats:
        ncols = min(3, len(zero_ratio_feats))
        nrows = (len(zero_ratio_feats) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 5))
        if nrows * ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for idx, feat in enumerate(zero_ratio_feats):
            ax = axes[idx]
            zero_mask = train["total_income"] == 0
            ax.hist(train.loc[~zero_mask, feat], bins=40, alpha=0.6, label="Income > 0",
                    density=True, color="#3498db")
            ax.hist(train.loc[zero_mask, feat], bins=40, alpha=0.6, label="Income = 0",
                    density=True, color="#e74c3c")
            ax.set_title(feat, fontsize=10)
            ax.legend(fontsize=8)
        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Impact of Zero-Income Rows on Ratio Features", fontsize=14, y=1.02)
        fig.tight_layout()
        savefig(fig, figdir, "zero_income_analysis.png")
        lines.append("![Zero Income Analysis](zero_income_analysis.png)\n")
    else:
        lines.append("*No ratio features available for zero-income analysis. Skipping.*\n")

    if pipeline_report and "feature_range_violations" in pipeline_report:
        violations = pipeline_report["feature_range_violations"]
        sec += 1
        lines.append(f"### 7.{sec} Feature Range Violations\n")
        lines.append("Note: The reported ranges apply to raw (pre-scaling) values. "
                     "The current processed data is StandardScaler-normalized (mean≈0, std≈1).\n")
        lines.append("| Feature | Expected Range | Below Count | Above Count |")
        lines.append("|---------|---------------|-------------|-------------|")
        for v in violations:
            lines.append(f"| {v['feature']} | {v['expected_range']} | {v['below_count']:,} | {v['above_count']:,} |")
        lines.append("")

    sec += 1
    lines.append(f"### 7.{sec} Outlier Summary (IQR Method)\n")
    outlier_feats = _available(train, eng_feats)[:8]
    if outlier_feats:
        outlier_data = []
        for feat in outlier_feats:
            q1 = train[feat].quantile(0.25)
            q3 = train[feat].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = ((train[feat] < lower) | (train[feat] > upper)).sum()
            outlier_data.append((feat, n_outliers, n_outliers / len(train) * 100))

        fig, ax = plt.subplots(figsize=(12, 5))
        names = [o[0] for o in outlier_data]
        counts = [o[1] for o in outlier_data]
        pcts = [o[2] for o in outlier_data]
        bars = ax.barh(names, counts, color="#9b59b6")
        ax.set_xlabel("Outlier Count (IQR Method)")
        ax.set_title("Outlier Counts by Feature (Top 8 Engineered)", fontsize=13)
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=9)
        fig.tight_layout()
        savefig(fig, figdir, "outlier_summary.png")
        lines.append("| Feature | Outlier Count | Outlier % |")
        lines.append("|---------|---------------|-----------|")
        for feat, count, pct in outlier_data:
            lines.append(f"| {feat} | {count:,} | {pct:.1f}% |")
        lines.append("")
        lines.append("![Outlier Summary](outlier_summary.png)\n")
    else:
        lines.append("*No engineered features available for outlier analysis. Skipping.*\n")

    sec += 1
    lines.append(f"### 7.{sec} Recommendations\n")
    lines.append(f"1. **Zero-income rows ({zero_pct:.1f}%):** Expected behavior — intentional zero-income months from irregular/project-based income patterns. Not a bug.")
    lines.append("2. **Skewed features:** Apply log or Box-Cox transformation to highly skewed features before linear models.")
    lines.append("3. **Multicollinearity:** `obligation_ratio` and `essential_income_ratio` capture distinct constructs (obligation load vs essential share of income) and are currently weakly correlated (r≈0.06); monitor for divergence once obligatory expense fields (debt/loan repayments, remittances) are added.")
    lines.append("4. **Month-1 degeneracy:** Early months have limited signal; consider excluding months 1-2 from training or adding minimum-history gates.")
    lines.append("5. **Outlier capping:** The preprocessing pipeline already caps `debt_to_income` and `savings_rate` at 99th percentile; consider extending to other features.")
    lines.append("")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report Assembly
# ---------------------------------------------------------------------------

def generate_report(sections: list[str], output_dir: str) -> str:
    header = (
        "# Exploratory Data Analysis Report\n\n"
        "**Generated by:** `scripts/eda.py`\n\n"
        "---\n\n"
    )
    toc = "## Table of Contents\n\n"
    for section in sections:
        title_line = section.split("\n")[0].strip()
        if title_line.startswith("## "):
            toc += f"- {title_line}\n"
    toc += "\n---\n\n"

    report = header + toc + "\n\n".join(sections)
    report_path = os.path.join(output_dir, "eda_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Odin ML — Exploratory Data Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default="datasets/engineered/",
                        help="Input directory with train.parquet, val.parquet, test.parquet")
    parser.add_argument("--output", type=str, default="figures/",
                        help="Output directory for report and plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    figdir = args.output
    os.makedirs(figdir, exist_ok=True)

    print(f"Loading data from {args.input}...")
    splits = load_data(args.input)
    if not splits:
        print("Error: No Parquet files found in input directory.")
        sys.exit(1)
    print(f"  Loaded splits: {', '.join(f'{k}({len(v):,})' for k, v in splits.items())}")

    pipeline_report = None
    report_path = os.path.join(args.input, "pipeline_report.json")
    if os.path.exists(report_path):
        pipeline_report = load_json(report_path)
        print(f"  Loaded pipeline_report.json")

    eng_feats, raw_feats = load_feature_schema(args.input)

    print("Generating Section 1: Dataset Overview...")
    s1 = section_overview(splits, eng_feats, raw_feats)

    print("Generating Section 2: Feature Distributions...")
    s2 = section_distributions(splits, figdir, eng_feats)

    print("Generating Section 3: Class Balance Analysis...")
    s3 = section_class_balance(splits, figdir)

    print("Generating Section 4: Correlation Analysis...")
    s4 = section_correlations(splits, figdir, eng_feats)

    print("Generating Section 5: Temporal Patterns...")
    s5 = section_temporal(splits, figdir, eng_feats, args.input)

    print("Generating Section 6: Anomaly Analysis...")
    s6 = section_anomalies(splits, figdir)

    print("Generating Section 7: Data Quality Report...")
    s7 = section_quality(splits, figdir, pipeline_report, eng_feats)

    sections = [s1, s2, s3, s4, s5, s6, s7]
    print("Assembling report...")
    report_path = generate_report(sections, args.output)

    plot_count = len([f for f in os.listdir(figdir) if f.endswith(".png")])
    print(f"\nDone!")
    print(f"  Report: {report_path}")
    print(f"  Plots:  {figdir}/ ({plot_count} files)")


if __name__ == "__main__":
    main()
