"""Regenerate the committed report/metadata artifacts for all model families.

Reads each family's existing `evaluation.json` (the canonical evaluation
result) and re-emits `evaluation_report.md` + `metadata.json` through the shared
app.ml.reporting writers, guaranteeing a uniform format and winner contract
(winner / winner_artifact / winner_params / threshold). No retraining happens.

Budget is a deterministic LP: this script runs `budget_service.optimize` over
a deterministic sample of synthetic personas and emits its first evaluation.json
(constraint satisfaction, utilization, deviation) plus report and metadata.

Usage:
    python -m training.scripts.modeling.regenerate_artifacts
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.ml.metadata import build_metadata, write_metadata
from app.ml.reporting import family_metadata, family_report, write_evaluation_report

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_SOURCES = {
    "forecaster": REPO_ROOT / "training/datasets/forecaster",
    "pfp": REPO_ROOT / "training/datasets/engineered",
    "anomaly": REPO_ROOT / "training/datasets/anomaly",
    "budget": REPO_ROOT / "training/synth/personas.parquet",
}

MODEL_DIRS = {
    "forecaster": REPO_ROOT / "models/forecaster",
    "pfp": REPO_ROOT / "models/pfp",
    "anomaly": REPO_ROOT / "models/anomaly",
    "budget": REPO_ROOT / "models/budget",
}

DEFAULT_WINNER_ARTIFACTS = {
    "forecaster": "tier3_sarima.joblib",
    "anomaly": "anomaly_detector.joblib",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _ensure_winner_artifact(evaluation: dict, family: str) -> None:
    if evaluation.get("winner_artifact"):
        return
    artifact = DEFAULT_WINNER_ARTIFACTS.get(family)
    if artifact:
        evaluation["winner_artifact"] = artifact


def regenerate_family(family: str) -> Path:
    out_dir = MODEL_DIRS[family]
    evaluation_path = out_dir / "evaluation.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"missing evaluation.json for {family}: {evaluation_path}")

    evaluation = json.loads(evaluation_path.read_text())
    evaluation.setdefault("generated_at", utc_now())
    _ensure_winner_artifact(evaluation, family)
    evaluation_path.write_text(json.dumps(evaluation, indent=2))

    write_evaluation_report(out_dir, **family_report(family, evaluation))

    data_sources = [DATA_SOURCES[family] / f"{name}.parquet" for name in ("train", "val", "test")]

    metadata = build_metadata(**family_metadata(family, evaluation, data_sources=data_sources))
    metadata["created_at"] = evaluation["generated_at"]
    previous = out_dir / "metadata.json"
    if previous.exists():
        previous_commit = json.loads(previous.read_text()).get("training_commit")
        if previous_commit:
            metadata["training_commit"] = previous_commit
    write_metadata(metadata, out_dir)
    return out_dir / "metadata.json"


def evaluate_budget(n_users: int = 600, seed: int = 42) -> dict:
    """Run the deterministic LP over synthetic personas and summarize."""
    import numpy as np
    import pandas as pd
    from app.schemas.budget import (
        BudgetCategory,
        BudgetPeriod,
        BudgetRequest,
        RestrictionLevel,
    )
    from app.services.budget_service import optimize

    personas = pd.read_parquet(DATA_SOURCES["budget"])
    sample = personas.sample(n_users, random_state=seed)

    category_order = ["food", "housing", "transport", "health", "education", "other"]
    expense_cols = {
        "food": "food_expense",
        "housing": "housing_expense",
        "transport": "transport_expense",
        "health": "health_expense",
        "education": "education_expense",
        "other": "other_expense",
    }

    satisfaction, utilizations, deviations = [], [], []
    infeasible = 0
    feasible = 0

    for _, row in sample.iterrows():
        income = float(row["monthly_income"])
        if income <= 0:
            continue
        spends = {c: float(row[expense_cols[c]]) for c in category_order}
        total_spend = sum(spends.values()) or 1.0
        targets = {c: spends[c] / total_spend for c in category_order}

        categories = []
        for c in category_order:
            if c == "housing":
                level = RestrictionLevel.FIXED
                floor = spends[c]
                ceiling = spends[c]
            else:
                level = RestrictionLevel.FLEXIBLE
                floor = 0.0
                ceiling = max(spends[c] * 1.5, 1.0)
            categories.append(
                BudgetCategory(
                    category_id=c,
                    restriction_level=level,
                    floor=floor,
                    ceiling=ceiling,
                    priority_weight=0.5,
                    current_spend=spends[c],
                )
            )

        request = BudgetRequest(
            request_id=f"eval-{row['persona_id']}",
            user_id=row["persona_id"],
            available_funds=income,
            period=BudgetPeriod(start="2026-01-01", end="2026-01-31"),
            categories=categories,
            target_ratios=targets,
            include_reasoning=False,
        )
        try:
            recommendation, _ = optimize(request)
        except ValueError:
            infeasible += 1
            continue

        satisfaction.append(float(recommendation.constraint_satisfaction))
        utilizations.append(float(recommendation.utilization_rate))
        allocations = {a.category_id: a.amount for a in recommendation.allocations}
        deviations.append(
            sum(
                abs(allocations.get(c, 0.0) - income * targets.get(c, 0.0)) / income
                for c in category_order
            )
        )
        if recommendation.feasibility == "FEASIBLE":
            feasible += 1
        elif recommendation.feasibility == "INFEASIBLE":
            infeasible += 1

    total = feasible + infeasible
    return {
        "generated_at": utc_now(),
        "task": "budget_optimization",
        "task_type": "deterministic_lp",
        "n_folds": 1,
        "n_users": total,
        "winner": "scipy_linprog",
        "winner_artifact": "budget_config.json",
        "winner_reason": "scipy.linprog (exact, fast) selected as current v1 per model candidate roster",
        "decision_rule_text": (
            "LP must satisfy all hard constraints with high utilization and "
            "minimal deviation from user preferences"
        ),
        "constraint_satisfaction_rate": float(np.mean(satisfaction)) if satisfaction else 0.0,
        "budget_utilization_rate": float(np.mean(utilizations)) if utilizations else 0.0,
        "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
        "feasibility": f"{feasible}/{total} FEASIBLE, {infeasible}/{total} INFEASIBLE",
        "feature_columns": [],
        "metrics": None,
    }


def regenerate_budget() -> Path:
    out_dir = MODEL_DIRS["budget"]
    evaluation = evaluate_budget()
    evaluation_path = out_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2))

    (out_dir / "budget_config.json").write_text(
        json.dumps(
            {
                "method": "highs",
                "artifact": "models/budget/budget_config.json",
                "description": (
                    "Deterministic two-stage LP formulation snapshot (Financial Planning "
                    "Management V4). The optimizer code lives in "
                    "app/services/budget_service.py; this config pins the solver and the "
                    "constraint/monitoring contract."
                ),
                "solver": "scipy.optimize.linprog",
                "decision_variables": [
                    "allocation x_i >= 0 (continuous, unrounded)",
                    "stage-1 Essential coverage level r >= 0",
                ],
                "bounds": {
                    "FIXED": "l_i = U_i = floor",
                    "MINIMUM": "l_i = floor, U_i = max(floor, min(ceiling, target))",
                    "FLEXIBLE": "l_i = floor, U_i = max(floor, min(ceiling, target))",
                    "target": "forecast amount, else available_funds * target_ratio (missing = 0)",
                },
                "stages": [
                    {
                        "name": "essential_coverage",
                        "objective": "maximize r",
                        "constraints": [
                            "x_i >= l_i + r * w_i * (U_i - l_i) for each Essential with U_i > l_i",
                            "l_i <= x_i <= U_i",
                            "sum(x) <= available_funds",
                        ],
                        "output": "stage-1 x_i becomes the floor for each Essential",
                    },
                    {
                        "name": "essential_first_allocation",
                        "objective": "maximize sum((w_i + P * I_essential_i) * x_i)",
                        "priority_coefficient": "P = sum(w_i * U_i) + 1",
                        "constraints": [
                            "stage-1 floor <= x_i <= U_i (Essential)",
                            "l_i <= x_i <= U_i (Non-essential)",
                            "sum(x) <= available_funds",
                        ],
                    },
                ],
                "weights": "w_i = max(priority_weight, eps)",
                "rounding": "none",
                "surplus": "available_funds - sum(x) is returned as unallocated_surplus",
                "infeasible_when": "sum(l_i) > available_funds",
                "eps": 1e-9,
                "feasibility_states": ["FEASIBLE", "INFEASIBLE"],
                "source_data": "training/synth/personas.parquet",
                "evaluation": "models/budget/evaluation.json",
            },
            indent=2,
        )
    )

    write_evaluation_report(out_dir, **family_report("budget", evaluation))

    metadata = build_metadata(
        **family_metadata(
            "budget",
            evaluation,
            data_sources=[DATA_SOURCES["budget"]],
        )
    )
    write_metadata(metadata, out_dir)
    return out_dir / "metadata.json"


def main() -> None:
    print(f"generated_at (UTC): {utc_now()}")
    for family in ("forecaster", "anomaly", "pfp"):
        path = regenerate_family(family)
        print(f"  {family}: {path}")
    budget_path = regenerate_budget()
    print(f"  budget: {budget_path} (evaluated via LP over synthetic personas)")


if __name__ == "__main__":
    main()
