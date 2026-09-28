"""Compare completed full-corpus v3 candidate reports without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.scripts.shared.logging_v3 import configure_logging, get_logger

LOGGER = get_logger("full_comparison")
EVALUATION_CONTRACT = "three_prior_month_user_relative_ratio_v1"
COMPARABLE_CANDIDATES = {"naive", "random_forest"}


def compare(output_dir: str | Path) -> dict:
    destination = Path(output_dir)
    reports = []
    for path in sorted((destination / "candidates").glob("*.json")):
        report = json.loads(path.read_text())
        if report.get("training_scope") != "full_household_corpus":
            continue
        if report.get("evaluation_contract") != EVALUATION_CONTRACT:
            continue
        if report.get("candidate") not in COMPARABLE_CANDIDATES:
            continue
        reports.append(report)
    if not reports:
        raise FileNotFoundError("no compatible normalized RF candidate reports found")
    reports.sort(key=lambda report: report["mae"])
    result = {
        "evaluation_level": "internal_synthetic_target_only",
        "evaluation_contract": EVALUATION_CONTRACT,
        "training_scope": "full_household_corpus",
        "ranking_by_mae": reports,
        "winner": reports[0]["candidate"],
        "winner_artifact": "random_forest.joblib"
        if reports[0]["candidate"] == "random_forest"
        else "",
        "feature_columns": reports[0].get("feature_columns", []),
        "served_forecaster_unchanged": False,
        "release_eligible": False,
        "release_reason": "External validation is required before any serving decision.",
    }
    (destination / "full_corpus_comparison.json").write_text(json.dumps(result, indent=2))
    (destination / "evaluation.json").write_text(json.dumps(result, indent=2))
    LOGGER.info(
        "Compared %d full-corpus candidates; current best is %s",
        len(reports),
        reports[0]["candidate"],
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare full-corpus v3 candidate reports.")
    parser.add_argument("--output", default="models/forecaster_v3")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    compare(args.output)
