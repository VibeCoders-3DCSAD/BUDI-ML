"""
Odin ML Synthetic Data Pipeline — DEPRECATED

This script is deprecated. Use preprocessor.py instead, which runs
synthesis and preprocessing in a single step:

    python scripts/preprocessor.py --input datasets/unprocessed/puf.parquet

This script remains for standalone use only. All synthesis logic has been
moved into preprocessor.py as _run_synthesis().
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts directory to path for imports
from training.scripts.data.fies_columns import (
    FIESColumns,
    FIESColumnError,
    FIESColumnWarning,
    validate_columns,
)
from training.scripts.pipeline.generate_personas import (
    SyntheticPersona,
    PersonaGenerationError,
    load_fies_data,
    filter_ncr,
    compute_fies_statistics,
    compute_expense_ratios,
    generate_all_personas,
    validate_personas,
    export_personas,
)
from training.scripts.pipeline.generate_transactions import (
    TransactionGenerationError,
    PersonaLoadError,
    generate_persona_transactions,
    validate_transactions,
    export_transactions,
)


class PipelineError(Exception):
    """Raised when pipeline fails."""
    pass


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {text}")
    print(f"{'=' * 60}")


def print_step(step: int, text: str) -> None:
    """Print formatted step."""
    print(f"\n[{step}/6] {text}")


def print_warning(text: str) -> None:
    """Print formatted warning."""
    print(f"  WARNING: {text}")


def print_error(text: str) -> None:
    """Print formatted error."""
    print(f"  ERROR: {text}")


def run_pipeline(
    input_path: str,
    output_path: str,
    personas_per_archetype: int = 1000,
    num_months: int = 12,
    seed: int = 42,
    skip_fies: bool = False,
    strict: bool = False,
) -> dict:
    """
    Run the full synthetic data pipeline.

    Args:
        input_path: Path to FIES data file (CSV or Parquet)
        output_path: Output directory
        personas_per_archetype: Number of personas per archetype
        num_months: Number of months to generate
        seed: Random seed
        skip_fies: Skip loading FIES data
        strict: Strict mode for column validation

    Returns:
        Pipeline report dict

    Raises:
        PipelineError: If pipeline fails
    """
    start_time = time.time()
    results = {
        "status": "started",
        "errors": [],
        "warnings": [],
    }

    print_header("Odin ML Synthetic Data Pipeline")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Personas per archetype: {personas_per_archetype}")
    print(f"Months: {num_months}")
    print(f"Seed: {seed}")
    print(f"Strict mode: {strict}")

    # Step 1: Load FIES data
    print_step(1, "Loading FIES NCR data")
    fies_stats = None
    expense_ratios = None

    if not skip_fies:
        try:
            df = load_fies_data(input_path)
            
            # Validate columns
            print("  Validating FIES columns...")
            report = validate_columns(df, strict=strict)
            
            print(f"  Found {report['found']}/{report['total_columns']} expected columns")
            
            if report["missing_ids"]:
                print_warning(f"Missing columns: {report['missing_ids']}")
                results["warnings"].append(f"Missing FIES columns: {report['missing_ids']}")
            
            if report["alternatives_used"]:
                print(f"  Using {len(report['alternatives_used'])} alternative column names")
                for alt in report["alternatives_used"]:
                    print(f"    {alt['fies_id']}: using '{alt['using']}' instead of '{alt['expected']}'")
            
            if not report["is_valid"]:
                missing_critical = report["critical_missing"]
                print_error(f"Missing critical columns: {missing_critical}")
                results["errors"].append(f"Missing critical columns: {missing_critical}")
                
                if strict:
                    raise PipelineError(
                        f"Strict mode: Missing critical columns: {missing_critical}\n"
                        "Use --skip-fies to bypass FIES data validation"
                    )
            
            ncr_df = filter_ncr(df)
            fies_stats = compute_fies_statistics(ncr_df)
            expense_ratios = compute_expense_ratios(ncr_df)
            
            results["fies_loaded"] = True
            results["fies_rows"] = len(ncr_df)
            results["fies_columns"] = report

        except FileNotFoundError as e:
            print_error(f"File not found: {e}")
            results["errors"].append(f"File not found: {e}")
            print("  Falling back to default statistics")
        except FIESColumnError as e:
            print_error(f"FIES column error: {e}")
            results["errors"].append(f"FIES column error: {e}")
            print("  Falling back to default statistics")
        except PipelineError:
            raise
        except Exception as e:
            print_error(f"Unexpected error loading FIES data: {e}")
            results["errors"].append(f"Unexpected error: {e}")
            print("  Falling back to default statistics")
    else:
        print("  Skipping FIES data loading (--skip-fies)")
        results["fies_loaded"] = False

    # Step 2: Generate personas
    print_step(2, "Generating synthetic personas")
    try:
        personas = generate_all_personas(
            personas_per_archetype=personas_per_archetype,
            fies_stats=fies_stats,
            expense_ratios=expense_ratios,
            seed=seed,
        )

        # Validate personas
        persona_validation = validate_personas(personas)
        results["persona_count"] = persona_validation["total_personas"]
        results["persona_validation"] = persona_validation

        # Export personas
        export_personas(personas, str(output_path))
        
    except PersonaGenerationError as e:
        print_error(f"Persona generation failed: {e}")
        results["errors"].append(f"Persona generation failed: {e}")
        raise PipelineError(f"Persona generation failed: {e}")
    except Exception as e:
        print_error(f"Unexpected error generating personas: {e}")
        results["errors"].append(f"Unexpected error: {e}")
        raise PipelineError(f"Unexpected error generating personas: {e}")

    # Step 3: Generate transactions
    print_step(3, "Generating transaction histories")
    all_transactions = []
    all_summaries = []
    failed_personas = 0

    for i, persona in enumerate(personas):
        if (i + 1) % 1000 == 0:
            print(f"  Processing persona {i + 1:,}/{len(personas):,}...")

        try:
            # Convert dataclass to dict if needed
            persona_dict = asdict(persona) if isinstance(persona, SyntheticPersona) else persona

            transactions, summaries = generate_persona_transactions(
                persona_dict,
                start_year=2023,
                start_month=1,
                num_months=num_months,
                seed=seed,
            )

            all_transactions.extend(transactions)
            all_summaries.extend(summaries)
        except TransactionGenerationError as e:
            warnings.warn(f"Failed for persona {i}: {e}")
            failed_personas += 1
        except Exception as e:
            warnings.warn(f"Unexpected error for persona {i}: {e}")
            failed_personas += 1

    results["transaction_count"] = len(all_transactions)
    results["summary_count"] = len(all_summaries)
    results["failed_personas"] = failed_personas

    if failed_personas > 0:
        print_warning(f"{failed_personas} personas failed to generate transactions")
        results["warnings"].append(f"{failed_personas} personas failed")

    # Step 4: Validate transactions
    print_step(4, "Validating transaction data")
    transaction_validation = validate_transactions(all_summaries)
    results["transaction_validation"] = transaction_validation

    # Step 5: Export transactions
    print_step(5, "Exporting transaction data")
    try:
        export_transactions(all_transactions, all_summaries, str(output_path))
    except Exception as e:
        print_error(f"Failed to export transactions: {e}")
        results["errors"].append(f"Export failed: {e}")

    # Step 6: Generate summary report
    print_step(6, "Generating summary report")
    summary_report = generate_summary_report(results, personas, all_summaries)

    # Save summary report
    report_path = Path(output_path) / "pipeline_report.json"
    try:
        with open(report_path, "w") as f:
            json.dump(summary_report, f, indent=2, cls=NumpyEncoder)
        print(f"Saved pipeline report to {report_path}")
    except Exception as e:
        print_error(f"Failed to save report: {e}")

    # Print summary
    elapsed_time = time.time() - start_time
    print_header("Pipeline Complete")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Personas generated: {results.get('persona_count', 0):,}")
    print(f"Transactions generated: {results.get('transaction_count', 0):,}")
    print(f"Monthly summaries: {results.get('summary_count', 0):,}")
    
    if results.get("errors"):
        print(f"\nErrors encountered: {len(results['errors'])}")
        for error in results["errors"]:
            print(f"  - {error}")
    
    if results.get("warnings"):
        print(f"\nWarnings: {len(results['warnings'])}")
    
    print(f"\nOutput directory: {output_path}")
    print(f"  personas.json - All personas")
    print(f"  personas.parquet - All personas (Parquet)")
    print(f"  transactions.parquet - All transactions")
    print(f"  monthly_summaries.parquet - Monthly summaries")
    print(f"  pipeline_report.json - This report")

    return summary_report


def generate_summary_report(
    results: dict,
    personas: list,
    summaries: list,
) -> dict:
    """Generate a comprehensive summary report."""
    report = {
        "pipeline_version": "1.0.0",
        "timestamp": pd.Timestamp.now().isoformat(),
        "status": "completed" if not results.get("errors") else "completed_with_errors",
        "parameters": {
            "personas_per_archetype": results.get("persona_count", 0) // 14 if results.get("persona_count") else 0,
            "num_months": 12,
            "seed": 42,
        },
        "fies_data": {
            "loaded": results.get("fies_loaded", False),
            "rows": results.get("fies_rows", 0),
            "column_validation": results.get("fies_columns", {}),
        },
        "generated_data": {
            "total_personas": results.get("persona_count", 0),
            "total_transactions": results.get("transaction_count", 0),
            "total_summaries": results.get("summary_count", 0),
            "failed_personas": results.get("failed_personas", 0),
        },
        "persona_distribution": results.get("persona_validation", {}),
        "transaction_statistics": results.get("transaction_validation", {}),
        "pfp_label_distribution": {},
        "archetype_distribution": {},
        "errors": results.get("errors", []),
        "warnings": results.get("warnings", []),
    }

    # Compute PFP label distribution
    for persona in personas:
        label = persona.pfp_label if isinstance(persona, SyntheticPersona) else persona.get("pfp_label", "Unknown")
        report["pfp_label_distribution"][label] = (
            report["pfp_label_distribution"].get(label, 0) + 1
        )

    # Compute archetype distribution
    for persona in personas:
        archetype = persona.archetype_id if isinstance(persona, SyntheticPersona) else persona.get("archetype_id", "Unknown")
        report["archetype_distribution"][archetype] = (
            report["archetype_distribution"].get(archetype, 0) + 1
        )

    # Add anomaly statistics
    if summaries:
        anomalous_months = sum(1 for s in summaries if s.is_anomalous)
        report["anomaly_statistics"] = {
            "total_anomalous_months": anomalous_months,
            "anomaly_rate": anomalous_months / len(summaries) if summaries else 0,
        }
    else:
        report["anomaly_statistics"] = {
            "total_anomalous_months": 0,
            "anomaly_rate": 0,
        }

    return report


def main():
    warnings.warn(
        "synthesizer.py is deprecated. Use preprocessor.py instead:\n"
        "  python scripts/preprocessor.py --input datasets/unprocessed/puf.parquet",
        DeprecationWarning,
        stacklevel=1,
    )
    parser = argparse.ArgumentParser(
        description="Odin ML Synthetic Data Pipeline (DEPRECATED — use preprocessor.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline with parquet data (produced by collector.py)
    python scripts/synthesizer.py --input datasets/unprocessed/puf.parquet

    # Run full pipeline with legacy CSV data
    python scripts/synthesizer.py --input datasets/raw/family_income_and_expenditure.csv

    # Run pipeline with default statistics (skip FIES validation)
    python scripts/synthesizer.py --skip-fies

    # Run with custom parameters
    python scripts/synthesizer.py --personas-per-archetype 500 --months 6 --seed 123

    # Run in strict mode (fail on missing critical columns)
    python scripts/synthesizer.py --strict
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/unprocessed/puf.parquet",
        help="Path to FIES data file (CSV or Parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synth/",
        help="Output directory for all generated data",
    )
    parser.add_argument(
        "--personas-per-archetype",
        type=int,
        default=1000,
        help="Number of personas to generate per archetype (default: 1000)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of months to generate (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-fies",
        action="store_true",
        help="Skip loading FIES data and use default statistics",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: fail on missing critical columns",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total number of personas (useful for testing)",
    )

    args = parser.parse_args()

    # Adjust personas per archetype if limit is set
    personas_per_archetype = args.personas_per_archetype
    if args.limit:
        personas_per_archetype = args.limit // 14 + 1
        print(f"Limiting to {args.limit} personas ({personas_per_archetype} per archetype)")

    try:
        report = run_pipeline(
            input_path=args.input,
            output_path=args.output,
            personas_per_archetype=personas_per_archetype,
            num_months=args.months,
            seed=args.seed,
            skip_fies=args.skip_fies,
            strict=args.strict,
        )
        
        # Exit with error code if there were errors
        if report.get("errors"):
            print("\nPipeline completed with errors.")
            sys.exit(1)
        else:
            print("\nPipeline completed successfully!")
            sys.exit(0)
            
    except PipelineError as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected pipeline failure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
