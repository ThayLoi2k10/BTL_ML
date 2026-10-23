"""Reproducible system benchmarking and trade-off integration for Week 4.

This module belongs to Nhân (Person 4).  It evaluates the three immutable
Week-3 model artifacts on the already preprocessed Week-2 test matrix.  It
never calls ``fit`` and never applies the preprocessing pipeline a second
time.

Run from the repository root with::

    python -m src.evaluation.benchmark

The command writes Nhân-owned JSON evidence and copies (byte-for-byte) the
provisionally recommended Week-3 artifact to ``models/champion_model.pkl``.
The recommendation remains subject to team confirmation.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_SPECS = {
    "linear": {
        "display_name": "Logistic Regression (Tuned)",
        "artifact": MODELS_DIR / "best_linear_model.pkl",
        "interpretability": "High",
        "interpretability_note": (
            "Signed coefficients are available, so direction and relative "
            "magnitude can be inspected directly."
        ),
    },
    "tree": {
        "display_name": "Random Forest (Tuned)",
        "artifact": MODELS_DIR / "best_tree_model.pkl",
        "interpretability": "Medium",
        "interpretability_note": (
            "Global feature importances are available, but the 300-tree "
            "ensemble is not directly traceable as a compact rule set."
        ),
    },
    "nonlinear": {
        "display_name": "SVC RBF (Tuned)",
        "artifact": MODELS_DIR / "best_nonlinear_model.pkl",
        "interpretability": "Low",
        "interpretability_note": (
            "The RBF decision function depends on support vectors and has no "
            "direct global coefficient interpretation."
        ),
    },
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Timing protocol for deterministic input selection and repeated runs."""

    single_sample_index: int = 0
    single_warmup_runs: int = 25
    single_repeats: int = 200
    batch_warmup_runs: int = 3
    batch_repeats: int = 15

    def validate(self) -> None:
        """Reject invalid benchmark settings before loading artifacts."""

        integer_fields = asdict(self)
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer; received {value!r}.")
        if self.single_sample_index < 0:
            raise ValueError("single_sample_index must be non-negative.")
        for name in (
            "single_warmup_runs",
            "single_repeats",
            "batch_warmup_runs",
            "batch_repeats",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without modifying it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_preprocessed_test_features(
    path: str | Path = PROCESSED_DATA_DIR / "X_test.parquet",
) -> pd.DataFrame:
    """Load and validate the immutable, already preprocessed test features."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed test features not found: {path}")

    X_test = pd.read_parquet(path)
    if not isinstance(X_test, pd.DataFrame) or X_test.empty:
        raise ValueError("X_test must be a non-empty pandas DataFrame.")
    if X_test.columns.duplicated().any():
        duplicates = X_test.columns[X_test.columns.duplicated()].tolist()
        raise ValueError(f"X_test contains duplicate columns: {duplicates}")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in X_test.dtypes):
        non_numeric = X_test.select_dtypes(exclude=[np.number]).columns.tolist()
        raise TypeError(
            "Saved Week-3 models require preprocessed numeric features; "
            f"non-numeric columns found: {non_numeric}"
        )

    values = X_test.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("X_test contains NaN or infinite values.")
    return X_test


def load_saved_models() -> dict[str, Any]:
    """Load the three Week-3 estimators without fitting or changing them."""

    models: dict[str, Any] = {}
    for family, spec in MODEL_SPECS.items():
        artifact = Path(spec["artifact"])
        if not artifact.exists():
            raise FileNotFoundError(f"Missing Week-3 model artifact: {artifact}")
        model = joblib.load(artifact)
        if not hasattr(model, "predict"):
            raise TypeError(f"{artifact} does not expose an sklearn predict method.")
        models[family] = model
    return models


def verify_model_contract(model: Any, X_test: pd.DataFrame) -> dict[str, Any]:
    """Verify that an estimator consumes the processed schema exactly once."""

    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is None:
        raise ValueError(
            f"{type(model).__name__} has no n_features_in_; input contract is unknown."
        )
    if int(expected_count) != X_test.shape[1]:
        raise ValueError(
            f"{type(model).__name__} expects {expected_count} features, but "
            f"X_test has {X_test.shape[1]}."
        )

    expected_names = getattr(model, "feature_names_in_", None)
    names_match: bool | None = None
    if expected_names is not None:
        names_match = list(expected_names) == list(X_test.columns)
        if not names_match:
            raise ValueError(
                f"{type(model).__name__} feature names/order do not match X_test."
            )

    # A Week-3 estimator is expected here, not the raw-data preprocessing
    # pipeline.  This check prevents accidental duplicate transformation.
    is_sklearn_pipeline = hasattr(model, "steps")
    if is_sklearn_pipeline:
        raise ValueError(
            f"{type(model).__name__} appears to be a Pipeline; benchmark input "
            "would need re-auditing before use."
        )

    return {
        "verified": True,
        "input_representation": "preprocessed_numeric_features",
        "preprocessing_applied_by_benchmark": False,
        "n_features_expected": int(expected_count),
        "feature_names_checked": expected_names is not None,
        "feature_names_match": names_match,
        "estimator_is_pipeline": False,
    }


def _distribution(values: list[float]) -> dict[str, float]:
    """Summarize repeated timings without hiding run-to-run variation."""

    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def benchmark_estimator(
    model: Any,
    X_test: pd.DataFrame,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Measure single-row latency and full-test batch throughput.

    Timing uses ``perf_counter_ns``.  Predict calls are warmed up, inputs are
    fixed, and garbage collection is paused only while the measured loops run.
    The loaded estimator's saved execution settings (including ``n_jobs``) are
    preserved.
    """

    if config.single_sample_index >= len(X_test):
        raise IndexError(
            f"single_sample_index={config.single_sample_index} exceeds "
            f"X_test length {len(X_test)}."
        )

    X_single = X_test.iloc[[config.single_sample_index]]
    for _ in range(config.single_warmup_runs):
        model.predict(X_single)
    for _ in range(config.batch_warmup_runs):
        model.predict(X_test)

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        single_ms: list[float] = []
        for _ in range(config.single_repeats):
            started = perf_counter_ns()
            prediction = model.predict(X_single)
            elapsed_ns = perf_counter_ns() - started
            if len(prediction) != 1:
                raise RuntimeError("Single-sample prediction returned wrong length.")
            single_ms.append(elapsed_ns / 1_000_000.0)

        batch_ms: list[float] = []
        throughput: list[float] = []
        for _ in range(config.batch_repeats):
            started = perf_counter_ns()
            prediction = model.predict(X_test)
            elapsed_ns = perf_counter_ns() - started
            if len(prediction) != len(X_test):
                raise RuntimeError("Batch prediction returned wrong length.")
            elapsed_seconds = elapsed_ns / 1_000_000_000.0
            batch_ms.append(elapsed_ns / 1_000_000.0)
            throughput.append(len(X_test) / elapsed_seconds)
    finally:
        if gc_was_enabled:
            gc.enable()

    return {
        "single_sample_latency_ms": {
            **_distribution(single_ms),
            "warmup_runs": config.single_warmup_runs,
            "measured_runs": config.single_repeats,
            "sample_index": config.single_sample_index,
        },
        "batch_latency_ms": {
            **_distribution(batch_ms),
            "warmup_runs": config.batch_warmup_runs,
            "measured_runs": config.batch_repeats,
            "samples_per_batch": len(X_test),
        },
        "batch_throughput_samples_per_second": {
            **_distribution(throughput),
            "measured_runs": config.batch_repeats,
            "samples_per_batch": len(X_test),
        },
    }


def recover_recorded_week3_times() -> dict[str, dict[str, Any]]:
    """Recover only timing values actually persisted by Week-3 work."""

    recovered: dict[str, dict[str, Any]] = {
        family: {
            "recorded_training_seconds": None,
            "recorded_tuning_seconds": None,
            "notes": [],
        }
        for family in MODEL_SPECS
    }

    hyperparameters_path = MODELS_DIR / "best_hyperparameters.json"
    if hyperparameters_path.exists():
        with hyperparameters_path.open("r", encoding="utf-8") as handle:
            hyperparameters = json.load(handle)
        for family in MODEL_SPECS:
            elapsed = hyperparameters.get(family, {}).get("elapsed_seconds")
            if elapsed is not None:
                recovered[family]["recorded_tuning_seconds"] = float(elapsed)
            note = hyperparameters.get(family, {}).get("elapsed_note")
            if note:
                recovered[family]["notes"].append(str(note))

    nonlinear_report = REPORTS_DIR / "nonlinear_model_benchmark.csv"
    if nonlinear_report.exists():
        table = pd.read_csv(nonlinear_report)
        time_column = "Thời gian huấn luyện full Train (s)"
        model_column = "Mô hình"
        if time_column in table.columns and model_column in table.columns:
            tuned = table[table[model_column].astype(str).str.contains("Tuned")]
            if not tuned.empty and pd.notna(tuned.iloc[0][time_column]):
                recovered["nonlinear"]["recorded_training_seconds"] = float(
                    tuned.iloc[0][time_column]
                )
                recovered["nonlinear"]["notes"].append(
                    "Full-train fit time recorded in reports/nonlinear_model_benchmark.csv."
                )

    for family in ("linear", "tree"):
        if recovered[family]["recorded_training_seconds"] is None:
            recovered[family]["notes"].append(
                "Final full-train fit time was not persisted by the Week-3 implementation."
            )
    return recovered


def environment_conditions() -> dict[str, Any]:
    """Record the software and hardware context needed to interpret timings."""

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "Unavailable from platform API",
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }


def _write_json(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def run_system_benchmarks(
    config: BenchmarkConfig | None = None,
    output_json: str | Path | None = REPORTS_DIR / "week4_system_benchmark.json",
) -> dict[str, Any]:
    """Benchmark all immutable Week-3 artifacts on processed ``X_test``."""

    config = config or BenchmarkConfig()
    config.validate()
    X_test_path = PROCESSED_DATA_DIR / "X_test.parquet"
    X_test = load_preprocessed_test_features(X_test_path)
    models = load_saved_models()
    recorded_times = recover_recorded_week3_times()

    records: list[dict[str, Any]] = []
    for family, model in models.items():
        spec = MODEL_SPECS[family]
        artifact = Path(spec["artifact"])
        contract = verify_model_contract(model, X_test)
        timings = benchmark_estimator(model, X_test, config)
        records.append(
            {
                "family": family,
                "model_name": spec["display_name"],
                "estimator_class": type(model).__name__,
                "artifact_path": artifact.relative_to(PROJECT_ROOT).as_posix(),
                "artifact_sha256": sha256_file(artifact),
                "model_size_bytes": artifact.stat().st_size,
                "model_size_mb": artifact.stat().st_size / 1_000_000.0,
                "input_contract": contract,
                **recorded_times[family],
                **timings,
            }
        )

    payload = {
        "schema_version": 1,
        "scope": "Week 4 Nhân system benchmark; prediction only; no fitting",
        "environment": environment_conditions(),
        "methodology": {
            "timer": "time.perf_counter_ns",
            "input": "data/processed/X_test.parquet",
            "input_is_preprocessed": True,
            "preprocessing_reapplied": False,
            "garbage_collection_paused_during_measured_loops": True,
            "saved_estimator_execution_settings_preserved": True,
            "model_size_unit": "decimal MB (1 MB = 1,000,000 bytes)",
            **asdict(config),
        },
        "test_data": {
            "path": X_test_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(X_test_path),
            "rows": len(X_test),
            "columns": X_test.shape[1],
            "all_numeric": True,
            "finite_values_only": True,
        },
        "models": records,
    }
    if output_json is not None:
        _write_json(payload, output_json)
    return payload


def build_tradeoff_matrix(
    benchmark_payload: dict[str, Any],
    evaluation_csv: str | Path = REPORTS_DIR / "evaluation_benchmark.csv",
    output_json: str | Path | None = REPORTS_DIR / "week4_tradeoff_matrix.json",
) -> list[dict[str, Any]]:
    """Join Lợi's held-out metrics with Nhân's operational measurements."""

    evaluation_csv = Path(evaluation_csv)
    if not evaluation_csv.exists():
        raise FileNotFoundError(
            "Person 1 evaluation summary is required for the trade-off matrix: "
            f"{evaluation_csv}"
        )
    evaluation = pd.read_csv(evaluation_csv).set_index("Model_Key")

    records: list[dict[str, Any]] = []
    for model_record in benchmark_payload["models"]:
        family = model_record["family"]
        if family not in evaluation.index:
            raise KeyError(f"No Week-4 evaluation row found for model family {family!r}.")
        metric = evaluation.loc[family]
        spec = MODEL_SPECS[family]
        records.append(
            {
                "family": family,
                "model_name": model_record["model_name"],
                "cv_macro_f1": float(metric["CV_Macro_F1 (W3)"]),
                "test_macro_f1": float(metric["Test_Macro_F1 (W4)"]),
                "generalization_gap_cv_minus_test": float(
                    metric["Generalization_Gap"]
                ),
                "test_accuracy": float(metric["Test_Accuracy"]),
                "test_precision_yes": float(metric["Test_Precision"]),
                "test_recall_yes": float(metric["Test_Recall"]),
                "test_roc_auc": float(metric["Test_ROC_AUC"]),
                "test_pr_auc": float(metric["Test_PR_AUC"]),
                "single_sample_latency_ms_median": model_record[
                    "single_sample_latency_ms"
                ]["median"],
                "single_sample_latency_ms_p95": model_record[
                    "single_sample_latency_ms"
                ]["p95"],
                "batch_throughput_samples_per_second_median": model_record[
                    "batch_throughput_samples_per_second"
                ]["median"],
                "batch_latency_ms_median": model_record["batch_latency_ms"][
                    "median"
                ],
                "model_size_mb": model_record["model_size_mb"],
                "recorded_training_seconds": model_record[
                    "recorded_training_seconds"
                ],
                "recorded_tuning_seconds": model_record[
                    "recorded_tuning_seconds"
                ],
                "interpretability": spec["interpretability"],
                "interpretability_note": spec["interpretability_note"],
            }
        )

    if output_json is not None:
        _write_json(records, output_json)
    return records


def recommend_provisional_champion(
    tradeoff_matrix: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recommend, but do not team-confirm, a model for batch campaign scoring."""

    if not tradeoff_matrix:
        raise ValueError("tradeoff_matrix must contain at least one model.")
    by_family = {record["family"]: record for record in tradeoff_matrix}
    required = set(MODEL_SPECS)
    if set(by_family) != required:
        missing = sorted(required - set(by_family))
        raise ValueError(f"Trade-off matrix is missing model families: {missing}")

    # The project's declared primary metric is Macro-F1.  For the bank
    # marketing use case, scoring is naturally batch-oriented.  Operational
    # measurements are therefore checked as feasibility/trade-off evidence,
    # not used to invent an undocumented latency SLA.
    selected = max(tradeoff_matrix, key=lambda row: row["test_macro_f1"])
    runner_up = sorted(
        tradeoff_matrix,
        key=lambda row: row["test_macro_f1"],
        reverse=True,
    )[1]

    return {
        "status": "provisional_recommendation_pending_team_confirmation",
        "selected_family": selected["family"],
        "selected_model": selected["model_name"],
        "selection_context": "offline_or_batch_bank_marketing_scoring",
        "primary_metric": "test_macro_f1",
        "evidence": {
            "test_macro_f1": selected["test_macro_f1"],
            "generalization_gap_cv_minus_test": selected[
                "generalization_gap_cv_minus_test"
            ],
            "test_recall_yes": selected["test_recall_yes"],
            "single_sample_latency_ms_median": selected[
                "single_sample_latency_ms_median"
            ],
            "batch_throughput_samples_per_second_median": selected[
                "batch_throughput_samples_per_second_median"
            ],
            "model_size_mb": selected["model_size_mb"],
            "interpretability": selected["interpretability"],
            "macro_f1_advantage_over_runner_up": (
                selected["test_macro_f1"] - runner_up["test_macro_f1"]
            ),
        },
        "rationale": (
            "The selected model has the highest held-out Macro-F1 and class-yes "
            "recall, a small CV-to-test gap, and measured batch throughput that "
            "supports campaign scoring on the available hardware. Its larger "
            "artifact and medium interpretability remain material trade-offs."
        ),
        "limitations": [
            "No documented team approval was found; this is not a confirmed team decision.",
            "The application has no documented latency, memory, or throughput SLA.",
            "Person 2's Week-4 plots module/output was not available on the inspected main branch.",
        ],
    }


def copy_provisional_champion(
    recommendation: dict[str, Any],
    output_path: str | Path = MODELS_DIR / "champion_model.pkl",
) -> dict[str, Any]:
    """Copy the selected artifact byte-for-byte, preserving all originals."""

    family = recommendation.get("selected_family")
    if family not in MODEL_SPECS:
        raise ValueError(f"Unknown selected_family: {family!r}")
    source = Path(MODEL_SPECS[family]["artifact"])
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Champion destination must not overwrite a Week-3 artifact.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise RuntimeError("Champion copy failed SHA-256 integrity verification.")
    return {
        "source_artifact": source.relative_to(PROJECT_ROOT).as_posix(),
        "champion_artifact": destination.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": destination_hash,
        "byte_identical_to_source": True,
        "original_artifact_preserved": True,
    }


def run_week4_nhan(
    config: BenchmarkConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Run Nhân's complete benchmark, comparison, and provisional selection."""

    benchmark_payload = run_system_benchmarks(config=config)
    tradeoff_matrix = build_tradeoff_matrix(benchmark_payload)
    recommendation = recommend_provisional_champion(tradeoff_matrix)
    recommendation["artifact_copy"] = copy_provisional_champion(recommendation)

    # Persist the recommendation beside the matrix without changing the matrix
    # schema consumed by the notebook.
    recommendation_path = REPORTS_DIR / "week4_champion_recommendation.json"
    _write_json(recommendation, recommendation_path)
    return benchmark_payload, tradeoff_matrix, recommendation


def main() -> None:
    """Execute the default reproducible benchmark protocol."""

    benchmark_payload, tradeoff_matrix, recommendation = run_week4_nhan()
    print("Week-4 system benchmark completed without fitting any model.")
    print(
        f"Test matrix: {benchmark_payload['test_data']['rows']} rows x "
        f"{benchmark_payload['test_data']['columns']} columns"
    )
    print("\nTrade-off matrix:")
    columns = [
        "family",
        "test_macro_f1",
        "generalization_gap_cv_minus_test",
        "single_sample_latency_ms_median",
        "batch_throughput_samples_per_second_median",
        "model_size_mb",
        "interpretability",
    ]
    print(pd.DataFrame(tradeoff_matrix)[columns].to_string(index=False))
    print(
        "\nProvisional recommendation: "
        f"{recommendation['selected_model']} "
        "(pending team confirmation)."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BenchmarkConfig",
    "build_tradeoff_matrix",
    "copy_provisional_champion",
    "load_preprocessed_test_features",
    "load_saved_models",
    "recommend_provisional_champion",
    "run_system_benchmarks",
    "run_week4_nhan",
    "verify_model_contract",
]
