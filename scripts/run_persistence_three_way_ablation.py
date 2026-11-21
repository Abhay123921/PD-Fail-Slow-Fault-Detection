from pathlib import Path
import json

import pandas as pd

from pd_failslow.detection.reference import HealthyReferenceModel
from pd_failslow.detection.preprocessing import (
    STAGE_METRICS,
    prepare_stage_residuals,
)
from pd_failslow.detection.anomaly import StageAnomalyDetector
from pd_failslow.detection.persistence import (
    PersistenceDetector,
    TimeAwarePersistenceDetector,
    TimeWindowDensityDetector,
)
from pd_failslow.detection.confidence import EvidenceConfidence
from pd_failslow.diagnosis.attribution import build_attribution_candidates
from pd_failslow.diagnosis.localization import rank_localized_components
from pd_failslow.diagnosis.decision import decide_diagnosis
from pd_failslow.evaluation.detection_eval import evaluate_request_detection
from pd_failslow.evaluation.localization_eval import evaluate_localization


ROOT = Path("results")
HEALTHY_DIR = ROOT / "healthy_characterization" / "traces"
INTEGRATED_DIR = ROOT / "integrated_fault_experiments"
FAULT_TRACE_DIR = INTEGRATED_DIR / "traces"
GT_DIR = INTEGRATED_DIR / "ground_truth"

OUTPUT_DIR = ROOT / "persistence_strict_density_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for folder in ("traces", "candidates", "localization"):
    (OUTPUT_DIR / folder).mkdir(exist_ok=True)

PROFILES = [
    "stable",
    "long_prompt",
    "long_output",
    "high_arrival",
    "mixed",
]

FAULT_SCENARIOS = [
    "healthy",
    "compute",
    "communication",
    "synchronization",
]

FAULT_TYPES = [
    "compute",
    "communication",
    "synchronization",
]

GT_COLUMNS = [
    "fault_id",
    "fault_type",
    "component_type",
    "component_id",
    "start_time_ms",
    "end_time_ms",
    "severity",
]


def load_healthy_seed(seed):
    frames = []

    for profile in PROFILES:
        path = HEALTHY_DIR / f"{profile}_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        frame["evaluation_profile"] = profile
        frames.append(frame)

    boundary_path = HEALTHY_DIR / f"integrated_4096_seed{46 if seed == 42 else 47}.csv"

    if not boundary_path.exists():
        raise FileNotFoundError(boundary_path)

    boundary = pd.read_csv(boundary_path)
    boundary["evaluation_profile"] = "integrated_4096"
    frames.append(boundary)

    return pd.concat(frames, ignore_index=True)


def load_ground_truth(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=GT_COLUMNS)

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=GT_COLUMNS)

    if frame.empty and not set(GT_COLUMNS).issubset(frame.columns):
        return pd.DataFrame(columns=GT_COLUMNS)

    return frame


def load_evaluation_datasets():
    datasets = []

    for profile in PROFILES:
        name = f"healthy_seed44_{profile}"
        path = HEALTHY_DIR / f"{profile}_seed44.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        datasets.append({
            "dataset": "healthy_holdout",
            "name": name,
            "trace": pd.read_csv(path),
            "ground_truth": pd.DataFrame(columns=GT_COLUMNS),
        })

    for scenario in FAULT_SCENARIOS:
        name = f"fault_seed45_{scenario}"
        trace_path = FAULT_TRACE_DIR / f"{name}_raw.csv"
        gt_path = GT_DIR / f"{name}_ground_truth.csv"

        if not trace_path.exists():
            raise FileNotFoundError(trace_path)

        datasets.append({
            "dataset": "fault_scenario",
            "name": name,
            "trace": pd.read_csv(trace_path),
            "ground_truth": load_ground_truth(gt_path),
        })

    return datasets


def fit_detector(train_trace, calibration_trace):
    reference_model = HealthyReferenceModel(
        min_samples=20,
        workload_conditioned=WORKLOAD_CONDITIONED,
    ).fit(train_trace)

    calibration_referenced = reference_model.predict(calibration_trace)
    calibration_residuals = prepare_stage_residuals(
        calibration_referenced
    )

    anomaly_detector = StageAnomalyDetector(
        quantile=0.99,
        min_samples=100,
        min_threshold_ms=1e-6,
    )
    anomaly_detector.fit(calibration_residuals)

    return reference_model, anomaly_detector


def analyze_trace(
    trace,
    reference_model,
    anomaly_detector,
    persistence_strategy,
):
    referenced = reference_model.predict(trace)
    residuals = prepare_stage_residuals(referenced)
    anomalies = anomaly_detector.predict(residuals)

    if persistence_strategy == "count":
        persistent = PersistenceDetector(
            window_size=5,
            min_anomalies=3,
        ).predict(anomalies)
    elif persistence_strategy == "time_aware":
        persistent = TimeAwarePersistenceDetector(
            window_size=5,
            min_anomalies=3,
            max_window_duration_ms=250,
        ).predict(anomalies)
    elif persistence_strategy in (
        "time_window_density",
        "strict_time_window_density",
    ):
        persistent = TimeWindowDensityDetector(
            window_duration_ms=500,
            min_anomalies=3,
            require_current_anomaly=(
                persistence_strategy == "strict_time_window_density"
            ),
        ).predict(anomalies)
    else:
        raise ValueError(
            f"Unknown persistence strategy: {persistence_strategy}"
        )

    confidence = EvidenceConfidence(
        reference_sample_target=100,
        calibration_sample_target=500,
    ).predict(persistent)

    candidates = build_attribution_candidates(confidence)

    localized = rank_localized_components(
        candidates,
        min_evidence=1,
    )

    decision = decide_diagnosis(
        confidence,
        candidates,
        localized,
    )

    return confidence, candidates, localized, decision


def main():
    global WORKLOAD_CONDITIONED

    print("=" * 75)
    print("REFERENCE MODEL x PERSISTENCE STRATEGY ABLATION")
    print("=" * 75)

    print("\nLoading identical training and calibration datasets...")
    train_trace = load_healthy_seed(42)
    calibration_trace = load_healthy_seed(43)
    datasets = load_evaluation_datasets()

    print(f"Training rows: {len(train_trace)}")
    print(f"Calibration rows: {len(calibration_trace)}")
    print(f"Evaluation datasets: {len(datasets)}")

    detection_rows = []
    prediction_frames = []
    localization_rows = []
    localization_detail_frames = []
    decision_rows = []
    threshold_rows = []

    strategies = [
        ("workload_conditioned_count", True, "count"),
        ("workload_conditioned_time_aware", True, "time_aware"),
        ("workload_conditioned_density_500ms", True, "time_window_density"),
        ("global_count", False, "count"),
        ("global_time_aware", False, "time_aware"),
        ("global_density_500ms", False, "time_window_density"),
        (
            "workload_conditioned_strict_density_500ms",
            True,
            "strict_time_window_density",
        ),
        (
            "global_strict_density_500ms",
            False,
            "strict_time_window_density",
        ),
    ]
    for strategy_name, WORKLOAD_CONDITIONED, persistence_strategy in strategies:
        print("\n" + "=" * 75)
        print(f"CONFIGURATION: {strategy_name}")
        print("=" * 75)

        reference_model, anomaly_detector = fit_detector(
            train_trace,
            calibration_trace,
        )

        for metric in STAGE_METRICS:
            threshold_rows.append({
                "strategy": strategy_name,
                "stage": metric,
                "threshold_ms": anomaly_detector.thresholds_.get(metric),
                "calibration_samples": anomaly_detector.sample_counts_.get(
                    metric, 0
                ),
            })

        for item in datasets:
            name = item["name"]
            trace = item["trace"]
            ground_truth = item["ground_truth"]

            analyzed, candidates, localized, decision = analyze_trace(
                trace,
                reference_model,
                anomaly_detector,
                persistence_strategy,
            )
            analyzed.to_csv(
                OUTPUT_DIR / "traces" / f"{strategy_name}_{name}_analyzed.csv",
                index=False,
            )

            candidates.to_csv(
                OUTPUT_DIR / "candidates" / f"{strategy_name}_{name}_candidates.csv",
                index=False,
            )

            localized.to_csv(
                OUTPUT_DIR / "localization" / f"{strategy_name}_{name}_ranking.csv",
                index=False,
            )

            decision_rows.append({
                "strategy": strategy_name,
                "dataset": item["dataset"],
                "name": name,
                **decision,
            })

            # Request-level detection for all three fault types.
            for fault_type in FAULT_TYPES:
                evaluation = evaluate_request_detection(
                    analyzed,
                    ground_truth,
                    fault_type,
                    prediction_mode="persistent",
                )

                detection_rows.append({
                    "strategy": strategy_name,
                    "dataset": item["dataset"],
                    "name": name,
                    "fault_type": fault_type,
                    **evaluation["metrics"],
                })

                details = evaluation["details"].copy()
                details["strategy"] = strategy_name
                details["dataset"] = item["dataset"]
                details["name"] = name
                prediction_frames.append(details)

            # Localization is evaluated only where actual fault GT exists.
            if not ground_truth.empty:
                localization_evaluation = evaluate_localization(
                    localized,
                    ground_truth,
                    top_k=(1, 3),
                )

                localization_rows.append({
                    "strategy": strategy_name,
                    "name": name,
                    **localization_evaluation["metrics"],
                })

                details = localization_evaluation["details"].copy()
                details["strategy"] = strategy_name
                details["name"] = name
                localization_detail_frames.append(details)

            print(
                f"{name:38s} | "
                f"status={decision['status']:20s} | "
                f"direct={decision['direct_anomalous_observation_count']}"
            )

    # Save all results separately from the original experiments.
    pd.DataFrame(detection_rows).to_csv(
        OUTPUT_DIR / "request_detection_metrics.csv",
        index=False,
    )

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "request_detection_predictions.csv",
        index=False,
    )

    pd.DataFrame(localization_rows).to_csv(
        OUTPUT_DIR / "localization_metrics.csv",
        index=False,
    )

    pd.concat(localization_detail_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "localization_details.csv",
        index=False,
    )

    pd.DataFrame(decision_rows).to_csv(
        OUTPUT_DIR / "diagnosis_decisions.csv",
        index=False,
    )

    pd.DataFrame(threshold_rows).to_csv(
        OUTPUT_DIR / "anomaly_thresholds.csv",
        index=False,
    )

    metadata = {
        "experiment": "three_way_persistence_ablation",
        "training_seed": 42,
        "calibration_seed": 43,
        "healthy_holdout_seed": 44,
        "fault_workload_seed": 45,
        "boundary_training_seed": 46,
        "boundary_calibration_seed": 47,
        "reference_min_samples": 20,
        "anomaly_quantile": 0.99,
        "anomaly_min_samples": 100,
        "persistence_window": 5,
        "persistence_min_anomalies": 3,
        "persistence_max_window_duration_ms": 250,
        "persistence_time_window_density_ms": 500,
        "persistence_time_window_density_min_anomalies": 3,
        "persistence_strategies": [
            "count",
            "time_aware",
            "time_window_density",
            "strict_time_window_density",
        ],
        "persistence_require_current_anomaly": True,
        "confidence_reference_sample_target": 100,
        "confidence_calibration_sample_target": 500,
        "reference_strategies": [
            "workload_conditioned",
            "global",
        ],
        "strategies": [
            "workload_conditioned_count",
            "workload_conditioned_time_aware",
            "workload_conditioned_density_500ms",
            "global_count",
            "global_time_aware",
            "global_density_500ms",
            "workload_conditioned_strict_density_500ms",
            "global_strict_density_500ms",
        ],
        "evaluation_uses_identical_saved_traces": True,
        "ground_truth_used_for_detection": False,
        "simulation_type": "deterministic_two_pass_simulator",
    }

    with (OUTPUT_DIR / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 75)
    print("ABLATION COMPLETE")
    print("=" * 75)

    print("\nRequest-level detection metrics:")
    print(pd.DataFrame(detection_rows).to_string(index=False))

    print("\nLocalization metrics:")
    print(pd.DataFrame(localization_rows).to_string(index=False))

    print(f"\nAll results saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
