from pathlib import Path
from collections import Counter
import copy
import json

import pandas as pd

from pd_failslow.utils.config import load_experiment_config, build_system
from pd_failslow.workload.generator import generate_requests
from pd_failslow.simulator.engine import PDSimulationEngine
from pd_failslow.faults.injector import FaultInjector
from pd_failslow.faults.communication_faults import CommunicationFault

from pd_failslow.detection.reference import HealthyReferenceModel
from pd_failslow.detection.preprocessing import prepare_stage_residuals
from pd_failslow.detection.anomaly import StageAnomalyDetector
from pd_failslow.detection.persistence import TimeWindowDensityDetector
from pd_failslow.detection.confidence import EvidenceConfidence

from pd_failslow.diagnosis.attribution import build_attribution_candidates
from pd_failslow.diagnosis.localization import rank_localized_components
from pd_failslow.diagnosis.decision import decide_diagnosis


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

ROOT = Path("results")
HEALTHY_DIR = ROOT / "healthy_characterization" / "traces"
OUTPUT_DIR = ROOT / "balanced_two_link_localization_link_conditioned"

NUM_REQUESTS = 1000
WORKLOAD_SEED = 45
FAULT_SEVERITY = 0.50
FAULT_START_MS = 0.0
FAULT_END_MS = 1_000_000.0
KV_CACHE_MB_PER_PROMPT_TOKEN = 0.01

PAIRS = [
    ("P1", "D1"),
    ("P1", "D2"),
    ("P2", "D1"),
    ("P2", "D2"),
]

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. Preserve existing results "
        "and choose a new output directory before rerunning."
    )

OUTPUT_DIR.mkdir(parents=True)
for folder in ("traces", "candidates", "localization", "ground_truth"):
    (OUTPUT_DIR / folder).mkdir()

config = load_experiment_config("configs")


# ---------------------------------------------------------
# Healthy reference training and calibration
# Same datasets and seeds as the selected ablation strategy
# ---------------------------------------------------------

def load_healthy_seed(seed):
    frames = []

    for profile in [
        "stable",
        "long_prompt",
        "long_output",
        "high_arrival",
        "mixed",
    ]:
        path = HEALTHY_DIR / f"{profile}_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        frame["evaluation_profile"] = profile
        frames.append(frame)

    boundary_seed = 46 if seed == 42 else 47
    boundary_path = (
        HEALTHY_DIR / f"integrated_4096_seed{boundary_seed}.csv"
    )

    if not boundary_path.exists():
        raise FileNotFoundError(boundary_path)

    boundary = pd.read_csv(boundary_path)
    boundary["evaluation_profile"] = "integrated_4096"
    frames.append(boundary)

    return pd.concat(frames, ignore_index=True)


print("=" * 70)
print("BALANCED TWO-LINK COMMUNICATION LOCALIZATION")
print("=" * 70)

print("\nLoading healthy training and calibration datasets...")
train_trace = load_healthy_seed(42)
calibration_trace = load_healthy_seed(43)

print(f"Training rows: {len(train_trace)}")
print(f"Calibration rows: {len(calibration_trace)}")

reference_model = HealthyReferenceModel(
    min_samples=20,
    workload_conditioned=True,
).fit(train_trace)

calibration_referenced = reference_model.predict(calibration_trace)
calibration_residuals = prepare_stage_residuals(
    calibration_referenced
)

anomaly_detector = StageAnomalyDetector(
    quantile=0.99,
    min_samples=100,
    min_threshold_ms=1e-6,
    link_conditioned_transfer=True,
)
anomaly_detector.fit(calibration_residuals)

threshold_records = []

for metric, threshold in anomaly_detector.thresholds_.items():
    threshold_records.append({
        "stage": metric,
        "calibration_scope": "global",
        "link_type": None,
        "configured_bandwidth_mb_per_ms": None,
        "threshold_ms": threshold,
        "calibration_samples": anomaly_detector.sample_counts_.get(
            metric, 0
        ),
    })

for key, threshold in anomaly_detector.link_thresholds_.items():
    link_type, bandwidth = key
    threshold_records.append({
        "stage": "transfer_time_ms",
        "calibration_scope": "link_conditioned",
        "link_type": link_type,
        "configured_bandwidth_mb_per_ms": bandwidth,
        "threshold_ms": threshold,
        "calibration_samples": anomaly_detector.link_sample_counts_[key],
    })

pd.DataFrame(threshold_records).to_csv(
    OUTPUT_DIR / "anomaly_thresholds.csv",
    index=False,
)


# ---------------------------------------------------------
# Matched workload and deterministic worker-pair assignment
# ---------------------------------------------------------

print("\nGenerating matched workload...")

requests = generate_requests(
    num_requests=NUM_REQUESTS,
    seed=WORKLOAD_SEED,
    arrival_distribution="constant",
    arrival_rate_per_ms=0.02,
    prompt_distribution="fixed",
    min_prompt_tokens=4096,
    max_prompt_tokens=4096,
    fixed_prompt_tokens=4096,
    output_distribution="fixed",
    min_output_tokens=128,
    max_output_tokens=128,
    fixed_output_tokens=128,
)

if len(requests) != NUM_REQUESTS:
    raise RuntimeError("Unexpected request count.")

worker_pair_assignments = {}

for request in requests:
    try:
        request_index = int(request.request_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        raise ValueError(
            f"Unexpected request ID format: {request.request_id}"
        )

    if not 0 <= request_index < NUM_REQUESTS:
        raise ValueError(f"Unexpected request index: {request_index}")

    pair_index = request_index % len(PAIRS)
    worker_pair_assignments[request.request_id] = PAIRS[pair_index]

assignment_counts = Counter(worker_pair_assignments.values())

for pair in PAIRS:
    if assignment_counts[pair] != 250:
        raise RuntimeError(
            f"Expected 250 assignments for {pair}, "
            f"found {assignment_counts[pair]}"
        )

print("Worker-pair assignments:")
for pair in PAIRS:
    print(f"  {pair[0]} -> {pair[1]}: {assignment_counts[pair]}")

print("Expected link distribution: L1=500, L2=500")


# ---------------------------------------------------------
# Fault injection
# ---------------------------------------------------------

def create_fault_injector(scenario):
    if scenario == "healthy":
        return None

    fault_targets = {
        "l1_fault": "L1",
        "l2_fault": "L2",
    }
    if scenario not in fault_targets:
        raise ValueError(f"Unknown scenario: {scenario}")

    link_id = fault_targets[scenario]
    injector = FaultInjector()
    injector.register_communication_fault(
        CommunicationFault(
            fault_id=f"balanced_communication_{link_id}",
            link_id=link_id,
            start_time_ms=FAULT_START_MS,
            end_time_ms=FAULT_END_MS,
            severity=FAULT_SEVERITY,
        )
    )
    return injector


# ---------------------------------------------------------
# Detector, persistence, attribution and localization
# ---------------------------------------------------------

def analyze_trace(trace):
    referenced = reference_model.predict(trace)
    residuals = prepare_stage_residuals(referenced)
    anomalies = anomaly_detector.predict(residuals)

    persistent = TimeWindowDensityDetector(
        window_duration_ms=500,
        min_anomalies=3,
        require_current_anomaly=True,
    ).predict(anomalies)

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


# ---------------------------------------------------------
# Link-specific evaluation
# ---------------------------------------------------------

def evaluate_links(analyzed, scenario):
    records = []
    fault_target = {
        "l1_fault": "L1",
        "l2_fault": "L2",
    }.get(scenario)

    for link_id in ("L1", "L2"):
        subset = analyzed.loc[
            analyzed["link_id"].eq(link_id)
        ].copy()

        anomaly_status = subset["transfer_time_ms_anomaly_status"]
        persistence_status = subset[
            "transfer_time_ms_persistence_status"
        ]

        raw_valid = anomaly_status.eq("scored")
        persistent_valid = persistence_status.eq("evaluated")

        raw_flags = (
            subset.loc[raw_valid, "transfer_time_ms_anomaly"]
            .fillna(False)
            .astype(bool)
        )
        persistent_flags = (
            subset.loc[
                persistent_valid,
                "transfer_time_ms_persistent_anomaly",
            ]
            .fillna(False)
            .astype(bool)
        )

        raw_count = len(raw_flags)
        persistent_count = len(persistent_flags)
        raw_positives = int(raw_flags.sum())
        persistent_positives = int(persistent_flags.sum())

        record = {
            "scenario": scenario,
            "link_id": link_id,
            "link_role": (
                "fault_target" if link_id == fault_target
                else "negative_control" if fault_target is not None
                else "healthy_baseline"
            ),
            "transfer_observations": len(subset),
            "raw_scored_observations": raw_count,
            "raw_anomalies": raw_positives,
            "raw_anomaly_rate": (
                raw_positives / raw_count if raw_count else None
            ),
            "persistent_scored_observations": persistent_count,
            "persistent_anomalies": persistent_positives,
            "persistent_anomaly_rate": (
                persistent_positives / persistent_count
                if persistent_count else None
            ),
        }

        if fault_target is not None:
            if link_id == fault_target:
                record["TP"] = persistent_positives
                record["FN"] = persistent_count - persistent_positives
                record["FP"] = None
                record["TN"] = None
            else:
                record["TP"] = None
                record["FN"] = None
                record["FP"] = persistent_positives
                record["TN"] = persistent_count - persistent_positives
        else:
            record.update({
                "TP": None,
                "FN": None,
                "FP": None,
                "TN": None,
            })

        records.append(record)

    return records


# ---------------------------------------------------------
# Execute matched healthy and L1-fault simulations
# ---------------------------------------------------------

all_link_metrics = []
localization_records = []
scenario_decisions = []

for scenario in ("healthy", "l1_fault", "l2_fault"):
    print(f"\nRunning scenario: {scenario}")

    injector = create_fault_injector(scenario)

    engine = PDSimulationEngine(
        build_system(config),
        kv_cache_mb_per_prompt_token=KV_CACHE_MB_PER_PROMPT_TOKEN,
        fault_injector=injector,
        worker_pair_assignments=worker_pair_assignments,
    )

    trace = engine.run(
        copy.deepcopy(requests),
        reset_system=True,
    )

    if trace.empty or len(trace) != NUM_REQUESTS:
        raise RuntimeError(f"Invalid trace for {scenario}")

    pair_counts = (
        trace.groupby(
            ["prefill_worker_id", "decode_worker_id"]
        ).size().to_dict()
    )

    expected_pairs = {
        ("P1", "D1"): 250,
        ("P1", "D2"): 250,
        ("P2", "D1"): 250,
        ("P2", "D2"): 250,
    }

    if pair_counts != expected_pairs:
        raise RuntimeError(
            f"Worker-pair distribution mismatch: {pair_counts}"
        )

    link_counts = trace["link_id"].value_counts().to_dict()

    if link_counts.get("L1", 0) != 500 or link_counts.get("L2", 0) != 500:
        raise RuntimeError(
            f"Link distribution mismatch: {link_counts}"
        )

    print(f"Worker-pair counts: {pair_counts}")
    print(f"Link counts: {link_counts}")

    trace_name = f"{scenario}_seed{WORKLOAD_SEED}"

    trace.to_csv(
        OUTPUT_DIR / "traces" / f"{trace_name}_raw.csv",
        index=False,
    )

    ground_truth = engine.collector.get_ground_truth_dataframe()
    ground_truth.to_csv(
        OUTPUT_DIR / "ground_truth" / f"{trace_name}_ground_truth.csv",
        index=False,
    )

    analyzed, candidates, localized, decision = analyze_trace(trace)

    analyzed.to_csv(
        OUTPUT_DIR / "traces" / f"{trace_name}_analyzed.csv",
        index=False,
    )
    candidates.to_csv(
        OUTPUT_DIR / "candidates" / f"{trace_name}_candidates.csv",
        index=False,
    )
    localized.to_csv(
        OUTPUT_DIR / "localization" / f"{trace_name}_ranking.csv",
        index=False,
    )
    pd.DataFrame([decision]).to_csv(
        OUTPUT_DIR / f"{trace_name}_decision.csv",
        index=False,
    )

    scenario_decisions.append({
        "scenario": scenario,
        **decision,
    })

    all_link_metrics.extend(
        evaluate_links(analyzed, scenario)
    )

    communication_ranking = localized.loc[
        localized["fault_type"].eq("communication")
    ].copy()

    communication_ranking.to_csv(
        OUTPUT_DIR / "localization" /
        f"{trace_name}_communication_ranking.csv",
        index=False,
    )

    l1_rows = communication_ranking.loc[
        communication_ranking["component_id"].eq("L1")
    ]
    l2_rows = communication_ranking.loc[
        communication_ranking["component_id"].eq("L2")
    ]

    localization_records.append({
        "scenario": scenario,
        "l1_rank": (
            int(l1_rows.iloc[0]["localization_rank"])
            if not l1_rows.empty else None
        ),
        "l2_rank": (
            int(l2_rows.iloc[0]["localization_rank"])
            if not l2_rows.empty else None
        ),
        "top_ranked_component": (
            communication_ranking.iloc[0]["component_id"]
            if not communication_ranking.empty else None
        ),
        "communication_candidates": len(communication_ranking),
    })


# ---------------------------------------------------------
# Save metrics and metadata
# ---------------------------------------------------------

link_metrics = pd.DataFrame(all_link_metrics)
link_metrics.to_csv(
    OUTPUT_DIR / "link_detection_metrics.csv",
    index=False,
)

localization_metrics = pd.DataFrame(localization_records)
localization_metrics.to_csv(
    OUTPUT_DIR / "link_localization_metrics.csv",
    index=False,
)

pd.DataFrame(scenario_decisions).to_csv(
    OUTPUT_DIR / "scenario_decisions.csv",
    index=False,
)

metadata = {
    "experiment": "balanced_two_link_communication_localization",
    "simulation_type": "deterministic_two_pass_simulator",
    "purpose": (
        "Controlled diagnostic experiment with balanced worker-pair "
        "assignments and an L2 negative-control link."
    ),
    "training_seed": 42,
    "calibration_seed": 43,
    "boundary_training_seed": 46,
    "boundary_calibration_seed": 47,
    "workload_seed": WORKLOAD_SEED,
    "request_count": NUM_REQUESTS,
    "prompt_tokens": 4096,
    "output_tokens": 128,
    "arrival_rate_per_ms": 0.02,
    "kv_cache_mb_per_prompt_token": KV_CACHE_MB_PER_PROMPT_TOKEN,
    "worker_pair_assignments": {
        "P1->D1": 250,
        "P1->D2": 250,
        "P2->D1": 250,
        "P2->D2": 250,
    },
    "expected_link_counts": {
        "L1": 500,
        "L2": 500,
    },
    "faults": [
        {
            "type": "communication",
            "component": "L1",
            "severity": FAULT_SEVERITY,
            "interval_ms": [FAULT_START_MS, FAULT_END_MS],
        },
        {
            "type": "communication",
            "component": "L2",
            "severity": FAULT_SEVERITY,
            "interval_ms": [FAULT_START_MS, FAULT_END_MS],
        },
    ],
    "reference": {
        "workload_conditioned": True,
        "min_samples": 20,
    },
    "anomaly_detector": {
        "quantile": 0.99,
        "min_samples": 100,
    },
    "persistence": {
        "strategy": "strict_time_window_density",
        "window_duration_ms": 500,
        "min_anomalies": 3,
        "require_current_anomaly": True,
        "component_specific": True,
    },
    "confidence_reference_sample_target": 100,
    "confidence_calibration_sample_target": 500,
    "detector_inputs": "telemetry_only",
    "ground_truth_used_for_detection": False,
    "routing_interpretation": (
        "Externally assigned worker pairs for controlled route coverage; "
        "not natural shortest-queue scheduler behavior."
    ),
}

with (OUTPUT_DIR / "experiment_metadata.json").open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, indent=2)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BALANCED TWO-LINK EXPERIMENT SUMMARY")
print("=" * 70)

print("\nLink-level detection:")
print(link_metrics.to_string(index=False))

print("\nCommunication localization:")
print(localization_metrics.to_string(index=False))

print("\nScenario decisions:")
print(pd.DataFrame(scenario_decisions).to_string(index=False))

print(f"\nResults saved under: {OUTPUT_DIR.resolve()}")
print("Ground truth was kept separate from detector inputs.")
print("Experiment completed.")
