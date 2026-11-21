from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd

from pd_failslow.utils.config import load_experiment_config, build_system
from pd_failslow.workload.generator import (
    generate_requests,
    generate_workload_profile,
)
from pd_failslow.simulator.engine import PDSimulationEngine

from pd_failslow.faults.injector import FaultInjector
from pd_failslow.faults.compute_faults import ComputeFault
from pd_failslow.faults.communication_faults import CommunicationFault
from pd_failslow.faults.synchronization_faults import SynchronizationFault

from pd_failslow.detection.reference import HealthyReferenceModel
from pd_failslow.detection.preprocessing import (
    STAGE_METRICS,
    prepare_stage_residuals,
)
from pd_failslow.detection.anomaly import StageAnomalyDetector
from pd_failslow.detection.persistence import PersistenceDetector
from pd_failslow.detection.confidence import EvidenceConfidence

from pd_failslow.diagnosis.attribution import build_attribution_candidates
from pd_failslow.diagnosis.localization import rank_localized_components
from pd_failslow.diagnosis.decision import decide_diagnosis


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

ROOT = Path("results")
TRACE_DIR = ROOT / "healthy_characterization" / "traces"
OUTPUT_DIR = ROOT / "integrated_fault_experiments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "traces").mkdir(exist_ok=True)
(OUTPUT_DIR / "ground_truth").mkdir(exist_ok=True)
(OUTPUT_DIR / "candidates").mkdir(exist_ok=True)
(OUTPUT_DIR / "localization").mkdir(exist_ok=True)

PROFILES = [
    "stable",
    "long_prompt",
    "long_output",
    "high_arrival",
    "mixed",
]

TRAIN_SEED = 42
CALIBRATION_SEED = 43
HEALTHY_TEST_SEED = 44
FAULT_SEED = 45

NUM_REQUESTS = 1000

FAULT_START_MS = 0.0
FAULT_END_MS = 1_000_000.0

FAULT_SEVERITY = 0.50

config = load_experiment_config("configs")


# ---------------------------------------------------------
# Load healthy reference training and calibration data
# ---------------------------------------------------------

def load_healthy_traces(seed):
    frames = []

    for profile in PROFILES:
        path = TRACE_DIR / f"{profile}_seed{seed}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        frame["evaluation_profile"] = profile
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


print("\n" + "=" * 65)
print("PHASE 1: HEALTHY REFERENCE TRAINING")
print("=" * 65)

def generate_boundary_healthy_trace(seed):
    """Generate healthy traces matching the integrated fault workload."""
    boundary_requests = generate_requests(
        num_requests=NUM_REQUESTS,
        seed=seed,
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

    engine = PDSimulationEngine(
        build_system(config),
        kv_cache_mb_per_prompt_token=0.01,
    )
    trace = engine.run(boundary_requests, reset_system=True)

    output_path = TRACE_DIR / f"integrated_4096_seed{seed}.csv"
    trace.to_csv(output_path, index=False)
    print(f"Saved boundary healthy trace: {output_path}")
    return trace


train_boundary = generate_boundary_healthy_trace(seed=46)
calibration_boundary = generate_boundary_healthy_trace(seed=47)

train_trace = pd.concat(
    [load_healthy_traces(TRAIN_SEED), train_boundary],
    ignore_index=True,
)
calibration_trace = pd.concat(
    [load_healthy_traces(CALIBRATION_SEED), calibration_boundary],
    ignore_index=True,
)

print(f"Reference training rows: {len(train_trace)}")
print(f"Anomaly calibration rows: {len(calibration_trace)}")

reference_model = HealthyReferenceModel(min_samples=20)
reference_model.fit(train_trace)

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

threshold_records = []

for metric in STAGE_METRICS:
    threshold_records.append({
        "stage": metric,
        "threshold_ms": anomaly_detector.thresholds_.get(metric),
        "calibration_samples": anomaly_detector.sample_counts_.get(
            metric, 0
        ),
    })

threshold_df = pd.DataFrame(threshold_records)
threshold_df.to_csv(
    OUTPUT_DIR / "healthy_anomaly_thresholds.csv",
    index=False,
)

print("\nHealthy anomaly thresholds:")
print(threshold_df.to_string(index=False))


# ---------------------------------------------------------
# Complete detector and diagnosis pipeline
# ---------------------------------------------------------

def analyze_trace(trace):
    """
    The detector receives telemetry only.
    Ground truth is deliberately excluded.
    """

    referenced = reference_model.predict(trace)
    residuals = prepare_stage_residuals(referenced)

    anomalies = anomaly_detector.predict(residuals)

    persistent = PersistenceDetector(
        window_size=5,
        min_anomalies=3,
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


def stage_summary(analyzed):
    records = []

    for metric in STAGE_METRICS:
        anomaly_col = f"{metric}_anomaly"
        persistent_col = f"{metric}_persistent_anomaly"
        status_col = f"{metric}_anomaly_status"

        anomaly_count = 0
        persistent_count = 0

        if anomaly_col in analyzed.columns:
            anomaly_count = int(
                analyzed[anomaly_col].eq(True).sum()
            )

        if persistent_col in analyzed.columns:
            persistent_count = int(
                analyzed[persistent_col].eq(True).sum()
            )

        scored_count = 0

        if status_col in analyzed.columns:
            scored_count = int(
                analyzed[status_col].eq("scored").sum()
            )

        records.append({
            "stage": metric,
            "scored_requests": scored_count,
            "anomalies": anomaly_count,
            "persistent_anomalies": persistent_count,
        })

    return records


def save_analysis(name, trace):
    analyzed, candidates, localized, decision = analyze_trace(trace)

    analyzed.to_csv(
        OUTPUT_DIR / "traces" / f"{name}_analyzed.csv",
        index=False,
    )

    candidates.to_csv(
        OUTPUT_DIR / "candidates" / f"{name}_candidates.csv",
        index=False,
    )

    localized.to_csv(
        OUTPUT_DIR / "localization" / f"{name}_ranking.csv",
        index=False,
    )

    pd.DataFrame([decision]).to_csv(
        OUTPUT_DIR / f"{name}_decision.csv",
        index=False,
    )

    pd.DataFrame(stage_summary(analyzed)).to_csv(
        OUTPUT_DIR / f"{name}_stage_summary.csv",
        index=False,
    )

    return analyzed, candidates, localized, decision


# ---------------------------------------------------------
# Phase 2: Held-out healthy evaluation
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("PHASE 2: HELD-OUT HEALTHY EVALUATION (SEED 44)")
print("=" * 65)

healthy_decisions = []
healthy_stage_records = []

for profile in PROFILES:
    path = TRACE_DIR / f"{profile}_seed{HEALTHY_TEST_SEED}.csv"

    if path.exists():
        trace = pd.read_csv(path)
    else:
        print(f"Generating missing healthy trace: {profile}, seed {HEALTHY_TEST_SEED}")

        healthy_requests = generate_workload_profile(
            profile,
            num_requests=NUM_REQUESTS,
            seed=HEALTHY_TEST_SEED,
        )

        healthy_engine = PDSimulationEngine(
            build_system(config),
            kv_cache_mb_per_prompt_token=0.01,
        )

        trace = healthy_engine.run(
            healthy_requests,
            reset_system=True,
        )

        trace.to_csv(path, index=False)
        print(f"Saved: {path}")

    analyzed, candidates, localized, decision = save_analysis(
        f"healthy_seed44_{profile}",
        trace,
    )

    decision_record = {
        "profile": profile,
        "request_count": len(trace),
        **decision,
    }

    healthy_decisions.append(decision_record)

    for record in stage_summary(analyzed):
        healthy_stage_records.append({
            "profile": profile,
            **record,
        })

    print(
        f"{profile:15s} | "
        f"status={decision['status']:20s} | "
        f"direct={decision['direct_anomalous_observation_count']} | "
        f"context={decision['context_anomalous_observation_count']}"
    )

pd.DataFrame(healthy_decisions).to_csv(
    OUTPUT_DIR / "healthy_holdout_decisions.csv",
    index=False,
)

pd.DataFrame(healthy_stage_records).to_csv(
    OUTPUT_DIR / "healthy_holdout_stage_metrics.csv",
    index=False,
)


# ---------------------------------------------------------
# Phase 3: Generate one matched workload
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("PHASE 3: MATCHED FAULT EXPERIMENT WORKLOAD")
print("=" * 65)

requests = generate_requests(
    num_requests=NUM_REQUESTS,
    seed=FAULT_SEED,
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

print(f"Requests: {len(requests)}")
print(f"Prompt tokens: {requests[0].prompt_tokens}")
print(f"Output tokens: {requests[0].output_tokens}")
print(f"Arrival seed: {FAULT_SEED}")
print("Same request workload will be reused in every simulation.")


# ---------------------------------------------------------
# Phase 4: Construct fault scenarios
# ---------------------------------------------------------

def create_injector(scenario):
    if scenario == "healthy":
        return None

    injector = FaultInjector()

    if scenario == "compute":
        injector.register_compute_fault(
            ComputeFault(
                fault_id="integrated_compute_P1",
                worker_id="P1",
                start_time_ms=FAULT_START_MS,
                end_time_ms=FAULT_END_MS,
                severity=FAULT_SEVERITY,
            )
        )

    elif scenario == "communication":
        injector.register_communication_fault(
            CommunicationFault(
                fault_id="integrated_communication_L1",
                link_id="L1",
                start_time_ms=FAULT_START_MS,
                end_time_ms=FAULT_END_MS,
                severity=FAULT_SEVERITY,
            )
        )

    elif scenario == "synchronization":
        injector.register_synchronization_fault(
            SynchronizationFault(
                fault_id="integrated_synchronization_D2",
                component_id="D2",
                start_time_ms=FAULT_START_MS,
                end_time_ms=FAULT_END_MS,
                severity=FAULT_SEVERITY,
                max_additional_delay_ms=10.0,
            )
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return injector


SCENARIOS = [
    "healthy",
    "compute",
    "communication",
    "synchronization",
]


# ---------------------------------------------------------
# Phase 5: Execute matched simulations
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("PHASE 4: MATCHED HEALTHY AND FAULTY SIMULATIONS")
print("=" * 65)

fault_decisions = []
fault_stage_records = []
ground_truth_records = []

for scenario in SCENARIOS:
    print(f"\nRunning scenario: {scenario}")

    system = build_system(config)
    injector = create_injector(scenario)

    engine = PDSimulationEngine(
        system,
        kv_cache_mb_per_prompt_token=0.01,
        fault_injector=injector,
    )

    # Deep copy prevents request-state contamination between runs.
    scenario_requests = copy.deepcopy(requests)

    trace = engine.run(
        scenario_requests,
        reset_system=True,
    )

    if trace.empty:
        raise RuntimeError(f"Empty trace for scenario: {scenario}")

    name = f"fault_seed45_{scenario}"

    trace.to_csv(
        OUTPUT_DIR / "traces" / f"{name}_raw.csv",
        index=False,
    )

    # Preserve ground truth separately from detector input.
    if injector is not None:
        gt = engine.collector.get_ground_truth_dataframe()
    else:
        gt = pd.DataFrame()

    gt.to_csv(
        OUTPUT_DIR / "ground_truth" / f"{name}_ground_truth.csv",
        index=False,
    )

    # Apply the complete pipeline to telemetry only.
    analyzed, candidates, localized, decision = save_analysis(
        name,
        trace,
    )

    if scenario == "compute":
        target_count = int(trace["prefill_worker_id"].eq("P1").sum())
        target_description = "P1 prefill assignments"

    elif scenario == "communication":
        target_count = int(trace["link_id"].eq("L1").sum())
        target_description = "L1 transfers"

    elif scenario == "synchronization":
        target_count = int(trace["decode_worker_id"].eq("D2").sum())
        target_description = "D2 decode assignments"

    else:
        target_count = len(trace)
        target_description = "all requests"

    record = {
        "scenario": scenario,
        "request_count": len(trace),
        "target_component_observations": target_count,
        "target_component_description": target_description,
        "status": decision["status"],
        "is_fault_suspected": decision["is_fault_suspected"],
        "diagnosed_fault_type": decision["fault_type"],
        "diagnosed_component_type": decision["component_type"],
        "diagnosed_component_id": decision["component_id"],
        "localization_rank": decision["localization_rank"],
        "evidence_count": decision["evidence_count"],
        "mean_evidence_quality": decision[
            "mean_evidence_quality_score"
        ],
        "direct_anomalies": decision[
            "direct_anomalous_observation_count"
        ],
        "context_anomalies": decision[
            "context_anomalous_observation_count"
        ],
        "persistent_context_evidence": decision[
            "persistent_context_evidence_count"
        ],
    }

    fault_decisions.append(record)

    for stage_record in stage_summary(analyzed):
        fault_stage_records.append({
            "scenario": scenario,
            **stage_record,
        })

    if injector is not None:
        for fault in injector.to_ground_truth_records():
            ground_truth_records.append({
                "scenario": scenario,
                **fault,
            })

    print(pd.Series(record).to_string())


# ---------------------------------------------------------
# Phase 6: Save final experiment results
# ---------------------------------------------------------

fault_summary = pd.DataFrame(fault_decisions)
fault_summary.to_csv(
    OUTPUT_DIR / "fault_experiment_decisions.csv",
    index=False,
)

pd.DataFrame(fault_stage_records).to_csv(
    OUTPUT_DIR / "fault_experiment_stage_metrics.csv",
    index=False,
)

pd.DataFrame(ground_truth_records).to_csv(
    OUTPUT_DIR / "fault_experiment_ground_truth_index.csv",
    index=False,
)

experiment_metadata = {
    "experiment": "integrated_workload_aware_failslow_detection",
    "reference_training_seed": TRAIN_SEED,
    "anomaly_calibration_seed": CALIBRATION_SEED,
    "healthy_holdout_seed": HEALTHY_TEST_SEED,
    "fault_workload_seed": FAULT_SEED,
    "request_count": NUM_REQUESTS,
    "prompt_tokens": 4096,
    "output_tokens": 128,
    "arrival_distribution": "constant",
    "arrival_rate_per_ms": 0.02,
    "kv_cache_mb_per_prompt_token": 0.01,
    "fault_severity": FAULT_SEVERITY,
    "fault_interval_ms": [
        FAULT_START_MS,
        FAULT_END_MS,
    ],
    "fault_scenarios": [
        {
            "fault_type": "compute",
            "component": "P1",
            "severity": FAULT_SEVERITY,
        },
        {
            "fault_type": "communication",
            "component": "L1",
            "severity": FAULT_SEVERITY,
        },
        {
            "fault_type": "synchronization",
            "component": "D2",
            "severity": FAULT_SEVERITY,
            "max_additional_delay_ms": 10.0,
        },
    ],
    "detector_inputs": "telemetry_only",
    "ground_truth_used_for_detection": False,
    "simulation_type": "deterministic_two_pass_simulator",
}

with (OUTPUT_DIR / "experiment_metadata.json").open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(experiment_metadata, file, indent=2)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("INTEGRATED FAULT EXPERIMENT: FINAL SUMMARY")
print("=" * 75)

print("\nFault scenario decisions:")
print(fault_summary.to_string(index=False))

print("\nHealthy holdout decisions:")
print(pd.DataFrame(healthy_decisions)[
    [
        "profile",
        "status",
        "is_fault_suspected",
        "direct_anomalous_observation_count",
        "context_anomalous_observation_count",
    ]
].to_string(index=False))

print("\nResults directory:")
print(OUTPUT_DIR.resolve())

print("\nGround truth was kept separate from detector inputs.")
print("Experiment completed.")
