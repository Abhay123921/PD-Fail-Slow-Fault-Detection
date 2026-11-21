import pandas as pd
import pytest

from pd_failslow.detection.preprocessing import STAGE_METRICS
from pd_failslow.diagnosis.decision import decide_diagnosis


def make_trace(
    anomalies=False,
    scored_stages=None,
    persistent_direct=False,
):
    if scored_stages is None:
        scored_stages = list(STAGE_METRICS)

    direct_stages = {
        "prefill_service_time_ms",
        "decode_service_time_ms",
        "transfer_time_ms",
        "handoff_wait_ms",
    }

    data = {}
    for metric in STAGE_METRICS:
        scored = metric in scored_stages
        data[f"{metric}_anomaly_status"] = [
            "scored" if scored else "not_comparable"
        ]
        data[f"{metric}_anomaly"] = [
            bool(anomalies) if scored else None
        ]
        data[f"{metric}_persistent_anomaly"] = [
            bool(persistent_direct and metric in direct_stages)
            if scored else None
        ]

    return pd.DataFrame(data)

def make_localized(
    fault_type="compute",
    component_id="P1",
    evidence_count=5,
    quality=0.8,
):
    return pd.DataFrame([{
        "fault_type": fault_type,
        "component_type": "worker",
        "component_id": component_id,
        "evidence_count": evidence_count,
        "localization_rank": 1,
        "mean_evidence_quality_score": quality,
    }])


def test_all_stages_scored_without_anomalies_is_healthy():
    trace = make_trace(anomalies=False)

    result = decide_diagnosis(
        trace,
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert result["status"] == "healthy"
    assert result["is_fault_suspected"] is False


def test_anomalies_without_localization_are_suspicious():
    trace = make_trace(anomalies=True, persistent_direct=True)

    result = decide_diagnosis(
        trace,
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert result["status"] == "suspicious"
    assert result["is_fault_suspected"] is True


def test_strong_direct_evidence_localizes_fault():
    trace = make_trace(anomalies=True, persistent_direct=True)
    localized = make_localized()

    result = decide_diagnosis(
        trace,
        pd.DataFrame(),
        localized,
    )

    assert result["status"] == "localized_fault"
    assert result["fault_type"] == "compute"
    assert result["component_id"] == "P1"
    assert result["evidence_count"] == 5


def test_low_evidence_count_does_not_localize():
    trace = make_trace(anomalies=True, persistent_direct=True)
    localized = make_localized(evidence_count=2)

    result = decide_diagnosis(
        trace,
        pd.DataFrame(),
        localized,
    )

    assert result["status"] == "suspicious"


def test_low_quality_does_not_localize():
    trace = make_trace(anomalies=True, persistent_direct=True)
    localized = make_localized(quality=0.3)

    result = decide_diagnosis(
        trace,
        pd.DataFrame(),
        localized,
    )

    assert result["status"] == "suspicious"


def test_competing_fault_types_remain_suspicious():
    trace = make_trace(anomalies=True, persistent_direct=True)

    localized = pd.DataFrame([
        {
            "fault_type": "compute",
            "component_type": "worker",
            "component_id": "P1",
            "evidence_count": 5,
            "localization_rank": 1,
            "mean_evidence_quality_score": 0.9,
        },
        {
            "fault_type": "communication",
            "component_type": "link",
            "component_id": "L1",
            "evidence_count": 5,
            "localization_rank": 1,
            "mean_evidence_quality_score": 0.9,
        },
    ])

    result = decide_diagnosis(
        trace,
        pd.DataFrame(),
        localized,
    )

    assert result["status"] == "suspicious"
    assert result["reason"] == "multiple_fault_types_compete"


def test_incomplete_coverage_without_anomalies_abstains():
    trace = make_trace(
        anomalies=False,
        scored_stages=list(STAGE_METRICS[:-1]),
    )

    result = decide_diagnosis(
        trace,
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert result["status"] == "insufficient_evidence"


def test_no_scored_observations_abstains():
    trace = make_trace(anomalies=False, scored_stages=[])

    result = decide_diagnosis(
        trace,
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert result["status"] == "insufficient_evidence"


def test_empty_trace_abstains():
    result = decide_diagnosis(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert result["status"] == "insufficient_evidence"


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        decide_diagnosis(
            make_trace(),
            pd.DataFrame(),
            pd.DataFrame(),
            min_evidence=0,
        )

    with pytest.raises(ValueError):
        decide_diagnosis(
            make_trace(),
            pd.DataFrame(),
            pd.DataFrame(),
            min_quality=1.5,
        )


def test_context_only_anomalies_are_workload_pressure():
    data = {}

    for metric in STAGE_METRICS:
        is_context = metric in {
            "prefill_queue_wait_ms",
            "admission_wait_ms",
        }
        data[f"{metric}_anomaly_status"] = ["scored"]
        data[f"{metric}_anomaly"] = [is_context]

    trace = pd.DataFrame(data)

    result = decide_diagnosis(
        trace,
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert result["status"] == "workload_pressure"
    assert result["is_fault_suspected"] is False
    assert result["direct_anomalous_observation_count"] == 0
    assert result["context_anomalous_observation_count"] == 2


def test_single_persistent_direct_anomaly_does_not_suspect_fault():
    trace = make_trace(anomalies=False)

    trace["transfer_time_ms_anomaly"] = [True]
    trace["transfer_time_ms_persistent_anomaly"] = [True]

    result = decide_diagnosis(
        trace,
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert result["status"] == "insufficient_evidence"
    assert result["is_fault_suspected"] is False
    assert result["persistent_direct_anomaly_count"] == 1
    assert result["reason"] == (
        "direct_anomalies_below_persistence_evidence_threshold"
    )
