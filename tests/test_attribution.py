import pandas as pd

from pd_failslow.diagnosis.attribution import (
    build_attribution_candidates,
)


def test_direct_stage_attribution():
    trace = pd.DataFrame({
        "request_id": ["r1", "r2", "r3", "r4"],
        "prefill_worker_id": ["P1", "P1", "P2", "P2"],
        "decode_worker_id": ["D1", "D1", "D2", "D2"],
        "link_id": ["L1", "L1", "L2", "L2"],

        "prefill_service_time_ms_persistent_anomaly": [True, False, False, False],
        "decode_service_time_ms_persistent_anomaly": [False, True, False, False],
        "transfer_time_ms_persistent_anomaly": [False, False, True, False],
        "handoff_wait_ms_persistent_anomaly": [False, False, False, True],

        "prefill_service_time_ms_persistence_status": ["evaluated"] * 4,
        "decode_service_time_ms_persistence_status": ["evaluated"] * 4,
        "transfer_time_ms_persistence_status": ["evaluated"] * 4,
        "handoff_wait_ms_persistence_status": ["evaluated"] * 4,

        "prefill_completion_ms": [10, 10, 10, 10],
        "decode_completion_ms": [20, 20, 20, 20],
        "transfer_completion_time_ms": [15, 15, 15, 15],
        "decode_admission_time_ms": [16, 16, 16, 16],
    })

    result = build_attribution_candidates(trace)

    direct = result[result["evidence_role"] == "direct"]

    assert len(direct) == 4

    assert set(zip(direct["fault_type"], direct["component_id"])) == {
        ("compute", "P1"),
        ("compute", "D1"),
        ("communication", "L2"),
        ("synchronization", "D2"),
    }


def test_queue_and_admission_are_context_only():
    trace = pd.DataFrame({
        "request_id": ["r1"],
        "prefill_queue_wait_ms_persistent_anomaly": [True],
        "admission_wait_ms_persistent_anomaly": [True],
        "prefill_queue_wait_ms_persistence_status": ["evaluated"],
        "admission_wait_ms_persistence_status": ["evaluated"],
        "prefill_start_ms": [10.0],
        "decode_execution_start_ms": [20.0],
    })

    result = build_attribution_candidates(trace)

    assert len(result) == 2
    assert result["evidence_role"].eq("context_only").all()
    assert result["component_id"].isna().all()
    assert result["fault_type"].isna().all()


def test_nonpersistent_anomalies_are_excluded():
    trace = pd.DataFrame({
        "request_id": ["r1"],
        "transfer_time_ms_persistent_anomaly": [False],
        "transfer_time_ms_persistence_status": ["evaluated"],
        "link_id": ["L1"],
    })

    result = build_attribution_candidates(trace)

    assert result.empty


def test_unavailable_persistence_evidence_is_excluded():
    trace = pd.DataFrame({
        "request_id": ["r1"],
        "transfer_time_ms_persistent_anomaly": [True],
        "transfer_time_ms_persistence_status": ["insufficient_history"],
        "link_id": ["L1"],
    })

    result = build_attribution_candidates(trace)

    assert result.empty


def test_missing_component_is_not_attributed():
    trace = pd.DataFrame({
        "request_id": ["r1"],
        "transfer_time_ms_persistent_anomaly": [True],
        "transfer_time_ms_persistence_status": ["evaluated"],
        "link_id": [None],
    })

    result = build_attribution_candidates(trace)

    assert result.empty


def test_ground_truth_is_not_required():
    trace = pd.DataFrame({
        "request_id": ["r1"],
        "transfer_time_ms_persistent_anomaly": [True],
        "transfer_time_ms_persistence_status": ["evaluated"],
        "link_id": ["L1"],
        "transfer_completion_time_ms": [10.0],
    })

    result = build_attribution_candidates(trace)

    assert len(result) == 1
    assert result.iloc[0]["component_id"] == "L1"
    assert "fault_id" not in result.columns
    assert "severity" not in result.columns
