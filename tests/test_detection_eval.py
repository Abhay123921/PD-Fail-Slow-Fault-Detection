import pandas as pd
import pytest

from pd_failslow.evaluation.detection_eval import (
    build_request_level_labels,
    evaluate_request_detection,
)


def make_trace():
    return pd.DataFrame({
        "request_id": [1, 2],
        "prefill_worker_id": ["P1", "P2"],
        "prefill_start_ms": [5.0, 5.0],
        "prefill_completion_ms": [15.0, 15.0],
        "prefill_service_time_ms_persistent_anomaly": [True, False],
        "decode_worker_id": ["D2", "D1"],
        "decode_execution_start_ms": [20.0, 20.0],
        "decode_completion_ms": [30.0, 30.0],
        "decode_service_time_ms_persistent_anomaly": [False, False],
        "link_id": ["L1", "L2"],
        "transfer_start_time_ms": [5.0, 5.0],
        "transfer_completion_time_ms": [15.0, 15.0],
        "transfer_time_ms_persistent_anomaly": [True, False],
        "handoff_wait_ms_persistent_anomaly": [True, False],
    })


def make_ground_truth(fault_type, component_id, start=10.0, end=20.0):
    return pd.DataFrame([{
        "fault_type": fault_type,
        "component_id": component_id,
        "start_time_ms": start,
        "end_time_ms": end,
    }])


def test_compute_exposure_and_healthy_worker_false_positive():
    trace = make_trace()
    trace["prefill_service_time_ms_persistent_anomaly"] = [True, True]

    result = evaluate_request_detection(
        trace,
        make_ground_truth("compute", "P1"),
        "compute",
    )

    assert result["details"]["y_true"].tolist() == [True, False]
    assert result["details"]["y_pred"].tolist() == [True, True]
    assert result["metrics"]["true_positive"] == 1
    assert result["metrics"]["false_positive"] == 1


def test_communication_interval_overlap():
    trace = make_trace()
    gt = make_ground_truth("communication", "L1", start=9.0)

    result = evaluate_request_detection(trace, gt, "communication")

    assert result["details"]["y_true"].tolist() == [True, False]
    assert result["details"]["y_pred"].tolist() == [True, False]
    assert result["metrics"]["true_positive"] == 1
    assert result["metrics"]["true_negative"] == 1


def test_communication_touching_boundary_is_not_exposed():
    trace = make_trace()
    trace.loc[0, "transfer_completion_time_ms"] = 10.0

    gt = make_ground_truth("communication", "L1", start=10.0)

    result = build_request_level_labels(trace, gt, "communication")

    assert result["y_true"].tolist() == [False, False]


def test_synchronization_uses_transfer_completion_timestamp():
    trace = make_trace()
    trace["transfer_completion_time_ms"] = [10.0, 9.0]

    result = evaluate_request_detection(
        trace,
        make_ground_truth("synchronization", "D2"),
        "synchronization",
    )

    assert result["details"]["y_true"].tolist() == [True, False]
    assert result["details"]["y_pred"].tolist() == [True, False]
    assert result["metrics"]["true_positive"] == 1
    assert result["metrics"]["true_negative"] == 1


def test_healthy_run_counts_anomalies_as_false_positives():
    trace = make_trace()
    trace["prefill_service_time_ms_persistent_anomaly"] = [True, True]

    empty_gt = pd.DataFrame(columns=[
        "fault_type", "component_id", "start_time_ms", "end_time_ms"
    ])

    result = evaluate_request_detection(trace, empty_gt, "compute")

    assert result["details"]["y_true"].tolist() == [False, False]
    assert result["details"]["y_pred"].tolist() == [True, True]
    assert result["metrics"]["false_positive"] == 2
    assert result["metrics"]["true_negative"] == 0


def test_unsupported_fault_type_raises_error():
    trace = make_trace()
    gt = make_ground_truth("compute", "P1")

    try:
        build_request_level_labels(trace, gt, "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unsupported fault type")

def test_raw_mode_detects_anomaly_filtered_by_persistence():
    trace = make_trace()

    trace["prefill_service_time_ms_anomaly"] = [True, False]
    trace["decode_service_time_ms_anomaly"] = [False, False]

    trace["prefill_service_time_ms_persistent_anomaly"] = [False, False]

    gt = make_ground_truth("compute", "P1")

    raw = evaluate_request_detection(
        trace, gt, "compute", prediction_mode="raw"
    )
    persistent = evaluate_request_detection(
        trace, gt, "compute", prediction_mode="persistent"
    )

    assert raw["details"]["y_true"].tolist() == [True, False]
    assert persistent["details"]["y_true"].tolist() == [True, False]

    assert raw["details"]["y_pred"].tolist() == [True, False]
    assert persistent["details"]["y_pred"].tolist() == [False, False]


def test_raw_and_persistent_modes_have_identical_ground_truth():
    trace = make_trace()
    trace["prefill_service_time_ms_anomaly"] = [True, True]
    trace["decode_service_time_ms_anomaly"] = [False, False]

    gt = make_ground_truth("compute", "P1")

    raw = build_request_level_labels(
        trace, gt, "compute", prediction_mode="raw"
    )
    persistent = build_request_level_labels(
        trace, gt, "compute", prediction_mode="persistent"
    )

    assert raw["y_true"].equals(persistent["y_true"])


def test_invalid_prediction_mode_raises_error():
    trace = make_trace()
    gt = make_ground_truth("compute", "P1")

    with pytest.raises(ValueError, match="prediction_mode"):
        build_request_level_labels(
            trace, gt, "compute", prediction_mode="unknown"
        )

