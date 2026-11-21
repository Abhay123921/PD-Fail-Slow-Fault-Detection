import pandas as pd

from pd_failslow.detection.comparability import get_comparable_mask


def test_accepts_valid_reference_statuses():
    trace = pd.DataFrame({
        "transfer_time_ms": [10.0, 12.0, 15.0],
        "transfer_time_ms_reference": [9.0, 9.0, 9.0],
        "transfer_time_ms_status": [
            "ok", "fallback_global", "out_of_distribution"
        ],
    })

    result = get_comparable_mask(trace, "transfer_time_ms")

    assert result.tolist() == [True, True, False]


def test_global_fallback_can_be_disabled():
    trace = pd.DataFrame({
        "transfer_time_ms": [10.0, 12.0],
        "transfer_time_ms_reference": [9.0, 9.0],
        "transfer_time_ms_status": ["ok", "fallback_global"],
    })

    result = get_comparable_mask(
        trace,
        "transfer_time_ms",
        allow_global_fallback=False,
    )

    assert result.tolist() == [True, False]


def test_rejects_missing_and_non_finite_values():
    trace = pd.DataFrame({
        "transfer_time_ms": [10.0, float("inf"), "invalid"],
        "transfer_time_ms_reference": [9.0, 9.0, 9.0],
        "transfer_time_ms_status": ["ok", "ok", "ok"],
    })

    result = get_comparable_mask(trace, "transfer_time_ms")

    assert result.tolist() == [True, False, False]


def test_missing_columns_return_false():
    trace = pd.DataFrame({"request_id": ["r1", "r2"]})

    result = get_comparable_mask(trace, "transfer_time_ms")

    assert result.tolist() == [False, False]


def test_invalid_metric_name_raises():
    trace = pd.DataFrame()

    try:
        get_comparable_mask(trace, "")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
