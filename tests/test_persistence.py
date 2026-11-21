import pandas as pd

from pd_failslow.detection.persistence import PersistenceDetector


def make_trace(flags, links=None, timestamps=None):
    n = len(flags)

    if links is None:
        links = ["L1"] * n

    if timestamps is None:
        timestamps = list(range(n))

    return pd.DataFrame({
        "transfer_time_ms_anomaly": pd.array(
            flags, dtype="boolean"
        ),
        "transfer_time_ms_anomaly_status": ["scored"] * n,
        "link_id": links,
        "transfer_completion_time_ms": timestamps,
    })


def test_three_of_five_anomalies_trigger_persistence():
    trace = make_trace([True, True, False, True, False])

    detector = PersistenceDetector(
        window_size=5,
        min_anomalies=3,
    )

    result = detector.predict(trace)

    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[-1] == True

    assert result[
        "transfer_time_ms_persistence_count"
    ].iloc[-1] == 3


def test_different_components_are_evaluated_independently():
    trace = make_trace(
        flags=[True, True, True, False, False, False],
        links=["L1", "L1", "L1", "L2", "L2", "L2"],
        timestamps=[1, 2, 3, 1, 2, 3],
    )

    result = PersistenceDetector(
        window_size=3,
        min_anomalies=3,
    ).predict(trace)

    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[2] == True

    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[5] == False


def test_observations_are_sorted_by_timestamp():
    trace = make_trace(
        flags=[True, False, True, True, False],
        timestamps=[3, 1, 4, 2, 5],
    )

    result = PersistenceDetector(
        window_size=5,
        min_anomalies=3,
    ).predict(trace)

    # Chronological sequence: False, True, True, True, False.
    # The final chronological observation is normal.
    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[4] == True

    assert result[
        "transfer_time_ms_persistence_count"
    ].iloc[4] == 3


def test_insufficient_history_does_not_trigger_alert():
    trace = make_trace([True, True, True, True])

    result = PersistenceDetector(
        window_size=5,
        min_anomalies=3,
    ).predict(trace)

    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].isna().all()

    assert result[
        "transfer_time_ms_persistence_status"
    ].eq("insufficient_history").all()


def test_invalid_anomaly_observation_is_excluded():
    trace = make_trace(
        flags=[True, True, True, True, False],
    )

    trace.loc[2, "transfer_time_ms_anomaly"] = pd.NA
    trace.loc[2, "transfer_time_ms_anomaly_status"] = (
        "not_comparable"
    )

    result = PersistenceDetector(
        window_size=5,
        min_anomalies=3,
    ).predict(trace)

    # Only four comparable observations exist.
    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].isna().all()


def test_invalid_window_configuration_is_rejected():
    try:
        PersistenceDetector(window_size=3, min_anomalies=4)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_time_aware_persistence_within_time_limit():
    from pd_failslow.detection.persistence import (
        TimeAwarePersistenceDetector,
    )

    trace = make_trace(
        [True, True, False, True, False],
        timestamps=[0, 40, 80, 120, 160],
    )

    result = TimeAwarePersistenceDetector(
        window_size=5,
        min_anomalies=3,
        max_window_duration_ms=250,
    ).predict(trace)

    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[-1] == True

    assert result[
        "transfer_time_ms_persistence_window_duration_ms"
    ].iloc[-1] == 160


def test_time_aware_persistence_rejects_slowly_spread_anomalies():
    from pd_failslow.detection.persistence import (
        TimeAwarePersistenceDetector,
    )

    trace = make_trace(
        [True, True, False, True, False],
        timestamps=[0, 100, 200, 300, 400],
    )

    result = TimeAwarePersistenceDetector(
        window_size=5,
        min_anomalies=3,
        max_window_duration_ms=250,
    ).predict(trace)

    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[-1] == False

    assert result[
        "transfer_time_ms_persistence_window_duration_ms"
    ].iloc[-1] == 400


def test_time_aware_persistence_accepts_exact_time_boundary():
    from pd_failslow.detection.persistence import (
        TimeAwarePersistenceDetector,
    )

    trace = make_trace(
        [True, False, True, False, True],
        timestamps=[0, 50, 100, 200, 250],
    )

    result = TimeAwarePersistenceDetector(
        window_size=5,
        min_anomalies=3,
        max_window_duration_ms=250,
    ).predict(trace)

    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[-1] == True


def test_time_aware_detector_rejects_invalid_duration():
    from pd_failslow.detection.persistence import (
        TimeAwarePersistenceDetector,
    )

    try:
        TimeAwarePersistenceDetector(
            max_window_duration_ms=-1
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_time_window_density_detects_slowly_spread_anomalies():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    trace = make_trace(
        [True, True, True],
        timestamps=[0, 163.84, 327.68],
    )

    result = TimeWindowDensityDetector(
        window_duration_ms=500,
        min_anomalies=3,
    ).predict(trace)

    assert result["transfer_time_ms_persistent_anomaly"].iloc[-1]
    assert result["transfer_time_ms_persistence_count"].iloc[-1] == 3
    assert result[
        "transfer_time_ms_persistence_window_duration_ms"
    ].iloc[-1] == 327.68


def test_time_window_density_excludes_exact_lower_boundary():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    trace = make_trace(
        [True, True, True],
        timestamps=[0, 163.84, 500],
    )

    result = TimeWindowDensityDetector(
        window_duration_ms=500,
        min_anomalies=3,
    ).predict(trace)

    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[-1] == False
    assert result[
        "transfer_time_ms_persistence_count"
    ].iloc[-1] == 2


def test_time_window_density_evaluates_components_independently():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    trace = make_trace(
        flags=[True, True, True, False, False, False],
        links=["L1", "L1", "L1", "L2", "L2", "L2"],
        timestamps=[0, 100, 200, 0, 100, 200],
    )

    result = TimeWindowDensityDetector(
        window_duration_ms=500,
        min_anomalies=3,
    ).predict(trace)

    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[2] == True
    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[5] == False


def test_time_window_density_handles_out_of_order_timestamps():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    trace = make_trace(
        [True, True, True],
        timestamps=[327.68, 0, 163.84],
    )

    result = TimeWindowDensityDetector(
        window_duration_ms=500,
        min_anomalies=3,
    ).predict(trace)

    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[0] == True
    assert result[
        "transfer_time_ms_persistence_count"
    ].iloc[0] == 3


def test_time_window_density_excludes_invalid_observations():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    trace = make_trace(
        [True, True, True, True],
        timestamps=[0, 100, 200, 300],
    )
    trace.loc[2, "transfer_time_ms_anomaly_status"] = "not_comparable"

    result = TimeWindowDensityDetector(
        window_duration_ms=500,
        min_anomalies=3,
    ).predict(trace)

    assert pd.isna(
        result["transfer_time_ms_persistent_anomaly"].iloc[2]
    )
    assert result[
        "transfer_time_ms_persistent_anomaly"
    ].iloc[3] == True
    assert result[
        "transfer_time_ms_persistence_count"
    ].iloc[3] == 3


def test_time_window_density_rejects_invalid_configuration():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    for kwargs in (
        {"window_duration_ms": 0},
        {"window_duration_ms": -1},
        {"min_anomalies": 0},
    ):
        try:
            TimeWindowDensityDetector(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")


def test_time_window_density_rejects_non_integer_thresholds():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    for value in (3.5, True, 0, -1):
        try:
            TimeWindowDensityDetector(min_anomalies=value)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected ValueError for min_anomalies={value}"
            )


def test_time_window_density_rejects_invalid_durations():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    for value in (0, -1, float("nan"), float("inf"), True, None):
        try:
            TimeWindowDensityDetector(window_duration_ms=value)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected ValueError for window_duration_ms={value}"
            )




def test_time_window_density_strict_positive_case():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    trace = make_trace(
        [True, True, True],
        timestamps=[0, 100, 200],
    )

    result = TimeWindowDensityDetector(
        window_duration_ms=500,
        min_anomalies=3,
        require_current_anomaly=True,
    ).predict(trace)

    assert result["transfer_time_ms_persistence_count"].iloc[-1] == 3
    assert result["transfer_time_ms_persistent_anomaly"].iloc[-1] == True


def test_time_window_density_strict_negative_case():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    trace = make_trace(
        [True, True, False],
        timestamps=[0, 100, 200],
    )

    result = TimeWindowDensityDetector(
        window_duration_ms=500,
        min_anomalies=3,
        require_current_anomaly=True,
    ).predict(trace)

    assert result["transfer_time_ms_persistence_count"].iloc[-1] == 2
    assert result["transfer_time_ms_persistent_anomaly"].iloc[-1] == False


def test_time_window_density_default_preserves_previous_behavior():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    trace = make_trace(
        [True, True, False],
        timestamps=[0, 100, 200],
    )

    result = TimeWindowDensityDetector(
        window_duration_ms=500,
        min_anomalies=2,
    ).predict(trace)

    assert result["transfer_time_ms_persistence_count"].iloc[-1] == 2
    assert result["transfer_time_ms_persistent_anomaly"].iloc[-1] == True


def test_time_window_density_rejects_non_boolean_strict_option():
    from pd_failslow.detection.persistence import TimeWindowDensityDetector

    for value in (1, 0, "True", None):
        try:
            TimeWindowDensityDetector(
                require_current_anomaly=value
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected ValueError for require_current_anomaly={value!r}"
            )

