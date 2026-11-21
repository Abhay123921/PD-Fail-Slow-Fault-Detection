import numpy as np
import pandas as pd
import pytest

from pd_failslow.detection.anomaly import StageAnomalyDetector


METRIC = "transfer_time_ms"


def make_trace(residuals, statuses=None):
    residuals = list(residuals)
    n = len(residuals)

    if statuses is None:
        statuses = ["ok"] * n

    reference = np.full(n, 10.0)

    return pd.DataFrame({
        METRIC: reference + residuals,
        f"{METRIC}_reference": reference,
        f"{METRIC}_status": statuses,
        f"{METRIC}_residual_ms": residuals,
        f"{METRIC}_positive_residual_ms": np.maximum(
            residuals, 0
        ),
    })


def test_calibrates_threshold_from_healthy_residuals():
    calibration = make_trace(np.linspace(0, 2, 100))

    detector = StageAnomalyDetector(
        quantile=0.90,
        min_samples=50,
    )
    detector.fit(calibration)

    expected = calibration[
        f"{METRIC}_positive_residual_ms"
    ].quantile(0.90)

    assert detector.thresholds_[METRIC] == pytest.approx(expected)
    assert detector.sample_counts_[METRIC] == 100


def test_detects_large_positive_residual():
    calibration = make_trace(np.zeros(100))

    detector = StageAnomalyDetector(
        quantile=0.99,
        min_samples=50,
    ).fit(calibration)

    test_data = make_trace([0.0, 0.0, 5.0])
    result = detector.predict(test_data)

    assert result[f"{METRIC}_anomaly"].tolist() == [
        False, False, True
    ]


def test_negative_residual_is_not_an_anomaly():
    calibration = make_trace(np.zeros(100))

    detector = StageAnomalyDetector(
        min_samples=50
    ).fit(calibration)

    result = detector.predict(make_trace([-5.0]))

    assert result[f"{METRIC}_anomaly"].iloc[0] == False


def test_ood_observation_is_not_scored():
    calibration = make_trace(np.zeros(100))

    detector = StageAnomalyDetector(
        min_samples=50
    ).fit(calibration)

    test_data = make_trace(
        [20.0],
        statuses=["out_of_distribution"],
    )
    result = detector.predict(test_data)

    assert pd.isna(result[f"{METRIC}_anomaly"].iloc[0])
    assert result[f"{METRIC}_anomaly_status"].iloc[0] == (
        "not_comparable"
    )


def test_insufficient_calibration_causes_abstention():
    calibration = make_trace([0.0, 1.0, 2.0])

    detector = StageAnomalyDetector(
        min_samples=10
    ).fit(calibration)

    result = detector.predict(make_trace([10.0]))

    assert pd.isna(result[f"{METRIC}_anomaly"].iloc[0])
    assert result[f"{METRIC}_anomaly_status"].iloc[0] == (
        "insufficient_calibration"
    )


def test_original_calibration_data_is_not_modified():
    calibration = make_trace(np.zeros(100))
    original = calibration.copy()

    StageAnomalyDetector(min_samples=50).fit(calibration)

    pd.testing.assert_frame_equal(calibration, original)


def test_rejects_invalid_quantile():
    with pytest.raises(ValueError):
        StageAnomalyDetector(quantile=1.0)


def make_link_trace(residuals, link_types, bandwidths):
    trace = make_trace(residuals)
    trace["link_type"] = link_types
    trace["configured_bandwidth_mb_per_ms"] = bandwidths
    return trace


def test_link_conditioned_calibration_learns_separate_thresholds():
    pcie = np.linspace(0, 1, 100)
    nvlink = np.linspace(0, 0.2, 100)

    calibration = make_link_trace(
        np.concatenate([pcie, nvlink]),
        ["PCIe"] * 100 + ["NVLink"] * 100,
        [20.0] * 100 + [80.0] * 100,
    )

    detector = StageAnomalyDetector(
        quantile=0.90,
        min_samples=50,
        link_conditioned_transfer=True,
    ).fit(calibration)

    assert detector.link_thresholds_[("PCIe", 20.0)] == pytest.approx(
        pcie.tolist() and pd.Series(pcie).quantile(0.90)
    )
    assert detector.link_thresholds_[("NVLink", 80.0)] == pytest.approx(
        pd.Series(nvlink).quantile(0.90)
    )
    assert detector.link_sample_counts_[("PCIe", 20.0)] == 100
    assert detector.link_sample_counts_[("NVLink", 80.0)] == 100


def test_link_conditioned_prediction_selects_threshold_per_row():
    pcie = np.linspace(0, 1, 100)
    nvlink = np.linspace(0, 0.2, 100)

    calibration = make_link_trace(
        np.concatenate([pcie, nvlink]),
        ["PCIe"] * 100 + ["NVLink"] * 100,
        [20.0] * 100 + [80.0] * 100,
    )

    detector = StageAnomalyDetector(
        quantile=0.90,
        min_samples=50,
        link_conditioned_transfer=True,
    ).fit(calibration)

    test_data = make_link_trace(
        [1.0, 0.2],
        ["PCIe", "NVLink"],
        [20.0, 80.0],
    )

    result = detector.predict(test_data)

    assert result["transfer_time_ms_anomaly"].tolist() == [True, True]
    assert result["transfer_time_ms_anomaly_threshold_ms"].tolist() == pytest.approx(
        [
            detector.link_thresholds_[("PCIe", 20.0)],
            detector.link_thresholds_[("NVLink", 80.0)],
        ]
    )
    assert result["transfer_time_ms_anomaly_calibration_count"].tolist() == [
        100, 100
    ]


def test_unseen_link_configuration_abstains():
    calibration = make_link_trace(
        [0.0] * 100,
        ["PCIe"] * 100,
        [20.0] * 100,
    )

    detector = StageAnomalyDetector(
        min_samples=50,
        link_conditioned_transfer=True,
    ).fit(calibration)

    test_data = make_link_trace(
        [10.0],
        ["InfiniBand"],
        [200.0],
    )

    result = detector.predict(test_data)

    assert pd.isna(result["transfer_time_ms_anomaly"].iloc[0])
    assert pd.isna(
        result["transfer_time_ms_anomaly_threshold_ms"].iloc[0]
    )
    assert result["transfer_time_ms_anomaly_calibration_count"].iloc[0] == 0
    assert result["transfer_time_ms_anomaly_status"].iloc[0] == (
        "insufficient_calibration"
    )


def test_link_conditioned_prediction_requires_link_metadata():
    calibration = make_link_trace(
        [0.0] * 100,
        ["PCIe"] * 100,
        [20.0] * 100,
    )

    detector = StageAnomalyDetector(
        min_samples=50,
        link_conditioned_transfer=True,
    ).fit(calibration)

    with pytest.raises(ValueError, match="Link-conditioned transfer prediction"):
        detector.predict(make_trace([1.0]))
