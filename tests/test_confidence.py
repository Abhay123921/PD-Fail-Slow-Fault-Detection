import numpy as np
import pandas as pd
import pytest

from pd_failslow.detection.confidence import EvidenceConfidence


METRIC = "transfer_time_ms"


def make_trace(
    reference_status="ok",
    reference_count=100,
    calibration_count=500,
    anomaly_status="scored",
    persistence_status="evaluated",
):
    return pd.DataFrame({
        f"{METRIC}_status": [reference_status],
        f"{METRIC}_sample_count": [reference_count],
        f"{METRIC}_anomaly_calibration_count": [calibration_count],
        f"{METRIC}_anomaly_status": [anomaly_status],
        f"{METRIC}_persistence_status": [persistence_status],
    })


def test_full_quality_evidence_gets_maximum_score():
    result = EvidenceConfidence().predict(make_trace())

    assert result[
        f"{METRIC}_evidence_quality_score"
    ].iloc[0] == pytest.approx(1.0)


def test_global_fallback_reduces_quality_score():
    result = EvidenceConfidence().predict(
        make_trace(reference_status="fallback_global")
    )

    assert result[
        f"{METRIC}_evidence_quality_score"
    ].iloc[0] == pytest.approx(0.88)


def test_low_sample_support_reduces_score():
    result = EvidenceConfidence().predict(
        make_trace(reference_count=25, calibration_count=100)
    )

    assert result[
        f"{METRIC}_evidence_quality_score"
    ].iloc[0] == pytest.approx(0.69)


def test_ood_reference_causes_abstention():
    result = EvidenceConfidence().predict(
        make_trace(reference_status="out_of_distribution")
    )

    assert np.isnan(
        result[f"{METRIC}_evidence_quality_score"].iloc[0]
    )
    assert result[
        f"{METRIC}_evidence_quality_status"
    ].iloc[0] == "not_comparable"


def test_insufficient_persistence_history_causes_abstention():
    result = EvidenceConfidence().predict(
        make_trace(persistence_status="insufficient_history")
    )

    assert np.isnan(
        result[f"{METRIC}_evidence_quality_score"].iloc[0]
    )
    assert result[
        f"{METRIC}_evidence_quality_status"
    ].iloc[0] == "insufficient_evidence"


def test_missing_inputs_cause_abstention():
    result = EvidenceConfidence().predict(pd.DataFrame(index=[0]))

    assert np.isnan(
        result[f"{METRIC}_evidence_quality_score"].iloc[0]
    )
    assert result[
        f"{METRIC}_evidence_quality_status"
    ].iloc[0] == "missing_inputs"


def test_invalid_sample_counts_cause_abstention():
    result = EvidenceConfidence().predict(
        make_trace(reference_count=0)
    )

    assert np.isnan(
        result[f"{METRIC}_evidence_quality_score"].iloc[0]
    )


def test_original_input_is_not_modified():
    trace = make_trace()
    original = trace.copy()

    EvidenceConfidence().predict(trace)

    pd.testing.assert_frame_equal(trace, original)
