import numpy as np
import pandas as pd

from pd_failslow.detection.preprocessing import (
    STAGE_METRICS,
    prepare_stage_residuals,
)


def test_positive_and_negative_residuals():
    metric = "transfer_time_ms"

    trace = pd.DataFrame({
        metric: [12.0, 8.0],
        f"{metric}_reference": [10.0, 10.0],
        f"{metric}_status": ["ok", "ok"],
    })

    result = prepare_stage_residuals(trace)

    assert result[f"{metric}_residual_ms"].tolist() == [2.0, -2.0]
    assert result[f"{metric}_positive_residual_ms"].tolist() == [2.0, 0.0]
    assert result[f"{metric}_comparable"].tolist() == [True, True]


def test_invalid_reference_status_is_not_comparable():
    metric = "transfer_time_ms"

    trace = pd.DataFrame({
        metric: [20.0, 20.0, 20.0],
        f"{metric}_reference": [10.0, 10.0, 10.0],
        f"{metric}_status": [
            "out_of_distribution",
            "insufficient_reference",
            "ok",
        ],
    })

    result = prepare_stage_residuals(trace)

    assert result[f"{metric}_comparable"].tolist() == [
        False, False, True
    ]
    assert np.isnan(result[f"{metric}_residual_ms"].iloc[0])
    assert result[f"{metric}_residual_ms"].iloc[2] == 10.0


def test_missing_and_non_numeric_values_are_handled():
    metric = "transfer_time_ms"

    trace = pd.DataFrame({
        metric: ["invalid", 10.0],
        f"{metric}_reference": [5.0, np.nan],
        f"{metric}_status": ["ok", "ok"],
    })

    result = prepare_stage_residuals(trace)

    assert result[f"{metric}_comparable"].tolist() == [False, False]
    assert result[f"{metric}_residual_ms"].isna().all()


def test_all_stages_receive_output_columns():
    result = prepare_stage_residuals(pd.DataFrame(index=[0]))

    for metric in STAGE_METRICS:
        assert f"{metric}_residual_ms" in result
        assert f"{metric}_positive_residual_ms" in result
        assert f"{metric}_comparable" in result


def test_input_dataframe_is_not_modified():
    metric = "transfer_time_ms"

    trace = pd.DataFrame({
        metric: [12.0],
        f"{metric}_reference": [10.0],
        f"{metric}_status": ["ok"],
    })

    original = trace.copy()
    prepare_stage_residuals(trace)

    pd.testing.assert_frame_equal(trace, original)
