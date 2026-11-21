"""Prepare workload-conditioned stage residuals for fault detection."""

import numpy as np
import pandas as pd


STAGE_METRICS = (
    "prefill_service_time_ms",
    "prefill_queue_wait_ms",
    "transfer_time_ms",
    "handoff_wait_ms",
    "admission_wait_ms",
    "decode_service_time_ms",
)

VALID_REFERENCE_STATUSES = {"ok", "fallback_global"}


def prepare_stage_residuals(trace: pd.DataFrame) -> pd.DataFrame:
    """Add signed and positive residuals for comparable stage observations.

    A stage is comparable only when its observed latency and healthy
    reference are finite and its reference status is either 'ok' or
    'fallback_global'.

    Original telemetry and reference columns are preserved.
    """

    if not isinstance(trace, pd.DataFrame):
        raise TypeError("trace must be a pandas DataFrame")

    result = trace.copy()

    for metric in STAGE_METRICS:
        reference_col = f"{metric}_reference"
        status_col = f"{metric}_status"

        residual_col = f"{metric}_residual_ms"
        positive_col = f"{metric}_positive_residual_ms"
        comparable_col = f"{metric}_comparable"

        if metric not in result.columns:
            observed = pd.Series(np.nan, index=result.index)
        else:
            observed = pd.to_numeric(
                result[metric], errors="coerce"
            )

        if reference_col not in result.columns:
            reference = pd.Series(np.nan, index=result.index)
        else:
            reference = pd.to_numeric(
                result[reference_col], errors="coerce"
            )

        if status_col not in result.columns:
            valid_status = pd.Series(False, index=result.index)
        else:
            valid_status = result[status_col].isin(
                VALID_REFERENCE_STATUSES
            )

        comparable = (
            valid_status
            & np.isfinite(observed)
            & np.isfinite(reference)
        )

        residual = (observed - reference).where(comparable)

        result[residual_col] = residual
        result[positive_col] = residual.clip(lower=0)
        result[comparable_col] = comparable

    return result
