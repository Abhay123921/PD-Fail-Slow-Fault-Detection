"""Determine whether telemetry observations have valid healthy references."""

import numpy as np
import pandas as pd


VALID_REFERENCE_STATUSES = {"ok", "fallback_global"}


def get_comparable_mask(
    trace: pd.DataFrame,
    metric: str,
    allow_global_fallback: bool = True,
) -> pd.Series:
    """Return a Boolean mask identifying comparable stage observations."""

    if not isinstance(trace, pd.DataFrame):
        raise TypeError("trace must be a pandas DataFrame")

    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty string")

    required = [
        metric,
        f"{metric}_reference",
        f"{metric}_status",
    ]

    if any(column not in trace.columns for column in required):
        return pd.Series(False, index=trace.index, dtype=bool)

    accepted_statuses = {"ok"}

    if allow_global_fallback:
        accepted_statuses.add("fallback_global")

    observed = pd.to_numeric(trace[metric], errors="coerce")
    reference = pd.to_numeric(
        trace[f"{metric}_reference"], errors="coerce"
    )

    valid_status = trace[f"{metric}_status"].isin(accepted_statuses)

    return (
        valid_status
        & np.isfinite(observed)
        & np.isfinite(reference)
    ).astype(bool)
