"""Request-level evaluation of fail-slow detection."""

import numpy as np
import pandas as pd

from pd_failslow.evaluation.metrices import binary_classification_metrics


def _boolean_prediction(series):
    """Convert nullable anomaly indicators into ordinary booleans."""
    return series.eq(True).fillna(False).to_numpy(dtype=bool)


def _interval_overlap(stage_start, stage_end, fault_start, fault_end):
    """Check overlap between execution and fault intervals."""
    stage_start = pd.to_numeric(stage_start, errors="coerce")
    stage_end = pd.to_numeric(stage_end, errors="coerce")

    valid = (
        stage_start.notna()
        & stage_end.notna()
        & np.isfinite(stage_start)
        & np.isfinite(stage_end)
        & (stage_end > stage_start)
    )

    return (
        valid
        & (stage_start < fault_end)
        & (fault_start < stage_end)
    ).to_numpy(dtype=bool)


def build_request_level_labels(trace, ground_truth, fault_type, prediction_mode="persistent"):
    """
    Build request-level ground-truth and prediction labels.

    Supported fault types: compute, communication, synchronization.

    Ground truth is derived exclusively from the separate fault table.
    Predictions are derived exclusively from trace anomaly columns.
    """
    if fault_type not in {"compute", "communication", "synchronization"}:
        raise ValueError("Unsupported fault_type")

    if prediction_mode not in {"raw", "persistent"}:
        raise ValueError("prediction_mode must be raw or persistent")
    prediction_suffix = (
        "anomaly" if prediction_mode == "raw" else "persistent_anomaly"
    )

    if not isinstance(trace, pd.DataFrame):
        raise TypeError("trace must be a pandas DataFrame")

    if not isinstance(ground_truth, pd.DataFrame):
        raise TypeError("ground_truth must be a pandas DataFrame")

    gt_required = {
        "fault_type",
        "component_id",
        "start_time_ms",
        "end_time_ms",
    }
    missing_gt = gt_required - set(ground_truth.columns)
    if missing_gt:
        raise ValueError(f"Missing ground-truth columns: {sorted(missing_gt)}")

    if "request_id" not in trace.columns:
        raise ValueError("trace must contain request_id")

    required = {
        "compute": {
            "prefill_worker_id",
            "prefill_start_ms",
            "prefill_completion_ms",
            "decode_worker_id",
            "decode_execution_start_ms",
            "decode_completion_ms",
            f"prefill_service_time_ms_{prediction_suffix}",
            f"decode_service_time_ms_{prediction_suffix}",
        },
        "communication": {
            "link_id",
            "transfer_start_time_ms",
            "transfer_completion_time_ms",
            f"transfer_time_ms_{prediction_suffix}",
        },
        "synchronization": {
            "decode_worker_id",
            "transfer_completion_time_ms",
            f"handoff_wait_ms_{prediction_suffix}",
        },
    }

    missing = required[fault_type] - set(trace.columns)
    if missing:
        raise ValueError(f"Missing trace columns: {sorted(missing)}")

    gt = ground_truth.loc[
        ground_truth["fault_type"].eq(fault_type)
    ]

    n = len(trace)
    y_true = np.zeros(n, dtype=bool)
    y_pred = np.zeros(n, dtype=bool)

    if fault_type == "compute":
        prefill_prediction = _boolean_prediction(
            trace[f"prefill_service_time_ms_{prediction_suffix}"]
        )
        decode_prediction = _boolean_prediction(
            trace[f"decode_service_time_ms_{prediction_suffix}"]
        )
        y_pred = prefill_prediction | decode_prediction

    elif fault_type == "communication":
        transfer_prediction = _boolean_prediction(
            trace[f"transfer_time_ms_{prediction_suffix}"]
        )
        y_pred = transfer_prediction

    else:
        handoff_prediction = _boolean_prediction(
            trace[f"handoff_wait_ms_{prediction_suffix}"]
        )
        y_pred = handoff_prediction

    for fault in gt.itertuples(index=False):
        component_id = fault.component_id
        fault_start = float(fault.start_time_ms)
        fault_end = float(fault.end_time_ms)

        if not np.isfinite(fault_start) or not np.isfinite(fault_end):
            raise ValueError("Fault interval must contain finite timestamps")

        if fault_end <= fault_start:
            raise ValueError("Fault end must be greater than fault start")

        if fault_type == "compute":
            prefill_match = (
                trace["prefill_worker_id"].eq(component_id).to_numpy()
            )
            decode_match = (
                trace["decode_worker_id"].eq(component_id).to_numpy()
            )

            prefill_exposure = _interval_overlap(
                trace["prefill_start_ms"],
                trace["prefill_completion_ms"],
                fault_start,
                fault_end,
            ) & prefill_match

            decode_exposure = _interval_overlap(
                trace["decode_execution_start_ms"],
                trace["decode_completion_ms"],
                fault_start,
                fault_end,
            ) & decode_match

            y_true |= prefill_exposure | decode_exposure


        elif fault_type == "communication":
            component_match = (
                trace["link_id"].eq(component_id).to_numpy()
            )

            exposure = _interval_overlap(
                trace["transfer_start_time_ms"],
                trace["transfer_completion_time_ms"],
                fault_start,
                fault_end,
            ) & component_match

            y_true |= exposure

        else:
            component_match = (
                trace["decode_worker_id"].eq(component_id).to_numpy()
            )

            completion = pd.to_numeric(
                trace["transfer_completion_time_ms"],
                errors="coerce",
            )

            valid_time = (
                completion.notna()
                & np.isfinite(completion)
            ).to_numpy(dtype=bool)

            exposure = (
                component_match
                & valid_time
                & (completion.to_numpy() >= fault_start)
                & (completion.to_numpy() < fault_end)
            )

            y_true |= exposure

    details = pd.DataFrame({
        "request_id": trace["request_id"].to_numpy(),
        "fault_type": fault_type,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    return details


def evaluate_request_detection(trace, ground_truth, fault_type, prediction_mode="persistent"):
    """Evaluate request-level detection for one fault type."""
    details = build_request_level_labels(
        trace=trace,
        ground_truth=ground_truth,
        fault_type=fault_type,
        prediction_mode=prediction_mode,
    )

    metrics = binary_classification_metrics(
        details["y_true"],
        details["y_pred"],
    )

    return {
        "fault_type": fault_type,
        "prediction_mode": prediction_mode,
        "metrics": metrics,
        "details": details,
    }
