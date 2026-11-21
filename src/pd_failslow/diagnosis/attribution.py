"""Map persistent stage anomalies to component-level evidence."""

import numpy as np
import pandas as pd


STAGE_ATTRIBUTION = {
    "prefill_service_time_ms": {
        "component_column": "prefill_worker_id",
        "fault_type": "compute",
        "component_type": "worker",
        "role": "direct",
        "timestamp_column": "prefill_completion_ms",
    },
    "decode_service_time_ms": {
        "component_column": "decode_worker_id",
        "fault_type": "compute",
        "component_type": "worker",
        "role": "direct",
        "timestamp_column": "decode_completion_ms",
    },
    "transfer_time_ms": {
        "component_column": "link_id",
        "fault_type": "communication",
        "component_type": "link",
        "role": "direct",
        "timestamp_column": "transfer_completion_time_ms",
    },
    "handoff_wait_ms": {
        "component_column": "decode_worker_id",
        "fault_type": "synchronization",
        "component_type": "worker",
        "role": "direct",
        "timestamp_column": "decode_admission_time_ms",
    },
    "prefill_queue_wait_ms": {
        "component_column": None,
        "fault_type": None,
        "component_type": None,
        "role": "context_only",
        "timestamp_column": "prefill_start_ms",
    },
    "admission_wait_ms": {
        "component_column": None,
        "fault_type": None,
        "component_type": None,
        "role": "context_only",
        "timestamp_column": "decode_execution_start_ms",
    },
}

OUTPUT_COLUMNS = [
    "request_id",
    "stage",
    "fault_type",
    "component_type",
    "component_id",
    "evidence_role",
    "timestamp_ms",
    "positive_residual_ms",
    "anomaly_excess_ms",
    "persistence_count",
    "evidence_quality_score",
]


def build_attribution_candidates(
    trace: pd.DataFrame,
) -> pd.DataFrame:
    """Convert persistent stage anomalies into attribution evidence.

    Direct evidence identifies a suspected fault type and component.
    Context-only evidence records persistent queue/admission delays
    without assigning blame to a component.

    Ground-truth fault records and fault severity are never used.
    """

    if not isinstance(trace, pd.DataFrame):
        raise TypeError("trace must be a pandas DataFrame")

    candidates = []

    for metric, mapping in STAGE_ATTRIBUTION.items():
        persistent_col = f"{metric}_persistent_anomaly"
        persistence_status_col = f"{metric}_persistence_status"

        if persistent_col not in trace.columns:
            continue

        if persistence_status_col not in trace.columns:
            continue

        component_column = mapping["component_column"]
        timestamp_column = mapping["timestamp_column"]

        for index, row in trace.iterrows():
            persistent = row[persistent_col]

            if pd.isna(persistent) or not bool(persistent):
                continue

            if row[persistence_status_col] != "evaluated":
                continue

            component_id = None

            if component_column is not None:
                if component_column not in trace.columns:
                    continue

                component_id = row[component_column]

                if pd.isna(component_id):
                    continue

            timestamp = (
                row[timestamp_column]
                if timestamp_column in trace.columns
                else np.nan
            )

            candidates.append({
                "request_id": row.get("request_id", index),
                "stage": metric,
                "fault_type": mapping["fault_type"],
                "component_type": mapping["component_type"],
                "component_id": component_id,
                "evidence_role": mapping["role"],
                "timestamp_ms": timestamp,
                "positive_residual_ms": row.get(
                    f"{metric}_positive_residual_ms", np.nan
                ),
                "anomaly_excess_ms": row.get(
                    f"{metric}_anomaly_excess_ms", np.nan
                ),
                "persistence_count": row.get(
                    f"{metric}_persistence_count", np.nan
                ),
                "evidence_quality_score": row.get(
                    f"{metric}_evidence_quality_score", np.nan
                ),
            })

    return pd.DataFrame(
        candidates,
        columns=OUTPUT_COLUMNS,
    )
