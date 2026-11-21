from typing import Any
import pandas as pd


TRACE_COLUMNS = [
    # Request identity and workload
    "request_id",
    "arrival_time_ms",
    "prompt_tokens",
    "output_tokens",
    "batch_id",
    "batch_size",

    # Component assignment
    "prefill_worker_id",
    "decode_worker_id",
    "link_id",
    "link_type",
    "link_shared",

    # Prefill timestamps and latency
    "prefill_queue_enter_ms",
    "prefill_start_ms",
    "prefill_completion_ms",
    "prefill_queue_wait_ms",
    "prefill_service_time_ms",

    # Communication timestamps and latency
    "transfer_start_time_ms",
    "transfer_completion_time_ms",
    "transfer_time_ms",
    "kv_cache_size_mb",
    "configured_bandwidth_mb_per_ms",
    "effective_bandwidth_mb_per_ms",
    "concurrent_transfers",
    "average_concurrent_transfers",
    "peak_concurrent_transfers",

    # Synchronization and admission
    "decode_admission_time_ms",
    "decode_execution_start_ms",
    "handoff_wait_ms",
    "admission_wait_ms",

    # Decode timestamps and latency
    "decode_completion_ms",
    "decode_service_time_ms",

    # End-to-end measurements
    "end_to_end_latency_ms",
    "ttft_ms",

    # Execution context
    "prefill_compute_capacity",
    "decode_compute_capacity",
    "prefill_average_compute_capacity",
    "decode_average_compute_capacity",
    "prefill_mapping_id",
    "decode_mapping_id",
    "prefill_queue_depth",
    "decode_queue_depth",
    "compute_interference",
    "memory_interference",
    "worker_utilization",
]


GROUND_TRUTH_COLUMNS = [
    "fault_id",
    "fault_type",
    "component_type",
    "component_id",
    "start_time_ms",
    "end_time_ms",
    "severity",
]


def _duration(
    start: Any,
    end: Any,
) -> float | None:
    """
    Calculate end - start.

    Return None if either timestamp is unavailable.
    """
    if start is None or end is None:
        return None

    return float(end) - float(start)


def build_trace_row(
    request,
    metadata: dict | None = None,
) -> dict:
    """
    Build one canonical telemetry row for a completed request.

    The request object is expected to contain the timestamp and
    assignment attributes defined in simulator/request.py.

    metadata can contain additional simulator measurements such as:
        batch_id
        batch_size
        link_type
        kv_cache_size_mb
        configured_bandwidth_mb_per_ms
        effective_bandwidth_mb_per_ms
        concurrent_transfers
        prefill_compute_capacity
        decode_compute_capacity
        mapping IDs
        queue depths
        interference measurements
    """
    metadata = metadata or {}

    arrival = request.arrival_time_ms

    prefill_start = request.prefill_start_ms
    prefill_completion = request.prefill_completion_ms

    transfer_start = request.transfer_start_ms
    transfer_completion = request.transfer_completion_ms

    decode_admission = request.decode_admission_ms
    decode_start = request.decode_start_ms
    decode_completion = request.decode_completion_ms

    row = {
        "request_id": request.request_id,
        "arrival_time_ms": arrival,
        "prompt_tokens": request.prompt_tokens,
        "output_tokens": request.output_tokens,

        "batch_id": metadata.get("batch_id"),
        "batch_size": metadata.get("batch_size", 1),

        "prefill_worker_id": request.prefill_worker_id,
        "decode_worker_id": request.decode_worker_id,
        "link_id": request.link_id,
        "link_type": metadata.get("link_type"),

        "prefill_queue_enter_ms": (
            request.prefill_queue_enter_ms
        ),
        "prefill_start_ms": prefill_start,
        "prefill_completion_ms": prefill_completion,

        "prefill_queue_wait_ms": _duration(
        request.prefill_queue_enter_ms
        if request.prefill_queue_enter_ms is not None
        else arrival,
        prefill_start,
),

        "prefill_service_time_ms": _duration(
            prefill_start,
            prefill_completion,
        ),

        "transfer_start_time_ms": transfer_start,
        "transfer_completion_time_ms": transfer_completion,

        "transfer_time_ms": _duration(
            transfer_start,
            transfer_completion,
        ),

        "kv_cache_size_mb": (
            request.kv_cache_size_mb
        ),

        "configured_bandwidth_mb_per_ms": metadata.get(
            "configured_bandwidth_mb_per_ms"
        ),

        "effective_bandwidth_mb_per_ms": metadata.get(
            "effective_bandwidth_mb_per_ms"
        ),

        "concurrent_transfers": metadata.get(
            "concurrent_transfers"
        ),
        "average_concurrent_transfers": metadata.get(
            "average_concurrent_transfers"
        ),
        "peak_concurrent_transfers": metadata.get(
            "peak_concurrent_transfers"
        ),
        "link_shared": metadata.get("link_shared"),

        "decode_admission_time_ms": decode_admission,
        "decode_execution_start_ms": decode_start,

        "handoff_wait_ms": _duration(
            transfer_completion,
            decode_admission,
        ),

        "admission_wait_ms": _duration(
            decode_admission,
            decode_start,
        ),

        "decode_completion_ms": decode_completion,

        "decode_service_time_ms": _duration(
            decode_start,
            decode_completion,
        ),

        "end_to_end_latency_ms": _duration(
            arrival,
            decode_completion,
        ),

        # This is a request-level approximation for now.
        # Later, token-level events should define actual TTFT.
        "ttft_ms": _duration(
            arrival,
            decode_start,
        ),

        "prefill_compute_capacity": metadata.get(
            "prefill_compute_capacity"
        ),

        "decode_compute_capacity": metadata.get(
            "decode_compute_capacity"
        ),

        "prefill_average_compute_capacity": metadata.get(
            "prefill_average_compute_capacity"
        ),

        "decode_average_compute_capacity": metadata.get(
            "decode_average_compute_capacity"
        ),

        "prefill_mapping_id": metadata.get(
            "prefill_mapping_id"
        ),

        "decode_mapping_id": metadata.get(
            "decode_mapping_id"
        ),

        "prefill_queue_depth": metadata.get(
            "prefill_queue_depth"
        ),

        "decode_queue_depth": metadata.get(
            "decode_queue_depth"
        ),

        "compute_interference": metadata.get(
            "compute_interference"
        ),

        "memory_interference": metadata.get(
            "memory_interference"
        ),

        "worker_utilization": metadata.get(
            "worker_utilization"
        ),
    }

    # Ensure a stable schema even when metadata is incomplete.
    return {
        column: row.get(column)
        for column in TRACE_COLUMNS
    }


def build_ground_truth_row(
    fault_id: str,
    fault_type: str,
    component_type: str,
    component_id: str,
    start_time_ms: float,
    end_time_ms: float,
    severity: float,
) -> dict:
    """
    Build a ground-truth record.

    This record belongs in a separate file and must not be passed
    to the detection algorithm.
    """
    if end_time_ms <= start_time_ms:
        raise ValueError(
            "end_time_ms must be greater than start_time_ms"
        )

    return {
        "fault_id": fault_id,
        "fault_type": fault_type,
        "component_type": component_type,
        "component_id": component_id,
        "start_time_ms": float(start_time_ms),
        "end_time_ms": float(end_time_ms),
        "severity": float(severity),
    }


def create_trace_dataframe(
    rows: list[dict],
) -> pd.DataFrame:
    """
    Create a DataFrame with the canonical trace schema.
    """
    return pd.DataFrame(
        rows,
        columns=TRACE_COLUMNS,
    )


def create_ground_truth_dataframe(
    rows: list[dict],
) -> pd.DataFrame:
    """
    Create a DataFrame with the ground-truth schema.
    """
    return pd.DataFrame(
        rows,
        columns=GROUND_TRUTH_COLUMNS,
    )


def validate_trace_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate required trace columns and basic timing consistency.

    Missing values are allowed for fields that are not yet populated.
    Negative stage durations are not allowed.
    """
    missing_columns = [
        column
        for column in TRACE_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing trace columns: {missing_columns}"
        )

    duration_columns = [
        "prefill_queue_wait_ms",
        "prefill_service_time_ms",
        "transfer_time_ms",
        "handoff_wait_ms",
        "admission_wait_ms",
        "decode_service_time_ms",
        "end_to_end_latency_ms",
    ]

    for column in duration_columns:
        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        invalid = values.dropna() < -1e-9

        if invalid.any():
            raise ValueError(
                f"Negative durations found in {column}"
            )

    if dataframe["request_id"].duplicated().any():
        raise ValueError("Duplicate request_id values found")


def save_trace_csv(
    rows: list[dict],
    output_path: str,
) -> pd.DataFrame:
    """
    Create, validate, and save a trace CSV.
    """
    dataframe = create_trace_dataframe(rows)

    validate_trace_dataframe(dataframe)

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return dataframe


def save_ground_truth_csv(
    rows: list[dict],
    output_path: str,
) -> pd.DataFrame:
    """
    Save ground-truth labels separately from the trace.
    """
    dataframe = create_ground_truth_dataframe(rows)

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return dataframe