from dataclasses import dataclass


@dataclass
class DecodeConfig:
    """
    Parameters for the abstract decode performance model.
    """
    alpha_ms_per_token: float = 0.05
    memory_interference_factor: float = 0.30
    min_capacity: float = 0.01
    min_mapping_efficiency: float = 0.01


def estimate_decode_time(
    output_tokens: int,
    compute_capacity: float = 1.0,
    mapping_efficiency: float = 1.0,
    memory_interference: float = 0.0,
    config: DecodeConfig | None = None,
) -> float:
    """
    Estimate total decode service time in milliseconds.

    This is a request-level abstraction. It does not yet simulate
    individual token-generation events or inter-token latency.
    """
    if output_tokens <= 0:
        raise ValueError("output_tokens must be greater than zero")

    if compute_capacity <= 0:
        raise ValueError("compute_capacity must be greater than zero")

    if mapping_efficiency <= 0:
        raise ValueError("mapping_efficiency must be greater than zero")

    if not 0.0 <= memory_interference <= 1.0:
        raise ValueError("memory_interference must be between 0 and 1")

    if config is None:
        config = DecodeConfig()

    effective_capacity = max(
        compute_capacity,
        config.min_capacity
    )

    effective_mapping = max(
        mapping_efficiency,
        config.min_mapping_efficiency
    )

    base_time = (
        config.alpha_ms_per_token * output_tokens
    ) / (
        effective_capacity * effective_mapping
    )

    interference_multiplier = (
        1.0
        + config.memory_interference_factor * memory_interference
    )

    return float(base_time * interference_multiplier)


def execute_decode(
    request,
    worker,
    start_time_ms: float,
    mapping_efficiency: float = 1.0,
    memory_interference: float = 0.0,
    config: DecodeConfig | None = None,
) -> dict:
    """
    Calculate decode execution timestamps.

    The caller is responsible for scheduling the completion event.
    """
    service_time_ms = estimate_decode_time(
        output_tokens=request.output_tokens,
        compute_capacity=worker.effective_compute_capacity(),
        mapping_efficiency=mapping_efficiency,
        memory_interference=memory_interference,
        config=config,
    )

    completion_time_ms = start_time_ms + service_time_ms

    return {
        "request_id": request.request_id,
        "worker_id": worker.worker_id,
        "start_time_ms": float(start_time_ms),
        "completion_time_ms": float(completion_time_ms),
        "service_time_ms": float(service_time_ms),
    }