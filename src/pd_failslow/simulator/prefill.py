from dataclasses import dataclass


@dataclass
class PrefillConfig:
    """
    Parameters for the abstract prefill performance model.

    alpha_ms_per_token:
        Calibration constant. It is a simulator parameter, not a
        measurement of a specific real LLM or GPU.

    memory_interference_factor:
        Controls how strongly memory interference increases service time.
    """
    alpha_ms_per_token: float = 0.02
    memory_interference_factor: float = 0.25
    min_capacity: float = 0.01
    min_mapping_efficiency: float = 0.01


def estimate_prefill_time(
    prompt_tokens: int,
    compute_capacity: float = 1.0,
    mapping_efficiency: float = 1.0,
    memory_interference: float = 0.0,
    config: PrefillConfig | None = None,
) -> float:
    """
    Estimate prefill service time in milliseconds.

    Parameters
    ----------
    prompt_tokens:
        Number of input tokens.

    compute_capacity:
        Effective compute capacity of the selected prefill worker.
        A value of 1.0 represents nominal configured capacity.

    mapping_efficiency:
        Abstract execution/mapping efficiency. Must be positive.

    memory_interference:
        Normalized memory interference in [0, 1].
        0 means no interference; 1 means maximum configured interference.

    config:
        Optional PrefillConfig.

    Returns
    -------
    float
        Estimated prefill service time in milliseconds.
    """
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be greater than zero")

    if compute_capacity <= 0:
        raise ValueError("compute_capacity must be greater than zero")

    if mapping_efficiency <= 0:
        raise ValueError("mapping_efficiency must be greater than zero")

    if not 0.0 <= memory_interference <= 1.0:
        raise ValueError("memory_interference must be between 0 and 1")

    if config is None:
        config = PrefillConfig()

    effective_capacity = max(
        compute_capacity,
        config.min_capacity
    )

    effective_mapping = max(
        mapping_efficiency,
        config.min_mapping_efficiency
    )

    base_time = (
        config.alpha_ms_per_token * prompt_tokens
    ) / (
        effective_capacity * effective_mapping
    )

    interference_multiplier = (
        1.0
        + config.memory_interference_factor * memory_interference
    )

    return float(base_time * interference_multiplier)


def execute_prefill(
    request,
    worker,
    start_time_ms: float,
    mapping_efficiency: float = 1.0,
    memory_interference: float = 0.0,
    config: PrefillConfig | None = None,
) -> dict:
    """
    Calculate prefill execution timestamps for a request.

    This function does not modify the simulation clock.

    The caller (engine.py) is responsible for scheduling the completion
    event at the returned completion_time_ms.
    """
    service_time_ms = estimate_prefill_time(
        prompt_tokens=request.prompt_tokens,
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