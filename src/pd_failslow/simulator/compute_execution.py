"""Time-integrated compute execution under piecewise-constant capacity."""

import math


def calculate_compute_completion(
    *,
    start_time_ms: float,
    nominal_service_time_ms: float,
    worker_id: str,
    healthy_capacity: float,
    min_capacity: float,
    fault_manager,
) -> dict:
    """Calculate completion time by integrating effective compute capacity.

    The nominal service time is the healthy execution time calculated by the
    existing prefill/decode model. Required work is held constant while
    effective capacity changes at registered fault boundaries.
    """

    values = {
        "start_time_ms": start_time_ms,
        "nominal_service_time_ms": nominal_service_time_ms,
        "healthy_capacity": healthy_capacity,
        "min_capacity": min_capacity,
    }

    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    if start_time_ms < 0:
        raise ValueError("start_time_ms must be non-negative")
    if nominal_service_time_ms <= 0:
        raise ValueError("nominal_service_time_ms must be positive")
    if healthy_capacity <= 0:
        raise ValueError("healthy_capacity must be positive")
    if min_capacity <= 0:
        raise ValueError("min_capacity must be positive")
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise ValueError("worker_id must be a non-empty string")

    # Match the capacity floor used by the existing estimator.
    healthy_rate = max(float(healthy_capacity), float(min_capacity))
    required_work = float(nominal_service_time_ms) * healthy_rate

    boundaries = sorted(
        {
            boundary
            for fault in fault_manager.faults
            if fault.worker_id == worker_id
            for boundary in (fault.start_time_ms, fault.end_time_ms)
            if boundary > start_time_ms
        }
    )

    current_time = float(start_time_ms)
    remaining_work = required_work

    # Include the final unbounded interval after the last fault boundary.
    for boundary in boundaries + [math.inf]:
        raw_capacity = fault_manager.effective_capacity(
            worker_id=worker_id,
            healthy_capacity=healthy_capacity,
            time_ms=current_time,
        )
        effective_rate = max(float(raw_capacity), float(min_capacity))

        if math.isinf(boundary):
            duration = remaining_work / effective_rate
            current_time += duration
            remaining_work = 0.0
            break

        interval_duration = boundary - current_time
        available_work = effective_rate * interval_duration

        if remaining_work <= available_work:
            current_time += remaining_work / effective_rate
            remaining_work = 0.0
            break

        remaining_work -= available_work
        current_time = boundary

    service_time = current_time - float(start_time_ms)

    return {
        "completion_time_ms": float(current_time),
        "service_time_ms": float(service_time),
        "average_effective_capacity": float(required_work / service_time),
    }
