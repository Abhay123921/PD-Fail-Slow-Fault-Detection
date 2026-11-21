from typing import Any, Sequence

import numpy as np

from pd_failslow.simulator.request import Request


def _length_statistics(values: Sequence[int]) -> dict[str, float | None]:
    """Calculate descriptive statistics for token lengths."""
    if not values:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
            "min": None,
            "max": None,
        }

    array = np.asarray(values, dtype=float)

    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_workload(
    requests: Sequence[Request],
    *,
    profile: str | None = None,
    seed: int | None = None,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a reproducible descriptive summary of a request workload.

    This function does not modify requests or simulator state.
    Arrival rate is estimated from the first-to-last arrival span:
    (number of requests - 1) / arrival span.
    """
    request_count = len(requests)

    arrival_times = sorted(
        float(request.arrival_time_ms) for request in requests
    )

    if request_count >= 2:
        inter_arrivals = np.diff(arrival_times)
        arrival_span = arrival_times[-1] - arrival_times[0]

        mean_interarrival = float(np.mean(inter_arrivals))
        observed_arrival_rate = (
            float((request_count - 1) / arrival_span)
            if arrival_span > 0
            else None
        )
    else:
        arrival_span = 0.0 if request_count == 1 else None
        mean_interarrival = None
        observed_arrival_rate = None

    prompt_lengths = [int(request.prompt_tokens) for request in requests]
    output_lengths = [int(request.output_tokens) for request in requests]

    return {
        "profile": profile,
        "seed": seed,
        "request_count": request_count,
        "arrival_span_ms": arrival_span,
        "mean_interarrival_ms": mean_interarrival,
        "observed_arrival_rate_per_ms": observed_arrival_rate,
        "prompt_tokens": _length_statistics(prompt_lengths),
        "output_tokens": _length_statistics(output_lengths),
        "configuration": dict(configuration) if configuration is not None else {},
    }
