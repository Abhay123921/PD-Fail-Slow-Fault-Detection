import numpy as np

from pd_failslow.simulator.request import Request


def _sample_length(
    rng: np.random.Generator,
    distribution: str,
    minimum: int,
    maximum: int,
    fixed_value: int | None = None,
) -> int:
    """
    Generate a prompt or output length.
    """
    if minimum <= 0 or maximum <= 0:
        raise ValueError("Token lengths must be positive")

    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")

    if distribution == "uniform":
        return int(rng.integers(minimum, maximum + 1))

    if distribution == "fixed":
        if fixed_value is None:
            raise ValueError(
                "fixed_value is required for fixed distribution"
            )

        if fixed_value <= 0:
            raise ValueError("fixed_value must be positive")

        return int(fixed_value)

    raise ValueError(
        f"Unsupported length distribution: {distribution}"
    )


def _generate_arrival_times(
    rng: np.random.Generator,
    num_requests: int,
    distribution: str,
    rate_per_ms: float,
) -> np.ndarray:
    """
    Generate non-negative request arrival timestamps.

    constant:
        Requests arrive at fixed intervals.

    poisson:
        Inter-arrival times follow an exponential distribution.
    """
    if num_requests < 0:
        raise ValueError("num_requests cannot be negative")

    if rate_per_ms <= 0:
        raise ValueError("rate_per_ms must be positive")

    if num_requests == 0:
        return np.array([], dtype=float)

    if distribution == "constant":
        interval_ms = 1.0 / rate_per_ms

        return (
            np.arange(num_requests, dtype=float) * interval_ms
        )

    if distribution == "poisson":
        inter_arrival_times = rng.exponential(
            scale=1.0 / rate_per_ms,
            size=num_requests,
        )

        return np.cumsum(inter_arrival_times)

    raise ValueError(
        f"Unsupported arrival distribution: {distribution}"
    )


def generate_requests(
    num_requests: int = 1000,
    seed: int = 42,

    arrival_distribution: str = "poisson",
    arrival_rate_per_ms: float = 0.03,

    prompt_distribution: str = "uniform",
    min_prompt_tokens: int = 128,
    max_prompt_tokens: int = 2048,
    fixed_prompt_tokens: int | None = None,

    output_distribution: str = "uniform",
    min_output_tokens: int = 32,
    max_output_tokens: int = 512,
    fixed_output_tokens: int | None = None,
) -> list[Request]:
    """
    Generate reproducible PD inference requests.

    Returns
    -------
    list[Request]
        Requests sorted by arrival time.
    """
    if num_requests < 0:
        raise ValueError("num_requests cannot be negative")

    rng = np.random.default_rng(seed)

    arrival_times = _generate_arrival_times(
        rng=rng,
        num_requests=num_requests,
        distribution=arrival_distribution,
        rate_per_ms=arrival_rate_per_ms,
    )

    requests = []

    for index in range(num_requests):

        prompt_tokens = _sample_length(
            rng=rng,
            distribution=prompt_distribution,
            minimum=min_prompt_tokens,
            maximum=max_prompt_tokens,
            fixed_value=fixed_prompt_tokens,
        )

        output_tokens = _sample_length(
            rng=rng,
            distribution=output_distribution,
            minimum=min_output_tokens,
            maximum=max_output_tokens,
            fixed_value=fixed_output_tokens,
        )

        request = Request(
            request_id=f"REQ_{index:06d}",
            arrival_time_ms=float(arrival_times[index]),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

        requests.append(request)

    return sorted(
        requests,
        key=lambda request: request.arrival_time_ms
    )


def generate_workload_profile(
    profile: str,
    num_requests: int = 1000,
    seed: int = 42,
) -> list[Request]:
    """
    Convenience function for generating predefined workloads.

    Supported profiles:
        stable
        long_prompt
        long_output
        high_arrival
        mixed
    """
    if profile == "stable":
        return generate_requests(
            num_requests=num_requests,
            seed=seed,
            arrival_distribution="constant",
            arrival_rate_per_ms=0.02,
            prompt_distribution="fixed",
            fixed_prompt_tokens=512,
            min_prompt_tokens=512,
            max_prompt_tokens=512,
            output_distribution="fixed",
            fixed_output_tokens=128,
            min_output_tokens=128,
            max_output_tokens=128,
        )

    if profile == "long_prompt":
        return generate_requests(
            num_requests=num_requests,
            seed=seed,
            arrival_rate_per_ms=0.02,
            min_prompt_tokens=1024,
            max_prompt_tokens=4096,
            min_output_tokens=64,
            max_output_tokens=256,
        )

    if profile == "long_output":
        return generate_requests(
            num_requests=num_requests,
            seed=seed,
            arrival_rate_per_ms=0.02,
            min_prompt_tokens=128,
            max_prompt_tokens=512,
            min_output_tokens=512,
            max_output_tokens=2048,
        )

    if profile == "high_arrival":
        return generate_requests(
            num_requests=num_requests,
            seed=seed,
            arrival_rate_per_ms=0.08,
            min_prompt_tokens=128,
            max_prompt_tokens=1024,
            min_output_tokens=32,
            max_output_tokens=256,
        )

    if profile == "mixed":
        return generate_requests(
            num_requests=num_requests,
            seed=seed,
            arrival_rate_per_ms=0.04,
            min_prompt_tokens=128,
            max_prompt_tokens=4096,
            min_output_tokens=32,
            max_output_tokens=1024,
        )

    raise ValueError(f"Unknown workload profile: {profile}")