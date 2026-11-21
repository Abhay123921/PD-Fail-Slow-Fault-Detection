import pytest

from pd_failslow.simulator.request import Request
from pd_failslow.workload.generator import generate_requests
from pd_failslow.workload.summary import summarize_workload


def test_workload_summary_statistics():
    requests = [
        Request("R1", 0.0, 100, 20),
        Request("R2", 10.0, 200, 40),
        Request("R3", 20.0, 300, 60),
        Request("R4", 30.0, 400, 80),
    ]

    summary = summarize_workload(
        requests,
        profile="test",
        seed=42,
    )

    assert summary["request_count"] == 4
    assert summary["profile"] == "test"
    assert summary["seed"] == 42
    assert summary["arrival_span_ms"] == 30.0
    assert summary["mean_interarrival_ms"] == 10.0
    assert summary["observed_arrival_rate_per_ms"] == pytest.approx(0.1)

    assert summary["prompt_tokens"]["mean"] == 250.0
    assert summary["prompt_tokens"]["median"] == 250.0
    assert summary["prompt_tokens"]["min"] == 100.0
    assert summary["prompt_tokens"]["max"] == 400.0

    assert summary["output_tokens"]["mean"] == 50.0


def test_empty_workload_summary():
    summary = summarize_workload([])

    assert summary["request_count"] == 0
    assert summary["arrival_span_ms"] is None
    assert summary["mean_interarrival_ms"] is None
    assert summary["observed_arrival_rate_per_ms"] is None
    assert summary["prompt_tokens"]["mean"] is None
    assert summary["output_tokens"]["mean"] is None


def test_single_request_workload_summary():
    requests = [Request("R1", 5.0, 128, 32)]

    summary = summarize_workload(requests)

    assert summary["request_count"] == 1
    assert summary["arrival_span_ms"] == 0.0
    assert summary["mean_interarrival_ms"] is None
    assert summary["observed_arrival_rate_per_ms"] is None
    assert summary["prompt_tokens"]["mean"] == 128.0
    assert summary["output_tokens"]["mean"] == 32.0


def test_simultaneous_arrivals_do_not_divide_by_zero():
    requests = [
        Request("R1", 5.0, 128, 32),
        Request("R2", 5.0, 256, 64),
    ]

    summary = summarize_workload(requests)

    assert summary["arrival_span_ms"] == 0.0
    assert summary["mean_interarrival_ms"] == 0.0
    assert summary["observed_arrival_rate_per_ms"] is None


def test_generated_workload_summary_is_reproducible():
    requests_a = generate_requests(num_requests=100, seed=42)
    requests_b = generate_requests(num_requests=100, seed=42)

    summary_a = summarize_workload(requests_a, profile="mixed", seed=42)
    summary_b = summarize_workload(requests_b, profile="mixed", seed=42)

    assert summary_a == summary_b


def test_configuration_metadata_is_copied():
    configuration = {"arrival_rate_per_ms": 0.03}

    summary = summarize_workload(
        [],
        configuration=configuration,
    )

    assert summary["configuration"] == configuration
    assert summary["configuration"] is not configuration
