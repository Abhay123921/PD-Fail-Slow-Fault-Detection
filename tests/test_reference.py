import numpy as np
import pandas as pd
import pytest

from pd_failslow.detection.reference import HealthyReferenceModel


def make_healthy_trace(n=100):
    prompt = np.linspace(100, 2000, n)
    output = np.linspace(32, 512, n)

    return pd.DataFrame({
        "request_id": [f"r{i}" for i in range(n)],
        "prompt_tokens": prompt,
        "output_tokens": output,
        "prefill_service_time_ms": 0.02 * prompt,
        "prefill_queue_wait_ms": np.zeros(n),
        "transfer_time_ms": 0.01 * prompt / 20 + 0.2,
        "handoff_wait_ms": np.full(n, 0.1),
        "admission_wait_ms": np.zeros(n),
        "decode_service_time_ms": 0.05 * output,
        "prefill_worker_id": ["P1" if i % 2 == 0 else "P2"
                              for i in range(n)],
        "decode_worker_id": ["D1" if i % 2 == 0 else "D2"
                             for i in range(n)],
        "link_id": ["L1"] * n,
        "link_type": ["PCIe"] * n,
        "link_shared": [True] * n,
        "configured_bandwidth_mb_per_ms": [20.0] * n,
        "kv_cache_size_mb": 0.01 * prompt,
        "average_concurrent_transfers": np.ones(n),
        "prefill_queue_depth": np.zeros(n),
        "decode_queue_depth": np.zeros(n),
        "arrival_time_ms": np.arange(n) * 50.0,
    })


def test_reference_fit_and_predict():
    trace = make_healthy_trace()
    model = HealthyReferenceModel(min_samples=10)
    model.fit(trace)

    result = model.predict(trace)

    assert len(result) == len(trace)
    assert "prefill_service_time_ms_reference" in result.columns
    assert "prefill_service_time_ms_mad" in result.columns
    assert "prefill_service_time_ms_sample_count" in result.columns


def test_reference_does_not_use_ground_truth():
    trace = make_healthy_trace()
    trace["fault_id"] = ["F001"] * len(trace)
    trace["fault_type"] = ["compute_reduction"] * len(trace)
    trace["severity"] = [0.8] * len(trace)

    model = HealthyReferenceModel(min_samples=10)
    model.fit(trace)

    assert "fault_id" not in model.feature_columns_
    assert "fault_type" not in model.feature_columns_
    assert "severity" not in model.feature_columns_


def test_worker_ids_are_treated_as_equivalent():
    trace = make_healthy_trace()
    model = HealthyReferenceModel(min_samples=10)
    model.fit(trace)

    assert model.worker_equivalence_enabled is True


def test_insufficient_reference_data_is_handled():
    trace = make_healthy_trace(n=5)
    model = HealthyReferenceModel(min_samples=10)
    model.fit(trace)

    result = model.predict(trace)

    assert "prefill_service_time_ms_status" in result.columns
    assert (
        result["prefill_service_time_ms_status"]
        == "insufficient_reference"
    ).all()


def test_invalid_empty_training_data():
    model = HealthyReferenceModel()

    with pytest.raises(ValueError):
        model.fit(pd.DataFrame())


def test_service_references_are_workload_conditioned():
    trace = make_healthy_trace(n=100)
    model = HealthyReferenceModel(min_samples=10)
    model.fit(trace)

    result = model.predict(trace)

    # Different prompt-size groups should have different references.
    assert (
        result["prefill_service_time_ms_reference"].nunique() > 1
    )

    # Different output-size groups should have different references.
    assert (
        result["decode_service_time_ms_reference"].nunique() > 1
    )


def test_out_of_distribution_payload_is_flagged():
    training = make_healthy_trace(n=100)
    model = HealthyReferenceModel(min_samples=10).fit(training)

    evaluation = make_healthy_trace(n=2)
    evaluation["kv_cache_size_mb"] = [600.0, 600.0]

    result = model.predict(evaluation)

    assert (
        result["transfer_time_ms_status"]
        == "out_of_distribution"
    ).all()

    assert result["transfer_time_ms_reference"].isna().all()


def test_global_reference_uses_one_reference_per_stage():
    trace = make_healthy_trace(n=100)
    model = HealthyReferenceModel(
        min_samples=10,
        workload_conditioned=False,
    ).fit(trace)

    result = model.predict(trace)

    assert (
        result["prefill_service_time_ms_reference"].nunique()
        == 1
    )
    assert (
        result["decode_service_time_ms_reference"].nunique()
        == 1
    )
    assert (
        result["prefill_service_time_ms_status"]
        == "ok"
    ).all()


def test_global_reference_preserves_ood_protection():
    training = make_healthy_trace(n=100)
    model = HealthyReferenceModel(
        min_samples=10,
        workload_conditioned=False,
    ).fit(training)

    evaluation = make_healthy_trace(n=2)
    evaluation["kv_cache_size_mb"] = [600.0, 600.0]

    result = model.predict(evaluation)

    assert (
        result["transfer_time_ms_status"]
        == "out_of_distribution"
    ).all()
    assert result["transfer_time_ms_reference"].isna().all()
