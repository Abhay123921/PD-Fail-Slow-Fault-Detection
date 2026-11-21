from types import SimpleNamespace

import numpy as np
import pandas as pd

from pd_failslow.detection.communication_diagnostic import (
    CommunicationDiagnostic,
)


def make_diagnostic(shared=True):
    system = SimpleNamespace(
        communication_links=[
            SimpleNamespace(
                link_id="L1",
                bandwidth_mb_per_ms=20.0,
                fixed_latency_ms=0.2,
                shared=shared,
            )
        ]
    )
    return CommunicationDiagnostic(system)


def make_trace(**overrides):
    row = {
        "link_id": "L1",
        "link_shared": True,
        "configured_bandwidth_mb_per_ms": 20.0,
        "kv_cache_size_mb": 20.0,
        "transfer_time_ms": 2.2,
        "concurrent_transfers": 2,
        "average_concurrent_transfers": 2.0,
        "peak_concurrent_transfers": 2,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_shared_link_constant_concurrency():
    result = make_diagnostic().predict(make_trace())

    assert np.isclose(
        result.loc[0, "communication_expected_time_ms"], 2.2
    )
    assert np.isclose(result.loc[0, "communication_residual_ms"], 0)
    assert result.loc[0, "communication_concurrency_stable"]
    assert result.loc[0, "communication_diagnostic_status"] == "estimated"


def test_shared_link_single_transfer():
    trace = make_trace(
        transfer_time_ms=1.2,
        concurrent_transfers=1,
        average_concurrent_transfers=1,
        peak_concurrent_transfers=1,
    )
    result = make_diagnostic().predict(trace)

    assert np.isclose(
        result.loc[0, "communication_expected_time_ms"], 1.2
    )


def test_variable_concurrency_abstains():
    trace = make_trace(
        concurrent_transfers=2,
        average_concurrent_transfers=3.5,
        peak_concurrent_transfers=5,
    )
    result = make_diagnostic().predict(trace)

    assert result.loc[0, "communication_diagnostic_status"] == (
        "variable_concurrency"
    )
    assert np.isnan(result.loc[0, "communication_expected_time_ms"])
    assert np.isnan(result.loc[0, "communication_residual_ms"])


def test_non_shared_link_does_not_require_concurrency():
    trace = make_trace(
        link_shared=False,
        concurrent_transfers=np.nan,
        average_concurrent_transfers=np.nan,
        peak_concurrent_transfers=np.nan,
        transfer_time_ms=1.2,
    )
    result = make_diagnostic(shared=False).predict(trace)

    assert np.isclose(
        result.loc[0, "communication_expected_time_ms"], 1.2
    )
    assert result.loc[0, "communication_diagnostic_status"] == "estimated"


def test_unknown_link_is_flagged():
    result = make_diagnostic().predict(
        make_trace(link_id="UNKNOWN")
    )

    assert result.loc[0, "communication_diagnostic_status"] == (
        "unknown_link"
    )


def test_configuration_mismatch_is_flagged():
    result = make_diagnostic().predict(
        make_trace(configured_bandwidth_mb_per_ms=40.0)
    )

    assert result.loc[0, "communication_diagnostic_status"] == (
        "configuration_mismatch"
    )
