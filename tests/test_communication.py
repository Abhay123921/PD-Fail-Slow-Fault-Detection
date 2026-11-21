import pytest

from types import SimpleNamespace

from pd_failslow.simulator.communication import CommunicationLink


def test_simultaneous_transfers_share_bandwidth_dynamically():

    link = CommunicationLink(
        link_id="L1",
        link_type="PCIe",
        bandwidth_mb_per_ms=10.0,
        fixed_latency_ms=0.0,
        shared=True,
    )

    request_a = SimpleNamespace(request_id="A")
    request_b = SimpleNamespace(request_id="B")

    link.start_transfer(
        request=request_a,
        current_time_ms=0.0,
        data_size_mb=100.0,
    )

    link.start_transfer(
        request=request_b,
        current_time_ms=0.0,
        data_size_mb=100.0,
    )

    transfers = {
        transfer.request_id: transfer
        for transfer in link.active_transfers
    }

    assert transfers["A"].completion_time_ms == 20.0
    assert transfers["B"].completion_time_ms == 20.0

    assert (
        transfers["A"].effective_bandwidth_mb_per_ms
        == 5.0
    )

    assert (
        transfers["B"].effective_bandwidth_mb_per_ms
        == 5.0
    )

def test_staggered_transfers_redistribute_bandwidth():

    link = CommunicationLink(
        link_id="L1",
        link_type="PCIe",
        bandwidth_mb_per_ms=10.0,
        fixed_latency_ms=0.0,
        shared=True,
    )

    request_a = SimpleNamespace(request_id="A")
    request_b = SimpleNamespace(request_id="B")

    link.start_transfer(
        request=request_a,
        current_time_ms=0.0,
        data_size_mb=100.0,
    )

    link.start_transfer(
        request=request_b,
        current_time_ms=5.0,
        data_size_mb=100.0,
    )

    transfers = {
        transfer.request_id: transfer
        for transfer in link.active_transfers
    }

    assert transfers["A"].completion_time_ms == 15.0
    assert transfers["B"].completion_time_ms == 20.0




def test_bandwidth_fault_changes_active_transfer_completion():
    link = CommunicationLink(
        link_id="L1",
        link_type="PCIe",
        bandwidth_mb_per_ms=10.0,
        fixed_latency_ms=0.0,
        shared=True,
    )

    link.schedule_bandwidth_fault(
        fault_id="F1",
        start_time_ms=5.0,
        end_time_ms=10.0,
        severity=0.5,
    )

    request = SimpleNamespace(request_id="A")

    link.start_transfer(
        request=request,
        current_time_ms=0.0,
        data_size_mb=100.0,
    )

    result = link.get_transfer_result("A")

    # 50 MB transmitted during 0-5 ms.
    # 25 MB transmitted during 5-10 ms at degraded bandwidth.
    # Remaining 25 MB takes 2.5 ms after restoration.
    assert result["completion_time_ms"] == 12.5


def test_overlapping_bandwidth_faults_combine_multiplicatively():
    link = CommunicationLink(
        link_id="L1",
        link_type="PCIe",
        bandwidth_mb_per_ms=10.0,
        fixed_latency_ms=0.0,
        shared=True,
    )

    link.schedule_bandwidth_fault(
        fault_id="F1",
        start_time_ms=0.0,
        end_time_ms=10.0,
        severity=0.3,
    )

    link.schedule_bandwidth_fault(
        fault_id="F2",
        start_time_ms=5.0,
        end_time_ms=15.0,
        severity=0.4,
    )

    request = SimpleNamespace(request_id="A")

    link.start_transfer(
        request=request,
        current_time_ms=0.0,
        data_size_mb=100.0,
    )

    result = link.get_transfer_result("A")

    # 0-5 ms: 10 * 0.7 = 7 MB/ms -> 35 MB
    # 5-10 ms: 10 * 0.7 * 0.6 = 4.2 MB/ms -> 21 MB
    # 10-15 ms: 10 * 0.6 = 6 MB/ms -> 30 MB
    # Remaining 14 MB at 10 MB/ms takes 1.4 ms.
    assert abs(result["completion_time_ms"] - 16.4) < 1e-9


def test_reset_restores_healthy_bandwidth_and_clears_faults():
    link = CommunicationLink(
        link_id="L1",
        link_type="PCIe",
        bandwidth_mb_per_ms=10.0,
    )

    link.schedule_bandwidth_fault(
        fault_id="F1",
        start_time_ms=0.0,
        end_time_ms=10.0,
        severity=0.5,
    )

    link.reset()

    assert link.bandwidth_mb_per_ms == 10.0
    assert len(link.active_transfers) == 0
    assert len(link.transfer_history) == 0
    assert len(link.bandwidth_faults) == 0

def test_actual_transfer_progress_across_fault_boundaries():
    link = CommunicationLink(
        link_id="L1",
        link_type="PCIe",
        bandwidth_mb_per_ms=10.0,
    )

    link.schedule_bandwidth_fault(
        fault_id="F1",
        start_time_ms=5.0,
        end_time_ms=10.0,
        severity=0.5,
    )

    link.start_transfer(
        SimpleNamespace(request_id="A"),
        current_time_ms=0.0,
        data_size_mb=100.0,
    )

    # Advancing to 10 ms should account for:
    # 0-5 ms at 10 MB/ms and 5-10 ms at 5 MB/ms.
    link._advance_to(10.0)

    transfer_a = link.transfer_history["A"]

    assert abs(transfer_a.remaining_data_mb - 25.0) < 1e-9

    link._recalculate_future_completions(10.0)

    assert abs(transfer_a.completion_time_ms - 12.5) < 1e-9


def test_legacy_bandwidth_reduction_preserves_healthy_baseline():
    link = CommunicationLink(
        link_id="L1",
        link_type="PCIe",
        bandwidth_mb_per_ms=10.0,
    )

    link.apply_bandwidth_reduction(0.5)

    assert link.bandwidth_mb_per_ms == 10.0
    assert link._healthy_bandwidth_mb_per_ms == 10.0

    link.reset()

    assert link.bandwidth_mb_per_ms == 10.0



def test_transfer_starting_during_active_fault_uses_degraded_bandwidth():
    link = CommunicationLink(
        link_id="L1",
        link_type="PCIe",
        bandwidth_mb_per_ms=10.0,
        fixed_latency_ms=2.0,
    )

    link.schedule_bandwidth_fault(
        fault_id="F1",
        start_time_ms=0.0,
        end_time_ms=10.0,
        severity=0.5,
    )

    result = link.start_transfer(
        SimpleNamespace(request_id="A"),
        current_time_ms=2.0,
        data_size_mb=100.0,
    )

    assert abs(result["completion_time_ms"] - 17.0) < 1e-9
    assert abs(
        result["effective_bandwidth_mb_per_ms"]
        - (100.0 / 13.0)
    ) < 1e-9


def test_simultaneous_transfers_record_average_and_peak_concurrency():
    link = CommunicationLink(
        link_id="L1",
        link_type="PCIe",
        bandwidth_mb_per_ms=10.0,
        fixed_latency_ms=0.0,
        shared=True,
    )

    link.start_transfer(
        SimpleNamespace(request_id="A"),
        current_time_ms=0.0,
        data_size_mb=100.0,
    )
    link.start_transfer(
        SimpleNamespace(request_id="B"),
        current_time_ms=0.0,
        data_size_mb=100.0,
    )

    result_a = link.get_transfer_result("A")
    result_b = link.get_transfer_result("B")

    for result in (result_a, result_b):
        assert result["average_concurrent_transfers"] == 2.0
        assert result["peak_concurrent_transfers"] == 2


def test_staggered_transfers_record_time_weighted_concurrency():
    link = CommunicationLink(
        link_id="L1",
        link_type="PCIe",
        bandwidth_mb_per_ms=10.0,
        fixed_latency_ms=0.0,
        shared=True,
    )

    link.start_transfer(
        SimpleNamespace(request_id="A"),
        current_time_ms=0.0,
        data_size_mb=100.0,
    )
    link.start_transfer(
        SimpleNamespace(request_id="B"),
        current_time_ms=5.0,
        data_size_mb=100.0,
    )

    result_a = link.get_transfer_result("A")
    result_b = link.get_transfer_result("B")

    expected_average = 5.0 / 15.0 + 10.0 / 15.0 * 2.0

    for result in (result_a, result_b):
        assert result["average_concurrent_transfers"] == pytest.approx(
            expected_average
        )
        assert result["peak_concurrent_transfers"] == 2
