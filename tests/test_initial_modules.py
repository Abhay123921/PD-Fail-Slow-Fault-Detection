from pd_failslow.simulator.prefill import estimate_prefill_time
from pd_failslow.simulator.decode import estimate_decode_time
from pd_failslow.simulator.communication import CommunicationLink
from pd_failslow.simulator.synchronization import SynchronizationModel
from pd_failslow.workload.generator import generate_requests
from pd_failslow.telemetry.schemas import (
    build_ground_truth_row,
    create_ground_truth_dataframe,
)


def test_prefill_time_increases_with_prompt_length():
    short = estimate_prefill_time(128)
    long = estimate_prefill_time(1024)

    assert long > short


def test_decode_time_increases_with_output_length():
    short = estimate_decode_time(32)
    long = estimate_decode_time(512)

    assert long > short


def test_lower_bandwidth_increases_transfer_time():
    link = CommunicationLink(
        link_id="L1",
        link_type="PCIe",
        bandwidth_mb_per_ms=20.0,
        fixed_latency_ms=0.2,
    )

    normal = link.estimate_transfer_time(
        data_size_mb=100,
        concurrent_transfers=1,
    )

    contended = link.estimate_transfer_time(
        data_size_mb=100,
        concurrent_transfers=2,
    )

    assert contended > normal


def test_handoff_delay():
    sync = SynchronizationModel()

    sync.set_handoff_delay("D1", 5.0)

    class DummyRequest:
        request_id = "REQ_001"

    result = sync.calculate_handoff(
        request=DummyRequest(),
        transfer_completion_time_ms=100.0,
        decode_worker_id="D1",
    )

    assert result["handoff_wait_ms"] == 5.1
    assert result["decode_admission_time_ms"] == 105.1


def test_workload_generation():
    requests = generate_requests(
        num_requests=20,
        seed=42,
    )

    assert len(requests) == 20

    assert requests == sorted(
        requests,
        key=lambda request: request.arrival_time_ms,
    )


def test_ground_truth_schema():
    row = build_ground_truth_row(
        fault_id="F001",
        fault_type="bandwidth_reduction",
        component_type="communication_link",
        component_id="L1",
        start_time_ms=100,
        end_time_ms=200,
        severity=0.4,
    )

    dataframe = create_ground_truth_dataframe([row])

    assert len(dataframe) == 1
    assert dataframe.iloc[0]["component_id"] == "L1"