import pytest
import numpy as np

from pd_failslow.simulator.system import PDSystem
from pd_failslow.simulator.engine import PDSimulationEngine
from pd_failslow.workload.generator import generate_requests


def create_test_system():

    prefill_config = {
        "workers": [
            {
                "id": "P1",
                "compute_capacity": 1.0,
                "memory_capacity": 1.0,
            },
            {
                "id": "P2",
                "compute_capacity": 1.0,
                "memory_capacity": 1.0,
            },
        ]
    }

    decode_config = {
        "workers": [
            {
                "id": "D1",
                "compute_capacity": 1.0,
                "memory_capacity": 1.0,
            },
            {
                "id": "D2",
                "compute_capacity": 1.0,
                "memory_capacity": 1.0,
            },
        ]
    }

    communication_config = {
        "links": [
            {
                "id": "L1",
                "link_type": "PCIe",
                "bandwidth_mb_per_ms": 20.0,
                "fixed_latency_ms": 0.2,
                "shared": True,
            },
            {
                "id": "L2",
                "link_type": "NVLink",
                "bandwidth_mb_per_ms": 80.0,
                "fixed_latency_ms": 0.05,
                "shared": True,
            },
        ]
    }

    scheduler_config = {
        "prefill_policy": "shortest_queue",
        "decode_policy": "shortest_queue",

        "link_routes": {
            "P1->D1": "L1",
            "P1->D2": "L2",
            "P2->D1": "L2",
            "P2->D2": "L1",
        },
    }

    synchronization_config = {
        "normal_handoff_delay_ms": 0.1
    }

    return PDSystem(
        prefill_config=prefill_config,
        decode_config=decode_config,
        communication_config=communication_config,
        scheduler_config=scheduler_config,
        synchronization_config=synchronization_config,
    )


def test_pd_simulation_end_to_end():

    system = create_test_system()

    engine = PDSimulationEngine(
        system=system,
        kv_cache_mb_per_prompt_token=0.01,
    )

    requests = generate_requests(
        num_requests=10,
        seed=42,
        arrival_distribution="constant",
        arrival_rate_per_ms=0.03,
    )

    trace = engine.run(requests)

    assert len(trace) == 10

    assert trace["request_id"].nunique() == 10

    assert trace["prefill_service_time_ms"].notna().all()

    assert trace["transfer_time_ms"].notna().all()

    assert trace["handoff_wait_ms"].notna().all()

    assert trace["decode_service_time_ms"].notna().all()

    assert trace["end_to_end_latency_ms"].notna().all()

    assert (
        trace["end_to_end_latency_ms"] > 0
    ).all()

    assert (
        trace["transfer_completion_time_ms"]
        >= trace["transfer_start_time_ms"]
    ).all()

    assert (
        trace["decode_admission_time_ms"]
        >= trace["transfer_completion_time_ms"]
    ).all()

    assert (
        trace["decode_execution_start_ms"]
        >= trace["decode_admission_time_ms"]
    ).all()

    assert (
        trace["decode_completion_ms"]
        >= trace["decode_execution_start_ms"]
    ).all()

def test_engine_uses_final_shared_transfer_completion_times():
    from pd_failslow.simulator.request import Request

    system = create_test_system()
    engine = PDSimulationEngine(
        system=system,
        kv_cache_mb_per_prompt_token=0.01,
    )

    requests = [
        Request("R1", 0.0, 1000, 100),
        Request("R2", 0.0, 1000, 100),
    ]

    trace = engine.run(requests)

    assert len(trace) == 2
    assert set(trace["link_id"]) == {"L1"}
    assert trace["transfer_start_time_ms"].nunique() == 1

    # Identical simultaneous transfers should finish together.
    assert trace["transfer_completion_time_ms"].nunique() == 1

    assert (trace["handoff_wait_ms"].round(6) == 0.1).all()

    expected_transfer_time = (
        trace["transfer_completion_time_ms"]
        - trace["transfer_start_time_ms"]
    )
    assert (
        trace["transfer_time_ms"].round(6)
        == expected_transfer_time.round(6)
    ).all()

    expected_admission = (
        trace["transfer_completion_time_ms"] + 0.1
    )
    assert (
        trace["decode_admission_time_ms"].round(6)
        == expected_admission.round(6)
    ).all()



def test_compute_fault_changes_prefill_service_time_and_preserves_ground_truth():
    from pd_failslow.faults.compute_faults import ComputeFault
    from pd_failslow.faults.injector import FaultInjector

    requests_healthy = generate_requests(
        num_requests=10,
        seed=42,
        arrival_distribution="constant",
        arrival_rate_per_ms=0.03,
    )

    requests_faulty = generate_requests(
        num_requests=10,
        seed=42,
        arrival_distribution="constant",
        arrival_rate_per_ms=0.03,
    )

    healthy_system = create_test_system()
    healthy_engine = PDSimulationEngine(system=healthy_system)
    healthy_trace = healthy_engine.run(requests_healthy)

    faulty_system = create_test_system()
    injector = FaultInjector()
    injector.register_compute_fault(
        ComputeFault(
            fault_id="CF_P1",
            worker_id="P1",
            start_time_ms=0,
            end_time_ms=100000,
            severity=0.50,
        )
    )

    faulty_engine = PDSimulationEngine(
        system=faulty_system,
        fault_injector=injector,
    )
    faulty_trace = faulty_engine.run(requests_faulty)

    # Compare the same requests under healthy and faulty conditions.
    comparison = healthy_trace[
        ["request_id", "prefill_worker_id", "prefill_service_time_ms"]
    ].merge(
        faulty_trace[
            [
                "request_id",
                "prefill_worker_id",
                "prefill_service_time_ms",
                "prefill_compute_capacity",
            ]
        ],
        on="request_id",
        suffixes=("_healthy", "_faulty"),
    )

    assert (
        comparison["prefill_worker_id_healthy"]
        == comparison["prefill_worker_id_faulty"]
    ).all()

    p1 = comparison[
        comparison["prefill_worker_id_faulty"] == "P1"
    ]
    p2 = comparison[
        comparison["prefill_worker_id_faulty"] == "P2"
    ]

    assert not p1.empty
    assert not p2.empty

    # P1 has half its configured compute capacity.
    assert (
        p1["prefill_compute_capacity"] == 0.5
    ).all()

    # With identical workload and configuration, service time doubles.
    np.testing.assert_allclose(
        p1["prefill_service_time_ms_faulty"].to_numpy(),
        p1["prefill_service_time_ms_healthy"].to_numpy() * 2,
    )

    # P2 remains unaffected.
    assert (
        p2["prefill_compute_capacity"] == 1.0
    ).all()
    np.testing.assert_allclose(
        p2["prefill_service_time_ms_faulty"].to_numpy(),
        p2["prefill_service_time_ms_healthy"].to_numpy(),
    )

    # Fault labels remain separate from detector-facing telemetry.
    ground_truth = faulty_engine.collector.get_ground_truth_dataframe()
    assert len(ground_truth) == 1
    assert ground_truth.iloc[0]["fault_id"] == "CF_P1"
    assert ground_truth.iloc[0]["component_id"] == "P1"
    assert "fault_id" not in faulty_trace.columns

    # Temporary fault capacity must not leak into worker state.
    for worker in faulty_system.prefill_workers:
        assert worker.current_compute_capacity == pytest.approx(
            worker.compute_capacity
        )


def test_communication_fault_increases_transfer_time_only_on_target_link():
    from pd_failslow.simulator.request import Request
    from pd_failslow.faults.communication_faults import CommunicationFault
    from pd_failslow.faults.injector import FaultInjector

    def configure_deterministic_routes(system):
        scheduler = system.scheduler

        prefill_workers = iter([
            scheduler.get_prefill_worker("P1"),
            scheduler.get_prefill_worker("P1"),
        ])

        decode_workers = iter([
            scheduler.get_decode_worker("D1"),
            scheduler.get_decode_worker("D2"),
        ])

        scheduler.select_prefill_worker = lambda: next(prefill_workers)
        scheduler.select_decode_worker = lambda: next(decode_workers)

    def make_requests():
        return [
            Request("R1", 0.0, 1000, 100),
            Request("R2", 0.0, 1000, 100),
        ]

    healthy_system = create_test_system()
    configure_deterministic_routes(healthy_system)

    healthy_engine = PDSimulationEngine(system=healthy_system)
    healthy_trace = healthy_engine.run(make_requests())

    faulty_system = create_test_system()
    configure_deterministic_routes(faulty_system)

    injector = FaultInjector()
    injector.register_communication_fault(
        CommunicationFault(
            fault_id="CF_L1",
            link_id="L1",
            start_time_ms=0,
            end_time_ms=100000,
            severity=0.50,
        )
    )

    faulty_engine = PDSimulationEngine(
        system=faulty_system,
        fault_injector=injector,
    )
    faulty_trace = faulty_engine.run(make_requests())

    columns = [
        "request_id",
        "prefill_worker_id",
        "decode_worker_id",
        "link_id",
        "transfer_time_ms",
        "effective_bandwidth_mb_per_ms",
                "link_shared",
        "average_concurrent_transfers",
        "peak_concurrent_transfers",
    ]

    comparison = healthy_trace[columns].merge(
        faulty_trace[columns],
        on="request_id",
        suffixes=("_healthy", "_faulty"),
    )

    for column in [
        "prefill_worker_id",
        "decode_worker_id",
        "link_id",
    ]:
        assert (
            comparison[f"{column}_healthy"]
            == comparison[f"{column}_faulty"]
        ).all()

        # Verify concurrency telemetry is populated in the request trace.
    assert comparison["link_shared_healthy"].notna().all()
    assert comparison["average_concurrent_transfers_healthy"].notna().all()
    assert comparison["peak_concurrent_transfers_healthy"].notna().all()

    # Both requests use separate links, so each sees one active transfer.
    np.testing.assert_allclose(
        comparison["average_concurrent_transfers_healthy"].to_numpy(),
        1.0,
    )

    assert (
        comparison["peak_concurrent_transfers_healthy"] == 1
    ).all()

    l1 = comparison[comparison["link_id_faulty"] == "L1"]
    l2 = comparison[comparison["link_id_faulty"] == "L2"]

    assert len(l1) == 1
    assert len(l2) == 1

    assert (
        l1["effective_bandwidth_mb_per_ms_faulty"]
        < l1["effective_bandwidth_mb_per_ms_healthy"]
    ).all()

    np.testing.assert_allclose(
        l2["effective_bandwidth_mb_per_ms_faulty"].to_numpy(),
        l2["effective_bandwidth_mb_per_ms_healthy"].to_numpy(),
    )

    assert (
        l1["transfer_time_ms_faulty"]
        > l1["transfer_time_ms_healthy"]
    ).all()

    np.testing.assert_allclose(
        l2["transfer_time_ms_faulty"].to_numpy(),
        l2["transfer_time_ms_healthy"].to_numpy(),
    )

    ground_truth = faulty_engine.collector.get_ground_truth_dataframe()

    assert len(ground_truth) == 1
    assert ground_truth.iloc[0]["fault_id"] == "CF_L1"
    assert ground_truth.iloc[0]["component_type"] == "link"
    assert ground_truth.iloc[0]["component_id"] == "L1"
    assert "fault_id" not in faulty_trace.columns


def test_synchronization_fault_increases_handoff_delay_only_on_target_worker():
    from pd_failslow.simulator.request import Request
    from pd_failslow.faults.synchronization_faults import SynchronizationFault
    from pd_failslow.faults.injector import FaultInjector

    def configure_deterministic_routes(system):
        scheduler = system.scheduler

        prefill_workers = iter([
            scheduler.get_prefill_worker("P1"),
            scheduler.get_prefill_worker("P1"),
        ])

        decode_workers = iter([
            scheduler.get_decode_worker("D1"),
            scheduler.get_decode_worker("D2"),
        ])

        scheduler.select_prefill_worker = lambda: next(prefill_workers)
        scheduler.select_decode_worker = lambda: next(decode_workers)

    def make_requests():
        return [
            Request("R1", 0.0, 1000, 100),
            Request("R2", 0.0, 1000, 100),
        ]

    healthy_system = create_test_system()
    configure_deterministic_routes(healthy_system)

    healthy_engine = PDSimulationEngine(system=healthy_system)
    healthy_trace = healthy_engine.run(make_requests())

    faulty_system = create_test_system()
    configure_deterministic_routes(faulty_system)

    injector = FaultInjector()
    injector.register_synchronization_fault(
        SynchronizationFault(
            fault_id="SF_D1",
            component_id="D1",
            start_time_ms=0,
            end_time_ms=100000,
            severity=0.5,
            max_additional_delay_ms=10.0,
        )
    )

    faulty_engine = PDSimulationEngine(
        system=faulty_system,
        fault_injector=injector,
    )
    faulty_trace = faulty_engine.run(make_requests())

    columns = [
        "request_id",
        "prefill_worker_id",
        "decode_worker_id",
        "link_id",
        "transfer_completion_time_ms",
        "transfer_time_ms",
        "handoff_wait_ms",
        "decode_admission_time_ms",
        "decode_service_time_ms",
    ]

    comparison = healthy_trace[columns].merge(
        faulty_trace[columns],
        on="request_id",
        suffixes=("_healthy", "_faulty"),
    )

    for column in [
        "prefill_worker_id",
        "decode_worker_id",
        "link_id",
    ]:
        assert (
            comparison[f"{column}_healthy"]
            == comparison[f"{column}_faulty"]
        ).all()

    d1 = comparison[comparison["decode_worker_id_faulty"] == "D1"]
    d2 = comparison[comparison["decode_worker_id_faulty"] == "D2"]

    assert len(d1) == 1
    assert len(d2) == 1

    # Communication must be unaffected.
    for column in [
        "transfer_completion_time_ms",
        "transfer_time_ms",
    ]:
        np.testing.assert_allclose(
            d1[f"{column}_faulty"].to_numpy(),
            d1[f"{column}_healthy"].to_numpy(),
        )

    # D1 handoff delay increases by exactly 5 ms.
    np.testing.assert_allclose(
        d1["handoff_wait_ms_faulty"].to_numpy(),
        d1["handoff_wait_ms_healthy"].to_numpy() + 5.0,
    )

    np.testing.assert_allclose(
        d1["decode_admission_time_ms_faulty"].to_numpy(),
        d1["decode_admission_time_ms_healthy"].to_numpy() + 5.0,
    )

    # D2 remains unaffected.
    for column in [
        "handoff_wait_ms",
        "decode_admission_time_ms",
        "decode_service_time_ms",
    ]:
        np.testing.assert_allclose(
            d2[f"{column}_faulty"].to_numpy(),
            d2[f"{column}_healthy"].to_numpy(),
        )

    # Decode execution duration for D1 should remain unchanged.
    np.testing.assert_allclose(
        d1["decode_service_time_ms_faulty"].to_numpy(),
        d1["decode_service_time_ms_healthy"].to_numpy(),
    )

    # Ground truth must identify D1, without contaminating trace telemetry.
    ground_truth = faulty_engine.collector.get_ground_truth_dataframe()

    assert len(ground_truth) == 1
    assert ground_truth.iloc[0]["fault_id"] == "SF_D1"
    assert ground_truth.iloc[0]["fault_type"] == "synchronization"
    assert ground_truth.iloc[0]["component_type"] == "worker"
    assert ground_truth.iloc[0]["component_id"] == "D1"
    assert "fault_id" not in faulty_trace.columns


def test_compute_fault_starting_mid_prefill_is_integrated_end_to_end():
    from pd_failslow.faults.compute_faults import ComputeFault
    from pd_failslow.faults.injector import FaultInjector

    healthy_requests = generate_requests(num_requests=1, seed=101)
    faulty_requests = generate_requests(num_requests=1, seed=101)

    healthy_engine = PDSimulationEngine(system=create_test_system())
    healthy_trace = healthy_engine.run(healthy_requests)

    baseline = healthy_trace.iloc[0]
    fault_start = (
        baseline["prefill_start_ms"]
        + baseline["prefill_service_time_ms"] / 2.0
    )

    injector = FaultInjector()
    injector.register_compute_fault(
        ComputeFault(
            fault_id="CF_MID_PREFILL",
            worker_id=baseline["prefill_worker_id"],
            start_time_ms=fault_start,
            end_time_ms=fault_start + 10000.0,
            severity=0.5,
        )
    )

    faulty_engine = PDSimulationEngine(
        system=create_test_system(),
        fault_injector=injector,
    )
    faulty_trace = faulty_engine.run(faulty_requests)
    faulty = faulty_trace.iloc[0]

    assert faulty["prefill_start_ms"] == pytest.approx(
        baseline["prefill_start_ms"]
    )
    assert faulty["prefill_compute_capacity"] == pytest.approx(1.0)
    assert faulty["prefill_service_time_ms"] == pytest.approx(
        baseline["prefill_service_time_ms"] * 1.5
    )
    assert faulty["prefill_average_compute_capacity"] == pytest.approx(
        2.0 / 3.0
    )

    assert "fault_id" not in faulty_trace.columns
    ground_truth = faulty_engine.collector.get_ground_truth_dataframe()
    assert len(ground_truth) == 1
    assert ground_truth.iloc[0]["fault_id"] == "CF_MID_PREFILL"


def test_compute_fault_starting_mid_decode_is_integrated_end_to_end():
    from pd_failslow.faults.compute_faults import ComputeFault
    from pd_failslow.faults.injector import FaultInjector

    healthy_requests = generate_requests(num_requests=1, seed=202)
    faulty_requests = generate_requests(num_requests=1, seed=202)

    healthy_engine = PDSimulationEngine(system=create_test_system())
    healthy_trace = healthy_engine.run(healthy_requests)

    baseline = healthy_trace.iloc[0]
    fault_start = (
        baseline["decode_execution_start_ms"]
        + baseline["decode_service_time_ms"] / 2.0
    )

    injector = FaultInjector()
    injector.register_compute_fault(
        ComputeFault(
            fault_id="CF_MID_DECODE",
            worker_id=baseline["decode_worker_id"],
            start_time_ms=fault_start,
            end_time_ms=fault_start + 10000.0,
            severity=0.5,
        )
    )

    faulty_engine = PDSimulationEngine(
        system=create_test_system(),
        fault_injector=injector,
    )
    faulty_trace = faulty_engine.run(faulty_requests)
    faulty = faulty_trace.iloc[0]

    assert faulty["decode_execution_start_ms"] == pytest.approx(
        baseline["decode_execution_start_ms"]
    )
    assert faulty["decode_compute_capacity"] == pytest.approx(1.0)
    assert faulty["decode_service_time_ms"] == pytest.approx(
        baseline["decode_service_time_ms"] * 1.5
    )
    assert faulty["decode_average_compute_capacity"] == pytest.approx(
        2.0 / 3.0
    )

    assert "fault_id" not in faulty_trace.columns
    ground_truth = faulty_engine.collector.get_ground_truth_dataframe()
    assert len(ground_truth) == 1
    assert ground_truth.iloc[0]["fault_id"] == "CF_MID_DECODE"


def test_controlled_worker_pair_assignments():
    from pd_failslow.simulator.request import Request

    assignments = {
        "R1": ("P1", "D1"),
        "R2": ("P1", "D2"),
        "R3": ("P2", "D1"),
        "R4": ("P2", "D2"),
    }

    engine = PDSimulationEngine(
        system=create_test_system(),
        kv_cache_mb_per_prompt_token=0.01,
        worker_pair_assignments=assignments,
    )

    requests = [
        Request(request_id, 0.0, 256, 32)
        for request_id in assignments
    ]

    trace = engine.run(requests)

    observed = {
        row.request_id: (
            row.prefill_worker_id,
            row.decode_worker_id,
            row.link_id,
        )
        for row in trace.itertuples()
    }

    assert observed == {
        "R1": ("P1", "D1", "L1"),
        "R2": ("P1", "D2", "L2"),
        "R3": ("P2", "D1", "L2"),
        "R4": ("P2", "D2", "L1"),
    }


def test_controlled_worker_pair_assignments_require_every_request():
    from pd_failslow.simulator.request import Request

    engine = PDSimulationEngine(
        system=create_test_system(),
        worker_pair_assignments={"R1": ("P1", "D1")},
    )

    requests = [
        Request("R1", 0.0, 256, 32),
        Request("R2", 1.0, 256, 32),
    ]

    with pytest.raises(ValueError, match="Missing worker-pair assignments"):
        engine.run(requests)


def test_controlled_worker_pair_assignments_reject_unknown_workers():
    with pytest.raises(ValueError, match="Unknown prefill worker"):
        PDSimulationEngine(
            system=create_test_system(),
            worker_pair_assignments={"R1": ("P9", "D1")},
        )
