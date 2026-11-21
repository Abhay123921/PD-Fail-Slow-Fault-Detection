import pytest

from pd_failslow.faults.compute_faults import (
    ComputeFault,
    ComputeFaultManager,
)


def test_compute_fault_capacity_over_time():
    manager = ComputeFaultManager()
    manager.register_fault(ComputeFault("F1", "decode_1", 5, 15, 0.30))

    assert manager.effective_capacity("decode_1", 100, 0) == 100
    assert manager.effective_capacity("decode_1", 100, 5) == pytest.approx(70)
    assert manager.effective_capacity("decode_1", 100, 14.9) == pytest.approx(70)
    assert manager.effective_capacity("decode_1", 100, 15) == 100


def test_overlapping_compute_faults_are_multiplicative():
    manager = ComputeFaultManager()
    manager.register_fault(ComputeFault("F1", "decode_1", 5, 15, 0.30))
    manager.register_fault(ComputeFault("F2", "decode_1", 10, 20, 0.20))

    assert manager.effective_capacity("decode_1", 100, 7) == pytest.approx(70)
    assert manager.effective_capacity("decode_1", 100, 12) == pytest.approx(56)
    assert manager.effective_capacity("decode_1", 100, 17) == pytest.approx(80)
    assert manager.effective_capacity("decode_1", 100, 20) == 100


def test_compute_fault_only_affects_target_worker():
    manager = ComputeFaultManager()
    manager.register_fault(ComputeFault("F1", "prefill_1", 5, 15, 0.40))

    assert manager.effective_capacity("prefill_1", 100, 10) == pytest.approx(60)
    assert manager.effective_capacity("decode_1", 100, 10) == 100


def test_duplicate_fault_id_is_rejected():
    manager = ComputeFaultManager()
    manager.register_fault(ComputeFault("F1", "decode_1", 5, 15, 0.30))

    with pytest.raises(ValueError):
        manager.register_fault(ComputeFault("F1", "decode_2", 5, 15, 0.20))


def test_invalid_fault_interval_and_severity_are_rejected():
    with pytest.raises(ValueError):
        ComputeFault("F1", "decode_1", 15, 5, 0.30)

    with pytest.raises(ValueError):
        ComputeFault("F2", "decode_1", 5, 15, 1.0)


from pd_failslow.faults.communication_faults import (
    CommunicationFault,
    CommunicationFaultManager,
)
from pd_failslow.simulator.communication import CommunicationLink


def test_communication_faults_are_applied_to_correct_link():
    manager = CommunicationFaultManager()
    manager.register_fault(
        CommunicationFault("C1", "link_1", 5, 15, 0.30)
    )

    link_1 = CommunicationLink("link_1", "network", 10.0)
    link_2 = CommunicationLink("link_2", "network", 10.0)

    manager.apply_to_link(link_1)
    manager.apply_to_link(link_2)

    assert "C1" in link_1.bandwidth_faults
    assert "C1" not in link_2.bandwidth_faults


def test_overlapping_communication_faults_are_multiplicative():
    manager = CommunicationFaultManager()
    manager.register_fault(
        CommunicationFault("C1", "link_1", 5, 15, 0.30)
    )
    manager.register_fault(
        CommunicationFault("C2", "link_1", 10, 20, 0.20)
    )

    link = CommunicationLink("link_1", "network", 100.0)
    manager.apply_to_link(link)

    assert link._bandwidth_at(0) == pytest.approx(100)
    assert link._bandwidth_at(7) == pytest.approx(70)
    assert link._bandwidth_at(12) == pytest.approx(56)
    assert link._bandwidth_at(17) == pytest.approx(80)
    assert link._bandwidth_at(20) == pytest.approx(100)


def test_communication_fault_ground_truth_records():
    manager = CommunicationFaultManager()
    manager.register_fault(
        CommunicationFault("C1", "link_1", 5, 15, 0.30)
    )

    record = manager.to_ground_truth_records()[0]

    assert record == {
        "fault_id": "C1",
        "fault_type": "communication",
        "component_type": "link",
        "component_id": "link_1",
        "start_time_ms": 5.0,
        "end_time_ms": 15.0,
        "severity": 0.30,
    }


from pd_failslow.faults.synchronization_faults import (
    SynchronizationFault,
    SynchronizationFaultManager,
)


def test_synchronization_fault_activation_boundaries():
    manager = SynchronizationFaultManager()
    manager.register_fault(
        SynchronizationFault("S1", "decode_1", 5, 15, 0.6, 10)
    )

    assert manager.additional_delay_ms("decode_1", 4.9) == 0
    assert manager.additional_delay_ms("decode_1", 5) == pytest.approx(6)
    assert manager.additional_delay_ms("decode_1", 14.9) == pytest.approx(6)
    assert manager.additional_delay_ms("decode_1", 15) == 0


def test_overlapping_synchronization_faults_accumulate_additively():
    manager = SynchronizationFaultManager()
    manager.register_fault(
        SynchronizationFault("S1", "decode_1", 5, 15, 0.5, 10)
    )
    manager.register_fault(
        SynchronizationFault("S2", "decode_1", 10, 20, 0.4, 20)
    )

    assert manager.additional_delay_ms("decode_1", 7) == pytest.approx(5)
    assert manager.additional_delay_ms("decode_1", 12) == pytest.approx(13)
    assert manager.additional_delay_ms("decode_1", 17) == pytest.approx(8)
    assert manager.additional_delay_ms("decode_1", 20) == 0


def test_synchronization_fault_only_affects_target_worker():
    manager = SynchronizationFaultManager()
    manager.register_fault(
        SynchronizationFault("S1", "decode_1", 5, 15, 0.5, 10)
    )

    assert manager.additional_delay_ms("decode_1", 10) == pytest.approx(5)
    assert manager.additional_delay_ms("decode_2", 10) == 0


def test_synchronization_fault_ground_truth_records():
    manager = SynchronizationFaultManager()
    manager.register_fault(
        SynchronizationFault("S1", "decode_1", 5, 15, 0.6, 10)
    )

    record = manager.to_ground_truth_records()[0]

    assert record == {
        "fault_id": "S1",
        "fault_type": "synchronization",
        "component_type": "worker",
        "component_id": "decode_1",
        "start_time_ms": 5.0,
        "end_time_ms": 15.0,
        "severity": 0.6,
    }


from types import SimpleNamespace

from pd_failslow.faults.injector import FaultInjector


def make_test_system():
    return SimpleNamespace(
        prefill_workers=[
            SimpleNamespace(worker_id="prefill_1")
        ],
        decode_workers=[
            SimpleNamespace(worker_id="decode_1")
        ],
        communication_links=[
            CommunicationLink("link_1", "network", 100.0),
            CommunicationLink("link_2", "network", 50.0),
        ],
    )


def test_fault_injector_registers_all_fault_types():
    injector = FaultInjector()

    injector.register_compute_fault(
        ComputeFault("F1", "prefill_1", 5, 15, 0.3)
    )
    injector.register_communication_fault(
        CommunicationFault("C1", "link_1", 5, 15, 0.2)
    )
    injector.register_synchronization_fault(
        SynchronizationFault("S1", "decode_1", 5, 15, 0.5, 10)
    )

    records = injector.to_ground_truth_records()

    assert len(records) == 3
    assert {record["fault_type"] for record in records} == {
        "compute", "communication", "synchronization"
    }


def test_fault_injector_prepares_communication_faults():
    injector = FaultInjector()
    injector.register_communication_fault(
        CommunicationFault("C1", "link_1", 5, 15, 0.3)
    )

    system = make_test_system()
    injector.prepare_run(system)

    assert "C1" in system.communication_links[0].bandwidth_faults
    assert "C1" not in system.communication_links[1].bandwidth_faults


def test_fault_injector_prepare_is_idempotent():
    injector = FaultInjector()
    injector.register_communication_fault(
        CommunicationFault("C1", "link_1", 5, 15, 0.3)
    )

    system = make_test_system()
    injector.prepare_run(system)
    injector.prepare_run(system)

    assert len(system.communication_links[0].bandwidth_faults) == 1


def test_fault_injector_reapplies_fault_after_link_reset():
    injector = FaultInjector()
    injector.register_communication_fault(
        CommunicationFault("C1", "link_1", 5, 15, 0.3)
    )

    system = make_test_system()

    injector.prepare_run(system)
    system.communication_links[0].reset()

    assert "C1" not in system.communication_links[0].bandwidth_faults

    injector.prepare_run(system)

    assert "C1" in system.communication_links[0].bandwidth_faults


def test_fault_injector_rejects_unknown_targets():
    injector = FaultInjector()
    injector.register_compute_fault(
        ComputeFault("F1", "unknown_worker", 5, 15, 0.3)
    )

    with pytest.raises(ValueError, match="Unknown compute-fault worker"):
        injector.prepare_run(make_test_system())
