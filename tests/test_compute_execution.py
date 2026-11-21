import pytest

from pd_failslow.faults.compute_faults import (
    ComputeFault,
    ComputeFaultManager,
)
from pd_failslow.simulator.compute_execution import (
    calculate_compute_completion,
)


def make_manager(*faults):
    manager = ComputeFaultManager()
    for fault in faults:
        manager.register_fault(fault)
    return manager


def calculate(manager, start=0.0, nominal=10.0, healthy=1.0, floor=0.01):
    return calculate_compute_completion(
        start_time_ms=start,
        nominal_service_time_ms=nominal,
        worker_id="P1",
        healthy_capacity=healthy,
        min_capacity=floor,
        fault_manager=manager,
    )


def test_no_fault_preserves_nominal_service_time():
    result = calculate(make_manager())

    assert result["completion_time_ms"] == pytest.approx(10.0)
    assert result["service_time_ms"] == pytest.approx(10.0)
    assert result["average_effective_capacity"] == pytest.approx(1.0)


def test_fault_covering_entire_execution_doubles_service_time():
    manager = make_manager(
        ComputeFault("F1", "P1", 0.0, 100.0, 0.5)
    )

    result = calculate(manager)

    assert result["completion_time_ms"] == pytest.approx(20.0)
    assert result["service_time_ms"] == pytest.approx(20.0)
    assert result["average_effective_capacity"] == pytest.approx(0.5)


def test_fault_starting_during_execution_slows_only_remaining_work():
    manager = make_manager(
        ComputeFault("F1", "P1", 5.0, 20.0, 0.5)
    )

    result = calculate(manager)

    # 5 units of work at rate 1, then 5 at rate 0.5.
    assert result["completion_time_ms"] == pytest.approx(15.0)
    assert result["service_time_ms"] == pytest.approx(15.0)


def test_fault_ending_during_execution_restores_healthy_rate():
    manager = make_manager(
        ComputeFault("F1", "P1", 0.0, 5.0, 0.5)
    )

    result = calculate(manager)

    # 2.5 units during the fault, then 7.5 at healthy rate.
    assert result["completion_time_ms"] == pytest.approx(12.5)


def test_overlapping_faults_combine_multiplicatively():
    manager = make_manager(
        ComputeFault("F1", "P1", 0.0, 10.0, 0.5),
        ComputeFault("F2", "P1", 5.0, 15.0, 0.5),
    )

    result = calculate(manager)

    # 2.5 units at 0.5, 1.25 at 0.25, 2.5 at 0.5,
    assert result["completion_time_ms"] == pytest.approx(18.75)


def test_minimum_capacity_floor_is_respected():
    manager = make_manager(
        ComputeFault("F1", "P1", 0.0, 100.0, 0.999)
    )

    result = calculate(manager, nominal=1.0, floor=0.01)

    assert result["completion_time_ms"] == pytest.approx(100.0)


def test_fault_on_another_worker_does_not_affect_execution():
    manager = make_manager(
        ComputeFault("F1", "P2", 0.0, 100.0, 0.9)
    )

    result = calculate(manager)

    assert result["completion_time_ms"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_time_ms": -1.0},
        {"nominal_service_time_ms": 0.0},
        {"healthy_capacity": 0.0},
        {"min_capacity": 0.0},
    ],
)
def test_invalid_parameters_are_rejected(overrides):
    arguments = {
        "start_time_ms": 0.0,
        "nominal_service_time_ms": 10.0,
        "worker_id": "P1",
        "healthy_capacity": 1.0,
        "min_capacity": 0.01,
        "fault_manager": make_manager(),
    }
    arguments.update(overrides)

    with pytest.raises(ValueError):
        calculate_compute_completion(**arguments)
