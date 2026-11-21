"""Time-aware compute fault definitions and capacity calculations."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ComputeFault:
    """A temporary compute degradation over [start_time_ms, end_time_ms)."""

    fault_id: str
    worker_id: str
    start_time_ms: float
    end_time_ms: float
    severity: float

    def __post_init__(self):
        if not isinstance(self.fault_id, str) or not self.fault_id.strip():
            raise ValueError("fault_id must be a non-empty string")
        if not isinstance(self.worker_id, str) or not self.worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")

        for name, value in (
            ("start_time_ms", self.start_time_ms),
            ("end_time_ms", self.end_time_ms),
            ("severity", self.severity),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")

        if self.start_time_ms < 0:
            raise ValueError("Fault start cannot be negative")
        if self.end_time_ms <= self.start_time_ms:
            raise ValueError("Fault end must be after fault start")
        if not 0.0 <= self.severity < 1.0:
            raise ValueError("severity must be in [0, 1)")

        object.__setattr__(self, "start_time_ms", float(self.start_time_ms))
        object.__setattr__(self, "end_time_ms", float(self.end_time_ms))
        object.__setattr__(self, "severity", float(self.severity))


class ComputeFaultManager:
    """Register faults and calculate worker capacity at arbitrary times.

    This manager does not mutate workers and does not maintain a simulation
    clock. This supports the simulator's non-chronological scheduling passes.
    """

    def __init__(self):
        self._faults = {}

    @property
    def faults(self):
        return tuple(self._faults.values())

    def register_fault(self, fault):
        if not isinstance(fault, ComputeFault):
            raise TypeError("fault must be a ComputeFault instance")
        if fault.fault_id in self._faults:
            raise ValueError(f"Duplicate fault ID: {fault.fault_id}")
        self._faults[fault.fault_id] = fault

    def get_active_faults(self, worker_id, time_ms):
        self._validate_query(worker_id, time_ms)

        return tuple(
            fault
            for fault in self._faults.values()
            if (
                fault.worker_id == worker_id
                and fault.start_time_ms <= time_ms < fault.end_time_ms
            )
        )

    def effective_capacity(self, worker_id, healthy_capacity, time_ms):
        """Return capacity after all active faults, without mutating a worker."""
        self._validate_query(worker_id, time_ms)

        if (
            isinstance(healthy_capacity, bool)
            or not isinstance(healthy_capacity, (int, float))
            or not math.isfinite(healthy_capacity)
            or healthy_capacity <= 0
        ):
            raise ValueError(
                "healthy_capacity must be a finite positive number"
            )

        capacity = float(healthy_capacity)

        for fault in self.get_active_faults(worker_id, time_ms):
            capacity *= 1.0 - fault.severity

        return capacity

    def to_ground_truth_records(self):
        """Return records compatible with the separate ground-truth schema."""
        return [
            {
                "fault_id": fault.fault_id,
                "fault_type": "compute",
                "component_type": "worker",
                "component_id": fault.worker_id,
                "start_time_ms": fault.start_time_ms,
                "end_time_ms": fault.end_time_ms,
                "severity": fault.severity,
            }
            for fault in self._faults.values()
        ]

    @staticmethod
    def _validate_query(worker_id, time_ms):
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")

        if (
            isinstance(time_ms, bool)
            or not isinstance(time_ms, (int, float))
            or not math.isfinite(time_ms)
            or time_ms < 0
        ):
            raise ValueError("time_ms must be a finite non-negative number")
