"""Time-aware synchronization and handoff fault definitions."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SynchronizationFault:
    """Additional handoff delay over [start_time_ms, end_time_ms)."""

    fault_id: str
    component_id: str
    start_time_ms: float
    end_time_ms: float
    severity: float
    max_additional_delay_ms: float

    def __post_init__(self):
        for name, value in (
            ("fault_id", self.fault_id),
            ("component_id", self.component_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        for name, value in (
            ("start_time_ms", self.start_time_ms),
            ("end_time_ms", self.end_time_ms),
            ("severity", self.severity),
            ("max_additional_delay_ms", self.max_additional_delay_ms),
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
        if self.max_additional_delay_ms < 0:
            raise ValueError("Maximum additional delay cannot be negative")

        object.__setattr__(self, "start_time_ms", float(self.start_time_ms))
        object.__setattr__(self, "end_time_ms", float(self.end_time_ms))
        object.__setattr__(self, "severity", float(self.severity))
        object.__setattr__(
            self,
            "max_additional_delay_ms",
            float(self.max_additional_delay_ms),
        )


class SynchronizationFaultManager:
    """Manage time-dependent handoff delays for decode workers."""

    def __init__(self):
        self._faults = {}

    @property
    def faults(self):
        return tuple(self._faults.values())

    def register_fault(self, fault):
        if not isinstance(fault, SynchronizationFault):
            raise TypeError("fault must be a SynchronizationFault instance")
        if fault.fault_id in self._faults:
            raise ValueError(f"Duplicate fault ID: {fault.fault_id}")
        self._faults[fault.fault_id] = fault

    def get_active_faults(self, component_id, time_ms):
        if not isinstance(component_id, str) or not component_id.strip():
            raise ValueError("component_id must be a non-empty string")
        if (
            isinstance(time_ms, bool)
            or not isinstance(time_ms, (int, float))
            or not math.isfinite(time_ms)
            or time_ms < 0
        ):
            raise ValueError("time_ms must be finite and non-negative")

        return tuple(
            fault
            for fault in self._faults.values()
            if (
                fault.component_id == component_id
                and fault.start_time_ms <= time_ms < fault.end_time_ms
            )
        )

    def additional_delay_ms(self, component_id, time_ms):
        """Combine simultaneous delays additively."""
        return sum(
            fault.severity * fault.max_additional_delay_ms
            for fault in self.get_active_faults(component_id, time_ms)
        )

    def to_ground_truth_records(self):
        return [
            {
                "fault_id": fault.fault_id,
                "fault_type": "synchronization",
                "component_type": "worker",
                "component_id": fault.component_id,
                "start_time_ms": fault.start_time_ms,
                "end_time_ms": fault.end_time_ms,
                "severity": fault.severity,
            }
            for fault in self._faults.values()
        ]
