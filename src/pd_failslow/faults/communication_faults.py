"""Time-aware communication-link fault definitions and management."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CommunicationFault:
    """Bandwidth degradation over the half-open interval [start, end)."""

    fault_id: str
    link_id: str
    start_time_ms: float
    end_time_ms: float
    severity: float

    def __post_init__(self):
        if not isinstance(self.fault_id, str) or not self.fault_id.strip():
            raise ValueError("fault_id must be a non-empty string")
        if not isinstance(self.link_id, str) or not self.link_id.strip():
            raise ValueError("link_id must be a non-empty string")

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


class CommunicationFaultManager:
    """Register link faults and apply them through the existing link API."""

    def __init__(self):
        self._faults = {}

    @property
    def faults(self):
        return tuple(self._faults.values())

    def register_fault(self, fault):
        if not isinstance(fault, CommunicationFault):
            raise TypeError("fault must be a CommunicationFault instance")
        if fault.fault_id in self._faults:
            raise ValueError(f"Duplicate fault ID: {fault.fault_id}")
        self._faults[fault.fault_id] = fault

    def get_faults_for_link(self, link_id):
        if not isinstance(link_id, str) or not link_id.strip():
            raise ValueError("link_id must be a non-empty string")
        return tuple(
            fault for fault in self._faults.values()
            if fault.link_id == link_id
        )

    def apply_to_link(self, link):
        """Schedule all registered faults for this link exactly once per call.

        Call this after the link has been reset and before transfers begin.
        The link itself validates scheduling constraints and duplicate IDs.
        """
        link_id = getattr(link, "link_id", None)
        if not isinstance(link_id, str) or not link_id.strip():
            raise TypeError("link must expose a non-empty link_id")

        for fault in self.get_faults_for_link(link_id):
            link.schedule_bandwidth_fault(
                fault_id=fault.fault_id,
                start_time_ms=fault.start_time_ms,
                end_time_ms=fault.end_time_ms,
                severity=fault.severity,
            )

    def to_ground_truth_records(self):
        return [
            {
                "fault_id": fault.fault_id,
                "fault_type": "communication",
                "component_type": "link",
                "component_id": fault.link_id,
                "start_time_ms": fault.start_time_ms,
                "end_time_ms": fault.end_time_ms,
                "severity": fault.severity,
            }
            for fault in self._faults.values()
        ]
