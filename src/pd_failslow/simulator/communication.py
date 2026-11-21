from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransferRecord:
    request_id: str
    data_size_mb: float
    start_time_ms: float
    completion_time_ms: float
    effective_bandwidth_mb_per_ms: float
    data_start_time_ms: float = 0.0
    remaining_data_mb: float = 0.0
    concurrent_transfers: int = 1
    concurrency_time_ms: float = 0.0
    average_concurrent_transfers: float = 0.0
    peak_concurrent_transfers: int = 0


@dataclass
class CommunicationLink:
    """
    Abstract communication link for KV-cache transfers.

    Shared links use equal bandwidth sharing between active transfers.
    Completion estimates are updated as transfers arrive and finish.
    """

    link_id: str
    link_type: str
    bandwidth_mb_per_ms: float
    fixed_latency_ms: float = 0.0
    shared: bool = True

    active_transfers: list[TransferRecord] = field(
        default_factory=list
    )

    transfer_history: dict[str, TransferRecord] = field(
        default_factory=dict,
        init=False,
    )

    bandwidth_faults: dict[str, dict[str, float]] = field(
        default_factory=dict,
        init=False,
    )

    _healthy_bandwidth_mb_per_ms: float = field(
        default=0.0,
        init=False,
        repr=False,
    )
    _legacy_bandwidth_factor: float = field(
        default=1.0,
        init=False,
        repr=False,
    )

    _clock_ms: Optional[float] = field(
        default=None,
        init=False,
        repr=False,
    )

    EPSILON = 1e-9

    def __post_init__(self):
        if self.bandwidth_mb_per_ms <= 0:
            raise ValueError("Link bandwidth must be positive")

        if self.fixed_latency_ms < 0:
            raise ValueError("Fixed latency cannot be negative")

        self._healthy_bandwidth_mb_per_ms = float(
            self.bandwidth_mb_per_ms
        )

    def schedule_bandwidth_fault(
        self,
        fault_id: str,
        start_time_ms: float,
        end_time_ms: float,
        severity: float,
    ) -> None:
        """Register a temporary bandwidth degradation."""

        if not fault_id:
            raise ValueError("fault_id cannot be empty")
        if fault_id in self.bandwidth_faults:
            raise ValueError(f"Duplicate fault ID: {fault_id}")
        if start_time_ms < 0:
            raise ValueError("Fault start cannot be negative")
        if end_time_ms <= start_time_ms:
            raise ValueError("Fault end must be after fault start")
        if not 0.0 <= severity < 1.0:
            raise ValueError("severity must be in [0, 1)")
        if (
            self._clock_ms is not None
            and start_time_ms < self._clock_ms - self.EPSILON
        ):
            raise ValueError("Cannot schedule a fault in the past")

        self.bandwidth_faults[fault_id] = {
            "start_time_ms": float(start_time_ms),
            "end_time_ms": float(end_time_ms),
            "severity": float(severity),
        }

        if self._clock_ms is not None:
            self._recalculate_future_completions(self._clock_ms)

    def _bandwidth_at(self, time_ms: float) -> float:
        """Healthy bandwidth multiplied by all active fault factors."""

        bandwidth = (
            self._healthy_bandwidth_mb_per_ms
            * self._legacy_bandwidth_factor
        )

        for fault in self.bandwidth_faults.values():
            if (
                fault["start_time_ms"] <= time_ms + self.EPSILON
                and time_ms < fault["end_time_ms"] - self.EPSILON
            ):
                bandwidth *= 1.0 - fault["severity"]

        return float(bandwidth)

    def _next_bandwidth_event(self, time_ms: float) -> float:
        """Return the next future fault start or end timestamp."""

        events = []

        for fault in self.bandwidth_faults.values():
            events.append(fault["start_time_ms"])
            events.append(fault["end_time_ms"])

        return min(
            (
                event for event in events
                if event > time_ms + self.EPSILON
            ),
            default=float("inf"),
        )

    def get_active_transfers(self, current_time_ms: float):
        return [
            transfer
            for transfer in self.active_transfers
            if transfer.completion_time_ms > current_time_ms
        ]

    def effective_bandwidth(
        self,
        concurrent_transfers: Optional[int] = None,
        current_time_ms: Optional[float] = None,
    ) -> float:

        if concurrent_transfers is None:
            if current_time_ms is None:
                concurrent_transfers = 1
            else:
                concurrent_transfers = (
                    len(self.get_active_transfers(current_time_ms)) + 1
                )

        if concurrent_transfers < 1:
            raise ValueError(
                "concurrent_transfers must be at least 1"
            )

        if not self.shared:
            return float(self.bandwidth_mb_per_ms)

        return float(
            self.bandwidth_mb_per_ms / concurrent_transfers
        )

    def estimate_transfer_time(
        self,
        data_size_mb: float,
        concurrent_transfers: int = 1,
    ) -> float:

        if data_size_mb < 0:
            raise ValueError("data_size_mb cannot be negative")

        bandwidth = self.effective_bandwidth(
            concurrent_transfers=concurrent_transfers
        )

        return float(
            self.fixed_latency_ms + data_size_mb / bandwidth
        )

    def _advance_to(self, target_time_ms: float):
        """Advance actual transfer progress through chronological events."""

        if self._clock_ms is None:
            self._clock_ms = target_time_ms
            return

        if target_time_ms < self._clock_ms:
            raise ValueError(
                "Transfer start times must be nondecreasing"
            )

        now = self._clock_ms

        while now < target_time_ms - self.EPSILON:
            transmitting = [
                t for t in self.active_transfers
                if t.data_start_time_ms <= now + self.EPSILON
                and t.remaining_data_mb > self.EPSILON
            ]

            pending = [
                t for t in self.active_transfers
                if t.data_start_time_ms > now + self.EPSILON
            ]

            next_fault = self._next_bandwidth_event(now)

            if not transmitting:
                next_start = min(
                    (t.data_start_time_ms for t in pending),
                    default=float("inf"),
                )
                next_event = min(
                    target_time_ms,
                    next_start,
                    next_fault,
                )
                if next_event <= now + self.EPSILON:
                    break
                now = next_event
                continue

            count = len(transmitting)
            bandwidth = self._bandwidth_at(now)
            rate = (
                bandwidth / count if self.shared else bandwidth
            )

            next_completion = min(
                now + t.remaining_data_mb / rate
                for t in transmitting
            )
            next_arrival = min(
                (t.data_start_time_ms for t in pending),
                default=float("inf"),
            )

            next_event = min(
                target_time_ms,
                next_completion,
                next_arrival,
                next_fault,
            )

            elapsed = next_event - now

            for transfer in transmitting:
                transfer.concurrency_time_ms += count * elapsed
                transfer.peak_concurrent_transfers = max(
                    transfer.peak_concurrent_transfers, count
                )
                transfer.remaining_data_mb = max(
                    0.0,
                    transfer.remaining_data_mb - rate * elapsed,
                )

            now = next_event

            completed = [
                t for t in transmitting
                if t.remaining_data_mb <= self.EPSILON
            ]

            for transfer in completed:
                transfer.remaining_data_mb = 0.0
                transfer.completion_time_ms = now

                duration = now - transfer.data_start_time_ms
                if duration > self.EPSILON:
                    transfer.average_concurrent_transfers = (
                        transfer.concurrency_time_ms / duration
                    )
                    transfer.effective_bandwidth_mb_per_ms = (
                        transfer.data_size_mb / duration
                    )

                self.active_transfers.remove(transfer)

        self._clock_ms = target_time_ms

    def _recalculate_future_completions(
        self,
        current_time_ms: float,
    ):
        """Predict completions across transfer and fault events."""

        remaining = {
            t.request_id: t.remaining_data_mb
            for t in self.active_transfers
        }

        pending = list(self.active_transfers)
        transmitting = []
        concurrency_area = {
            t.request_id: t.concurrency_time_ms
            for t in self.active_transfers
        }
        peak_concurrency = {
            t.request_id: t.peak_concurrent_transfers
            for t in self.active_transfers
        }
        now = current_time_ms

        while pending or transmitting:
            newly_started = [
                t for t in pending
                if t.data_start_time_ms <= now + self.EPSILON
            ]

            if newly_started:
                for transfer in newly_started:
                    pending.remove(transfer)
                    transmitting.append(transfer)

                count = len(transmitting)
                for transfer in transmitting:
                    peak_concurrency[transfer.request_id] = max(
                        peak_concurrency[transfer.request_id], count
                    )
                for transfer in newly_started:
                    transfer.concurrent_transfers = count

            next_fault = self._next_bandwidth_event(now)

            if not transmitting:
                next_start = min(
                    (t.data_start_time_ms for t in pending),
                    default=float("inf"),
                )
                next_event = min(next_start, next_fault)

                if next_event == float("inf"):
                    break

                now = next_event
                continue

            count = len(transmitting)
            bandwidth = self._bandwidth_at(now)
            rate = (
                bandwidth / count if self.shared else bandwidth
            )

            next_completion = min(
                now + remaining[t.request_id] / rate
                for t in transmitting
            )
            next_arrival = min(
                (t.data_start_time_ms for t in pending),
                default=float("inf"),
            )

            next_event = min(
                next_completion,
                next_arrival,
                next_fault,
            )

            elapsed = next_event - now

            for transfer in transmitting:
                concurrency_area[transfer.request_id] += count * elapsed
                peak_concurrency[transfer.request_id] = max(
                    peak_concurrency[transfer.request_id], count
                )
                remaining[transfer.request_id] = max(
                    0.0,
                    remaining[transfer.request_id] - rate * elapsed,
                )

            now = next_event

            completed = [
                t for t in transmitting
                if remaining[t.request_id] <= self.EPSILON
            ]

            for transfer in completed:
                transfer.completion_time_ms = now

                duration = now - transfer.data_start_time_ms
                if duration > self.EPSILON:
                    transfer.average_concurrent_transfers = (
                        concurrency_area[transfer.request_id] / duration
                    )
                    transfer.peak_concurrent_transfers = (
                        peak_concurrency[transfer.request_id]
                    )
                    transfer.effective_bandwidth_mb_per_ms = (
                        transfer.data_size_mb / duration
                    )

                transmitting.remove(transfer)

    def _build_result(self, transfer: TransferRecord) -> dict:

        return {
            "request_id": transfer.request_id,
            "link_id": self.link_id,
            "link_type": self.link_type,
            "data_size_mb": transfer.data_size_mb,
            "start_time_ms": transfer.start_time_ms,
            "completion_time_ms": transfer.completion_time_ms,
            "transfer_time_ms": (
                transfer.completion_time_ms
                - transfer.start_time_ms
            ),
            "configured_bandwidth_mb_per_ms": (
                self._healthy_bandwidth_mb_per_ms
            ),
            "effective_bandwidth_mb_per_ms": (
                transfer.effective_bandwidth_mb_per_ms
            ),
            "concurrent_transfers": (
                transfer.concurrent_transfers
            ),
            "average_concurrent_transfers": (
                transfer.average_concurrent_transfers
            ),
            "peak_concurrent_transfers": (
                transfer.peak_concurrent_transfers
            ),
        }

    def get_transfer_result(self, request_id: str) -> dict:
        """Return the latest completion estimate for a transfer."""

        if request_id not in self.transfer_history:
            raise KeyError(
                f"Unknown transfer: {request_id}"
            )

        return self._build_result(
            self.transfer_history[request_id]
        )

    def start_transfer(
        self,
        request,
        current_time_ms: float,
        data_size_mb: float,
        concurrent_transfers: Optional[int] = None,
    ) -> dict:

        if current_time_ms < 0:
            raise ValueError(
                "current_time_ms cannot be negative"
            )

        if data_size_mb < 0:
            raise ValueError(
                "data_size_mb cannot be negative"
            )

        if concurrent_transfers is not None:
            if concurrent_transfers < 1:
                raise ValueError(
                    "concurrent_transfers must be at least 1"
                )

            raise ValueError(
                "Dynamic sharing derives concurrency from "
                "registered transfers. Start each transfer "
                "individually instead."
            )

        request_id = request.request_id

        if request_id in self.transfer_history:
            raise ValueError(
                f"Duplicate transfer request ID: {request_id}"
            )

        self._advance_to(current_time_ms)

        data_start_time_ms = (
            current_time_ms + self.fixed_latency_ms
        )

        record = TransferRecord(
            request_id=request_id,
            data_size_mb=float(data_size_mb),
            start_time_ms=float(current_time_ms),
            completion_time_ms=float(data_start_time_ms),
            effective_bandwidth_mb_per_ms=float(
                self._bandwidth_at(data_start_time_ms)
            ),
            data_start_time_ms=float(data_start_time_ms),
            remaining_data_mb=float(data_size_mb),
        )

        self.transfer_history[request_id] = record

        if data_size_mb > 0:
            self.active_transfers.append(record)
        else:
            record.remaining_data_mb = 0.0

        self._recalculate_future_completions(
            current_time_ms
        )

        return self._build_result(record)

    def complete_transfer(
        self,
        request_id: str,
        completion_time_ms: float,
    ) -> None:

        self._advance_to(completion_time_ms)

        self.active_transfers = [
            t for t in self.active_transfers
            if t.request_id != request_id
        ]

        self._recalculate_future_completions(
            completion_time_ms
        )

    def reset(self):
        """Reset communication state between experiments."""
        self.active_transfers.clear()
        self.transfer_history.clear()
        self.bandwidth_faults.clear()
        self._legacy_bandwidth_factor = 1.0
        self.bandwidth_mb_per_ms = (
            self._healthy_bandwidth_mb_per_ms
        )
        self._clock_ms = None

    def apply_bandwidth_reduction(self, severity: float) -> None:
        """Apply a persistent legacy reduction until reset."""
        if not 0.0 <= severity < 1.0:
            raise ValueError("severity must be in [0, 1)")

        self._legacy_bandwidth_factor *= (1.0 - severity)

        if self._clock_ms is not None:
            self._recalculate_future_completions(self._clock_ms)