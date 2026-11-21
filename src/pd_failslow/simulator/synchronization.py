from dataclasses import dataclass


@dataclass
class SynchronizationConfig:
    normal_handoff_delay_ms: float = 0.1


class SynchronizationModel:
    """
    Models handoff and decode admission timing.

    This module does not simulate the communication transfer itself.
    It receives the transfer completion timestamp from communication.py.
    """

    def __init__(
        self,
        config: SynchronizationConfig | None = None,
    ):
        self.config = config or SynchronizationConfig()

        self.injected_handoff_delays: dict[str, float] = {}

    def set_handoff_delay(
        self,
        component_id: str,
        delay_ms: float,
    ) -> None:
        """
        Configure an additional handoff delay for a component.

        Example:
            set_handoff_delay("D1", 5.0)
        """
        if delay_ms < 0:
            raise ValueError("delay_ms cannot be negative")

        self.injected_handoff_delays[component_id] = float(delay_ms)

    def remove_handoff_delay(
        self,
        component_id: str,
    ) -> None:
        self.injected_handoff_delays.pop(component_id, None)

    def calculate_handoff(
        self,
        request,
        transfer_completion_time_ms: float,
        decode_worker_id: str,
        additional_delay_ms: float = 0.0,
    ) -> dict:
        """
        Calculate decode admission time.

        Handoff wait =
            decode admission - transfer completion

        The injected delay is associated with the selected decode
        worker in this initial abstraction.
        """
        if transfer_completion_time_ms < 0:
            raise ValueError(
                "transfer_completion_time_ms cannot be negative"
            )

        if additional_delay_ms < 0:
            raise ValueError("additional_delay_ms cannot be negative")

        injected_delay = self.injected_handoff_delays.get(
            decode_worker_id,
            0.0
        )

        fault_delay = float(additional_delay_ms)

        handoff_wait_ms = (
            self.config.normal_handoff_delay_ms
            + injected_delay
            + fault_delay
        )

        decode_admission_time_ms = (
            transfer_completion_time_ms + handoff_wait_ms
        )

        return {
            "request_id": request.request_id,
            "decode_worker_id": decode_worker_id,
            "transfer_completion_time_ms": float(
                transfer_completion_time_ms
            ),
            "decode_admission_time_ms": float(
                decode_admission_time_ms
            ),
            "normal_handoff_delay_ms": float(
                self.config.normal_handoff_delay_ms
            ),
            "injected_handoff_delay_ms": float(injected_delay),
            "fault_handoff_delay_ms": fault_delay,
            "handoff_wait_ms": float(handoff_wait_ms),
        }

    @staticmethod
    def calculate_admission_wait(
        decode_admission_time_ms: float,
        decode_execution_start_time_ms: float,
    ) -> float:
        """
        Calculate queueing after decode admission.

        This is separate from handoff wait.
        """
        wait_ms = (
            decode_execution_start_time_ms
            - decode_admission_time_ms
        )

        if wait_ms < -1e-9:
            raise ValueError(
                "Decode execution cannot start before decode admission"
            )

        return float(max(0.0, wait_ms))