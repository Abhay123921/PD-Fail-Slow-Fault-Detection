"""Central orchestration for compute, communication and synchronization faults."""

from pd_failslow.faults.compute_faults import (
    ComputeFault,
    ComputeFaultManager,
)
from pd_failslow.faults.communication_faults import (
    CommunicationFault,
    CommunicationFaultManager,
)
from pd_failslow.faults.synchronization_faults import (
    SynchronizationFault,
    SynchronizationFaultManager,
)


class FaultInjector:
    """Coordinate fault registration, preparation and effect queries.

    Compute and synchronization effects are queried at the relevant
    simulation timestamp. Communication faults are scheduled on links
    during prepare_run(), after system reset and before transfers begin.
    """

    def __init__(self):
        self.compute = ComputeFaultManager()
        self.communication = CommunicationFaultManager()
        self.synchronization = SynchronizationFaultManager()
        self._prepared_system = None

    def register_compute_fault(self, fault):
        self.compute.register_fault(fault)
        self._prepared_system = None

    def register_communication_fault(self, fault):
        self.communication.register_fault(fault)
        self._prepared_system = None

    def register_synchronization_fault(self, fault):
        self.synchronization.register_fault(fault)
        self._prepared_system = None

    def prepare_run(self, system):
        """Validate fault targets and schedule communication faults.

        Call after system.reset() and before the simulation starts.
        Repeated preparation on the same system is safe when the
        existing link schedules match the registered faults.
        """
        workers = {
            worker.worker_id: worker
            for worker in (
                list(system.prefill_workers)
                + list(system.decode_workers)
            )
        }

        links = {
            link.link_id: link
            for link in system.communication_links
        }

        for fault in self.compute.faults:
            if fault.worker_id not in workers:
                raise ValueError(
                    f"Unknown compute-fault worker: {fault.worker_id}"
                )

        for fault in self.communication.faults:
            if fault.link_id not in links:
                raise ValueError(
                    f"Unknown communication-fault link: {fault.link_id}"
                )

        decode_worker_ids = {
            worker.worker_id for worker in system.decode_workers
        }

        for fault in self.synchronization.faults:
            if fault.component_id not in decode_worker_ids:
                raise ValueError(
                    "Synchronization fault must target a decode worker: "
                    f"{fault.component_id}"
                )

        for link in system.communication_links:
            for fault in self.communication.get_faults_for_link(
                link.link_id
            ):
                existing = link.bandwidth_faults.get(fault.fault_id)

                expected = {
                    "start_time_ms": fault.start_time_ms,
                    "end_time_ms": fault.end_time_ms,
                    "severity": fault.severity,
                }

                if existing is None:
                    link.schedule_bandwidth_fault(
                        fault_id=fault.fault_id,
                        start_time_ms=fault.start_time_ms,
                        end_time_ms=fault.end_time_ms,
                        severity=fault.severity,
                    )
                elif existing != expected:
                    raise ValueError(
                        f"Conflicting existing schedule for "
                        f"communication fault {fault.fault_id}"
                    )

        self._prepared_system = system

    def effective_compute_capacity(
        self,
        worker_id,
        healthy_capacity,
        time_ms,
    ):
        return self.compute.effective_capacity(
            worker_id,
            healthy_capacity,
            time_ms,
        )

    def additional_synchronization_delay(
        self,
        decode_worker_id,
        transfer_completion_time_ms,
    ):
        return self.synchronization.additional_delay_ms(
            decode_worker_id,
            transfer_completion_time_ms,
        )

    def to_ground_truth_records(self):
        return (
            self.compute.to_ground_truth_records()
            + self.communication.to_ground_truth_records()
            + self.synchronization.to_ground_truth_records()
        )

    def reset(self):
        """Clear registered faults and injector preparation state."""
        self.compute = ComputeFaultManager()
        self.communication = CommunicationFaultManager()
        self.synchronization = SynchronizationFaultManager()
        self._prepared_system = None
