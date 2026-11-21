from pd_failslow.simulator.workers import Worker
from pd_failslow.simulator.communication import CommunicationLink
from pd_failslow.simulator.scheduler import Scheduler
from pd_failslow.simulator.synchronization import (
    SynchronizationModel,
    SynchronizationConfig,
)


class PDSystem:
    """
    Represents the complete PD-disaggregated inference infrastructure.
    """

    def __init__(
        self,
        prefill_config,
        decode_config,
        communication_config,
        scheduler_config=None,
        synchronization_config=None,
    ):
        scheduler_config = scheduler_config or {}

        # -------------------------------------------------
        # 1. Create prefill workers
        # -------------------------------------------------

        self.prefill_workers = [
            Worker(
                worker_id=item["id"],
                worker_type="prefill",
                compute_capacity=item["compute_capacity"],
                memory_capacity=item["memory_capacity"],
            )
            for item in prefill_config["workers"]
        ]

        # -------------------------------------------------
        # 2. Create decode workers
        # -------------------------------------------------

        self.decode_workers = [
            Worker(
                worker_id=item["id"],
                worker_type="decode",
                compute_capacity=item["compute_capacity"],
                memory_capacity=item["memory_capacity"],
            )
            for item in decode_config["workers"]
        ]

        # -------------------------------------------------
        # 3. Create communication links
        # -------------------------------------------------

        self.communication_links = [
            CommunicationLink(
                link_id=item["id"],
                link_type=item["link_type"],
                bandwidth_mb_per_ms=item["bandwidth_mb_per_ms"],
                fixed_latency_ms=item.get(
                    "fixed_latency_ms", 0.0
                ),
                shared=item.get("shared", True),
            )
            for item in communication_config["links"]
        ]

        # -------------------------------------------------
        # 4. Configure communication routes
        # -------------------------------------------------

        link_routes = scheduler_config.get(
            "link_routes", {}
        )

        normalized_routes = {}

        for key, link_id in link_routes.items():
            if isinstance(key, str):
                prefill_id, decode_id = key.split("->")
                normalized_routes[
                    (prefill_id.strip(), decode_id.strip())
                ] = link_id
            else:
                normalized_routes[key] = link_id

        # -------------------------------------------------
        # 5. Create scheduler
        # -------------------------------------------------

        self.scheduler = Scheduler(
            prefill_workers=self.prefill_workers,
            decode_workers=self.decode_workers,
            communication_links=self.communication_links,
            prefill_policy=scheduler_config.get(
                "prefill_policy", "shortest_queue"
            ),
            decode_policy=scheduler_config.get(
                "decode_policy", "shortest_queue"
            ),
            link_routes=normalized_routes,
        )

        # -------------------------------------------------
        # 6. Create synchronization model
        # -------------------------------------------------

        sync_config = SynchronizationConfig(
            normal_handoff_delay_ms=(
                synchronization_config or {}
            ).get("normal_handoff_delay_ms", 0.1)
        )

        self.synchronization = SynchronizationModel(
            config=sync_config
        )

    def reset(self):
        """
        Restore the system to a clean initial state.
        """
        for worker in self.prefill_workers:
            worker.reset()

        for worker in self.decode_workers:
            worker.reset()

        for link in self.communication_links:
            link.reset()

        self.synchronization.injected_handoff_delays.clear()

    def summary(self):
        """
        Return a summary of the simulated infrastructure.
        """
        return {
            "prefill_workers": [
                worker.worker_id
                for worker in self.prefill_workers
            ],
            "decode_workers": [
                worker.worker_id
                for worker in self.decode_workers
            ],
            "communication_links": [
                {
                    "id": link.link_id,
                    "type": link.link_type,
                    "bandwidth_mb_per_ms": (
                        link.bandwidth_mb_per_ms
                    ),
                }
                for link in self.communication_links
            ],
        }