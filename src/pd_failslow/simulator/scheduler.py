class Scheduler:
    """
    Basic scheduler for a PD-disaggregated inference simulator.

    Policies:
        prefill: shortest_queue
        decode: shortest_queue
        link: configured_route
    """

    def __init__(
        self,
        prefill_workers: list,
        decode_workers: list,
        communication_links: list,
        prefill_policy: str = "shortest_queue",
        decode_policy: str = "shortest_queue",
        link_routes: dict | None = None,
    ):
        if not prefill_workers:
            raise ValueError("At least one prefill worker is required")

        if not decode_workers:
            raise ValueError("At least one decode worker is required")

        if not communication_links:
            raise ValueError("At least one communication link is required")

        self.prefill_workers = {
            worker.worker_id: worker
            for worker in prefill_workers
        }

        self.decode_workers = {
            worker.worker_id: worker
            for worker in decode_workers
        }

        self.communication_links = {
            link.link_id: link
            for link in communication_links
        }

        self.prefill_policy = prefill_policy
        self.decode_policy = decode_policy

        self.link_routes = link_routes or {}

        if len(self.prefill_workers) != len(prefill_workers):
            raise ValueError("Duplicate prefill worker IDs detected")

        if len(self.decode_workers) != len(decode_workers):
            raise ValueError("Duplicate decode worker IDs detected")

        if len(self.communication_links) != len(communication_links):
            raise ValueError("Duplicate communication link IDs detected")

    @staticmethod
    def _queue_length(worker) -> int:
        """
        Return the worker's queue length.

        The Worker class should expose a queue attribute.
        """
        return len(worker.queue)

    @staticmethod
    def _busy_until(worker) -> float:
        """
        Return the worker's current busy-until timestamp.
        """
        return float(getattr(worker, "busy_until_ms", 0.0))

    def select_prefill_worker(self):
        """
        Select a prefill worker.
        """
        if self.prefill_policy != "shortest_queue":
            raise NotImplementedError(
                f"Unsupported prefill policy: {self.prefill_policy}"
            )

        return min(
            self.prefill_workers.values(),
            key=lambda worker: (
                self._queue_length(worker),
                self._busy_until(worker),
                worker.worker_id,
            ),
        )

    def select_decode_worker(self):
        """
        Select a decode worker.
        """
        if self.decode_policy != "shortest_queue":
            raise NotImplementedError(
                f"Unsupported decode policy: {self.decode_policy}"
            )

        return min(
            self.decode_workers.values(),
            key=lambda worker: (
                self._queue_length(worker),
                self._busy_until(worker),
                worker.worker_id,
            ),
        )

    def select_link(
        self,
        prefill_worker_id: str,
        decode_worker_id: str,
    ):
        """
        Select the configured communication route.

        link_routes example:
        {
            ("P1", "D1"): "L1",
            ("P1", "D2"): "L2",
            ("P2", "D1"): "L2",
            ("P2", "D2"): "L1",
        }

        If no route is configured for a pair, select the link with
        the highest configured bandwidth.
        """
        route_key = (
            prefill_worker_id,
            decode_worker_id,
        )

        if route_key in self.link_routes:
            link_id = self.link_routes[route_key]

            if link_id not in self.communication_links:
                raise ValueError(
                    f"Configured link {link_id} does not exist"
                )

            return self.communication_links[link_id]

        return max(
            self.communication_links.values(),
            key=lambda link: (
                link.bandwidth_mb_per_ms,
                link.link_id,
            ),
        )

    def get_prefill_worker(self, worker_id: str):
        return self.prefill_workers[worker_id]

    def get_decode_worker(self, worker_id: str):
        return self.decode_workers[worker_id]

    def get_link(self, link_id: str):
        return self.communication_links[link_id]

    def get_system_state(self) -> dict:
        """
        Return a lightweight scheduler state snapshot.
        """
        return {
            "prefill_workers": {
                worker_id: {
                    "queue_length": self._queue_length(worker),
                    "busy_until_ms": self._busy_until(worker),
                }
                for worker_id, worker in self.prefill_workers.items()
            },
            "decode_workers": {
                worker_id: {
                    "queue_length": self._queue_length(worker),
                    "busy_until_ms": self._busy_until(worker),
                }
                for worker_id, worker in self.decode_workers.items()
            },
            "communication_links": {
                link_id: {
                    "bandwidth_mb_per_ms": link.bandwidth_mb_per_ms,
                    "active_transfers": len(link.active_transfers),
                }
                for link_id, link in self.communication_links.items()
            },
        }