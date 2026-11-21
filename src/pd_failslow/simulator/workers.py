class Worker:
    """
    Represents a prefill or decode worker.

    compute_capacity:
        Configured healthy compute capacity.

    effective_compute_capacity():
        Current capacity after any injected degradation.

    busy_until_ms:
        Earliest time at which the worker becomes available.
    """

    def __init__(
        self,
        worker_id,
        worker_type,
        compute_capacity,
        memory_capacity,
    ):
        if compute_capacity <= 0:
            raise ValueError("compute_capacity must be positive")

        if memory_capacity <= 0:
            raise ValueError("memory_capacity must be positive")

        if worker_type not in ("prefill", "decode"):
            raise ValueError(
                "worker_type must be 'prefill' or 'decode'"
            )

        self.worker_id = worker_id
        self.worker_type = worker_type

        self.compute_capacity = float(compute_capacity)
        self.memory_capacity = float(memory_capacity)

        self.current_compute_capacity = float(compute_capacity)

        self.queue = []
        self.busy_until_ms = 0.0

        self.mapping_id = "default"

    def get_queue_length(self):
        return len(self.queue)

    def effective_compute_capacity(self):
        return float(self.current_compute_capacity)

    def set_effective_compute_capacity(self, capacity):
        if capacity <= 0:
            raise ValueError("Effective capacity must be positive")

        self.current_compute_capacity = float(capacity)

    def restore_healthy_capacity(self):
        self.current_compute_capacity = self.compute_capacity

    def add_to_queue(self, request):
        self.queue.append(request)

    def remove_from_queue(self, request):
        if request in self.queue:
            self.queue.remove(request)

    def is_available(self, current_time_ms):
        return self.busy_until_ms <= current_time_ms

    def reserve(self, completion_time_ms):
        self.busy_until_ms = max(
            self.busy_until_ms,
            float(completion_time_ms),
        )

    def reset(self):
        self.queue.clear()
        self.busy_until_ms = 0.0
        self.restore_healthy_capacity()