from pd_failslow.simulator.prefill import (
    execute_prefill,
    estimate_prefill_time,
    PrefillConfig,
)

from pd_failslow.simulator.decode import (
    execute_decode,
    estimate_decode_time,
    DecodeConfig,
)

from pd_failslow.telemetry.collector import TelemetryCollector
from pd_failslow.simulator.compute_execution import calculate_compute_completion


class PDSimulationEngine:
    """
    Initial two-pass PD inference simulation engine.

    Pass 1:
        Schedule prefill execution for all requests.

    Pass 2:
        Process requests in prefill-completion order:
        KV transfer -> handoff -> decode admission -> decode execution.

    This is an initial deterministic simulator, not yet a fully
    dynamic discrete-event network simulator.
    """

    def __init__(
        self,
        system,
        collector=None,
        prefill_config=None,
        decode_config=None,
        kv_cache_mb_per_prompt_token=0.01,
        prefill_mapping_efficiency=1.0,
        decode_mapping_efficiency=1.0,
        memory_interference=0.0,
        compute_interference=0.0,
        fault_injector=None,
        worker_pair_assignments=None,
    ):
        self.system = system

        self.collector = (
            collector
            if collector is not None
            else TelemetryCollector()
        )

        self.prefill_config = (
            prefill_config
            if prefill_config is not None
            else PrefillConfig()
        )

        self.decode_config = (
            decode_config
            if decode_config is not None
            else DecodeConfig()
        )

        if kv_cache_mb_per_prompt_token <= 0:
            raise ValueError(
                "kv_cache_mb_per_prompt_token must be positive"
            )

        self.kv_cache_mb_per_prompt_token = float(
            kv_cache_mb_per_prompt_token
        )

        self.prefill_mapping_efficiency = (
            prefill_mapping_efficiency
        )

        self.decode_mapping_efficiency = (
            decode_mapping_efficiency
        )

        self.memory_interference = memory_interference
        self.compute_interference = compute_interference

        self.fault_injector = fault_injector
        self.worker_pair_assignments = None

        if worker_pair_assignments is not None:
            if not isinstance(worker_pair_assignments, dict):
                raise TypeError("worker_pair_assignments must be a dictionary")

            self.worker_pair_assignments = {}
            for request_id, pair in worker_pair_assignments.items():
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError(
                        f"Invalid worker-pair assignment for {request_id}"
                    )

                prefill_id, decode_id = pair

                if prefill_id not in self.system.scheduler.prefill_workers:
                    raise ValueError(f"Unknown prefill worker: {prefill_id}")

                if decode_id not in self.system.scheduler.decode_workers:
                    raise ValueError(f"Unknown decode worker: {decode_id}")

                self.worker_pair_assignments[request_id] = (
                    prefill_id, decode_id
                )

        self._recorded_fault_ids = set()

        self.prefill_records = []

    def _execute_prefill_with_capacity_profile(self, request, worker, start_time_ms):
        """Execute prefill using capacity integrated across fault boundaries."""

        capacity_at_start = worker.compute_capacity

        if self.fault_injector is None:
            result = execute_prefill(
                request=request,
                worker=worker,
                start_time_ms=start_time_ms,
                mapping_efficiency=self.prefill_mapping_efficiency,
                memory_interference=self.memory_interference,
                config=self.prefill_config,
            )
            result["average_effective_capacity"] = max(
                worker.compute_capacity, self.prefill_config.min_capacity
            )
            return result, capacity_at_start

        capacity_at_start = self.fault_injector.effective_compute_capacity(
            worker_id=worker.worker_id,
            healthy_capacity=worker.compute_capacity,
            time_ms=start_time_ms,
        )

        nominal_service_time = estimate_prefill_time(
            prompt_tokens=request.prompt_tokens,
            compute_capacity=worker.compute_capacity,
            mapping_efficiency=self.prefill_mapping_efficiency,
            memory_interference=self.memory_interference,
            config=self.prefill_config,
        )

        result = execute_prefill(
            request=request,
            worker=worker,
            start_time_ms=start_time_ms,
            mapping_efficiency=self.prefill_mapping_efficiency,
            memory_interference=self.memory_interference,
            config=self.prefill_config,
        )

        integrated = calculate_compute_completion(
            start_time_ms=start_time_ms,
            nominal_service_time_ms=nominal_service_time,
            worker_id=worker.worker_id,
            healthy_capacity=worker.compute_capacity,
            min_capacity=self.prefill_config.min_capacity,
            fault_manager=self.fault_injector.compute,
        )

        result.update(integrated)
        return result, capacity_at_start

    def _execute_decode_with_capacity_profile(self, request, worker, start_time_ms):
        """Execute decode using capacity integrated across fault boundaries."""

        capacity_at_start = worker.compute_capacity

        if self.fault_injector is None:
            result = execute_decode(
                request=request,
                worker=worker,
                start_time_ms=start_time_ms,
                mapping_efficiency=self.decode_mapping_efficiency,
                memory_interference=self.memory_interference,
                config=self.decode_config,
            )
            result["average_effective_capacity"] = max(
                worker.compute_capacity, self.decode_config.min_capacity
            )
            return result, capacity_at_start

        capacity_at_start = self.fault_injector.effective_compute_capacity(
            worker_id=worker.worker_id,
            healthy_capacity=worker.compute_capacity,
            time_ms=start_time_ms,
        )

        nominal_service_time = estimate_decode_time(
            output_tokens=request.output_tokens,
            compute_capacity=worker.compute_capacity,
            mapping_efficiency=self.decode_mapping_efficiency,
            memory_interference=self.memory_interference,
            config=self.decode_config,
        )

        result = execute_decode(
            request=request,
            worker=worker,
            start_time_ms=start_time_ms,
            mapping_efficiency=self.decode_mapping_efficiency,
            memory_interference=self.memory_interference,
            config=self.decode_config,
        )

        integrated = calculate_compute_completion(
            start_time_ms=start_time_ms,
            nominal_service_time_ms=nominal_service_time,
            worker_id=worker.worker_id,
            healthy_capacity=worker.compute_capacity,
            min_capacity=self.decode_config.min_capacity,
            fault_manager=self.fault_injector.compute,
        )

        result.update(integrated)
        return result, capacity_at_start

    def _calculate_kv_cache_size(self, request):
        """
        Initial abstract KV-cache size approximation.

        This is configurable and should later be replaced with
        a model-specific KV-cache calculation.
        """
        return (
            request.prompt_tokens
            * self.kv_cache_mb_per_prompt_token
        )

    def _assigned_worker_pair(self, request):
        if self.worker_pair_assignments is None:
            return None

        try:
            return self.worker_pair_assignments[request.request_id]
        except KeyError as exc:
            raise ValueError(
                f"Missing worker-pair assignment for {request.request_id}"
            ) from exc

    def _schedule_prefill(self, requests):
        """
        First pass: schedule prefill execution.

        Workers are reserved until their calculated completion time.
        """
        self.prefill_records = []

        sorted_requests = sorted(
            requests,
            key=lambda request: request.arrival_time_ms,
        )

        for request in sorted_requests:

            assigned_pair = self._assigned_worker_pair(request)
            if assigned_pair is None:
                worker = self.system.scheduler.select_prefill_worker()
            else:
                worker = self.system.scheduler.get_prefill_worker(
                    assigned_pair[0]
                )

            request.prefill_queue_enter_ms = (
            request.arrival_time_ms
            )

            queue_depth = int(
            worker.busy_until_ms > request.arrival_time_ms
            )

            worker.add_to_queue(request)

            prefill_start_ms = max(
                request.arrival_time_ms,
                worker.busy_until_ms,
            )

            result, prefill_capacity = (
                self._execute_prefill_with_capacity_profile(
                    request=request,
                    worker=worker,
                    start_time_ms=prefill_start_ms,
                )
            )

            request.prefill_worker_id = worker.worker_id
            request.prefill_start_ms = (
                result["start_time_ms"]
            )
            request.prefill_completion_ms = (
                result["completion_time_ms"]
            )

            worker.reserve(
                result["completion_time_ms"]
            )

            worker.remove_from_queue(request)

            self.prefill_records.append({
                "request": request,
                "worker": worker,
                "queue_depth": queue_depth,
                "compute_capacity": prefill_capacity,
                "average_compute_capacity": (
                    result["average_effective_capacity"]
                ),
            })

    def _schedule_transfer_and_decode(self):
        """
        Second pass:

        1. Register all KV-cache transfers.
        2. Finalize transfer completion predictions.
        3. Calculate handoff and decode admission.
        4. Schedule decode in final admission order.
        5. Record finalized telemetry.

        Decode worker assignments are selected provisionally using the
        existing shortest-queue scheduler. Provisional reservations are
        then restored before final decode scheduling.
        """

        sorted_records = sorted(
            self.prefill_records,
            key=lambda record: (
                record["request"].prefill_completion_ms,
                record["request"].request_id,
            ),
        )

        decode_workers = list(
            self.system.scheduler.decode_workers.values()
        )

        original_busy_until = {
            worker.worker_id: worker.busy_until_ms
            for worker in decode_workers
        }

        transfer_records = []

        # -------------------------------------------------
        # Stage 1: Register transfers and assign workers.
        # -------------------------------------------------
        for record in sorted_records:
            request = record["request"]
            prefill_worker = record["worker"]

            kv_size_mb = self._calculate_kv_cache_size(request)
            request.kv_cache_size_mb = kv_size_mb

            assigned_pair = self._assigned_worker_pair(request)
            if assigned_pair is None:
                decode_worker = (
                    self.system.scheduler.select_decode_worker()
                )
            else:
                decode_worker = self.system.scheduler.get_decode_worker(
                    assigned_pair[1]
                )

            link = self.system.scheduler.select_link(
                prefill_worker_id=prefill_worker.worker_id,
                decode_worker_id=decode_worker.worker_id,
            )

            request.decode_worker_id = decode_worker.worker_id
            request.link_id = link.link_id
            request.transfer_start_ms = (
                request.prefill_completion_ms
            )

            provisional_result = link.start_transfer(
                request=request,
                current_time_ms=request.transfer_start_ms,
                data_size_mb=kv_size_mb,
            )

            # Use the provisional estimate only to reserve the worker
            # for subsequent worker-selection decisions.
            provisional_completion = provisional_result["completion_time_ms"]

            provisional_fault_delay = 0.0
            if self.fault_injector is not None:
                provisional_fault_delay = (
                    self.fault_injector.additional_synchronization_delay(
                        decode_worker_id=decode_worker.worker_id,
                        transfer_completion_time_ms=provisional_completion,
                    )
                )

            provisional_handoff = (
                self.system.synchronization.calculate_handoff(
                    request=request,
                    transfer_completion_time_ms=provisional_completion,
                    decode_worker_id=decode_worker.worker_id,
                    additional_delay_ms=provisional_fault_delay,
                )
            )

            provisional_admission = (
                provisional_handoff["decode_admission_time_ms"]
            )

            provisional_start = max(
                provisional_admission,
                decode_worker.busy_until_ms,
            )

            provisional_decode, _ = (
                self._execute_decode_with_capacity_profile(
                    request=request,
                    worker=decode_worker,
                    start_time_ms=provisional_start,
                )
            )

            decode_worker.reserve(
                provisional_decode["completion_time_ms"]
            )

            transfer_records.append({
                "record": record,
                "request": request,
                "prefill_worker": prefill_worker,
                "decode_worker": decode_worker,
                "link": link,
                "kv_size_mb": kv_size_mb,
            })

        # -------------------------------------------------
        # Stage 2: Restore worker availability.
        # -------------------------------------------------
        # Provisional reservations were only for worker assignment.
        # Do not reset workers: that could erase fault-injected state.
        for worker in decode_workers:
            worker.busy_until_ms = original_busy_until[
                worker.worker_id
            ]

        # -------------------------------------------------
        # Stage 3: Retrieve final transfer results and
        # calculate final admission timestamps.
        # -------------------------------------------------
        for item in transfer_records:
            request = item["request"]
            link = item["link"]
            decode_worker = item["decode_worker"]

            transfer_result = link.get_transfer_result(
                request.request_id
            )

            request.transfer_completion_ms = (
                transfer_result["completion_time_ms"]
            )

            final_fault_delay = 0.0
            if self.fault_injector is not None:
                final_fault_delay = (
                    self.fault_injector.additional_synchronization_delay(
                        decode_worker_id=decode_worker.worker_id,
                        transfer_completion_time_ms=request.transfer_completion_ms,
                    )
                )

            handoff_result = (
                self.system.synchronization.calculate_handoff(
                    request=request,
                    transfer_completion_time_ms=request.transfer_completion_ms,
                    decode_worker_id=decode_worker.worker_id,
                    additional_delay_ms=final_fault_delay,
                )
            )

            request.decode_admission_ms = (
                handoff_result["decode_admission_time_ms"]
            )

            item["transfer_result"] = transfer_result

        # -------------------------------------------------
        # Stage 4: Schedule decode using final admission
        # timestamps.
        # -------------------------------------------------
        ordered_records = sorted(
            transfer_records,
            key=lambda item: (
                item["request"].decode_admission_ms,
                item["request"].request_id,
            ),
        )

        for item in ordered_records:
            request = item["request"]
            record = item["record"]
            prefill_worker = item["prefill_worker"]
            decode_worker = item["decode_worker"]
            link = item["link"]
            transfer_result = item["transfer_result"]
            kv_size_mb = item["kv_size_mb"]

            decode_queue_depth = int(
                decode_worker.busy_until_ms
                > request.decode_admission_ms
            )

            decode_worker.add_to_queue(request)

            decode_start_ms = max(
                request.decode_admission_ms,
                decode_worker.busy_until_ms,
            )

            request.decode_start_ms = decode_start_ms

            decode_result, decode_capacity = (
                self._execute_decode_with_capacity_profile(
                    request=request,
                    worker=decode_worker,
                    start_time_ms=decode_start_ms,
                )
            )

            request.decode_completion_ms = (
                decode_result["completion_time_ms"]
            )

            decode_worker.reserve(
                request.decode_completion_ms
            )

            decode_worker.remove_from_queue(request)

            metadata = {
                "batch_id": None,
                "batch_size": 1,
                "link_type": link.link_type,
                "link_shared": link.shared,
                "kv_cache_size_mb": kv_size_mb,
                "configured_bandwidth_mb_per_ms": (
                    transfer_result[
                        "configured_bandwidth_mb_per_ms"
                    ]
                ),
                "effective_bandwidth_mb_per_ms": (
                    transfer_result[
                        "effective_bandwidth_mb_per_ms"
                    ]
                ),
                "concurrent_transfers": (
                    transfer_result["concurrent_transfers"]
                ),
                "average_concurrent_transfers": (
                    transfer_result["average_concurrent_transfers"]
                ),
                "peak_concurrent_transfers": (
                    transfer_result["peak_concurrent_transfers"]
                ),
                "prefill_compute_capacity": record["compute_capacity"],
                "decode_compute_capacity": decode_capacity,
                "prefill_average_compute_capacity": (
                    record["average_compute_capacity"]
                ),
                "decode_average_compute_capacity": (
                    decode_result["average_effective_capacity"]
                ),
                "prefill_mapping_id": prefill_worker.mapping_id,
                "decode_mapping_id": decode_worker.mapping_id,
                "prefill_queue_depth": record["queue_depth"],
                "decode_queue_depth": decode_queue_depth,
                "compute_interference": self.compute_interference,
                "memory_interference": self.memory_interference,
                "worker_utilization": None,
            }

            self.collector.record_request(
                request=request,
                metadata=metadata,
            )

    def run(self, requests, reset_system=True):
        """
        Run the complete PD simulation.

        Parameters
        ----------
        requests:
            List of Request objects.

        reset_system:
            Reset worker availability and communication state
            before running this experiment.

        Returns
        -------
        pandas.DataFrame
            Completed request telemetry.
        """
        if self.worker_pair_assignments is not None:
            missing = [
                request.request_id
                for request in requests
                if request.request_id not in self.worker_pair_assignments
            ]
            if missing:
                raise ValueError(
                    "Missing worker-pair assignments for: "
                    + ", ".join(missing[:5])
                )

        if reset_system:
            self.system.reset()
            self.collector.reset()
            self._recorded_fault_ids.clear()

        if self.fault_injector is not None:
            self.fault_injector.prepare_run(self.system)

            for fault in self.fault_injector.to_ground_truth_records():
                fault_id = fault["fault_id"]

                if fault_id not in self._recorded_fault_ids:
                    self.collector.record_fault(**fault)
                    self._recorded_fault_ids.add(fault_id)

        if not requests:
            return self.collector.get_trace_dataframe()

        # Pass 1
        self._schedule_prefill(requests)

        # Pass 2
        self._schedule_transfer_and_decode()

        trace_df = self.collector.get_trace_dataframe()

        if not trace_df.empty:
            trace_df = trace_df.sort_values(
                "arrival_time_ms"
            ).reset_index(drop=True)

        return trace_df