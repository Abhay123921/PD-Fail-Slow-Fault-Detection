"""Configuration loading and construction for reproducible PD experiments."""

from pathlib import Path

import yaml

from pd_failslow.simulator.system import PDSystem
from pd_failslow.workload.generator import generate_requests
from pd_failslow.faults.injector import FaultInjector
from pd_failslow.faults.compute_faults import ComputeFault
from pd_failslow.faults.communication_faults import CommunicationFault
from pd_failslow.faults.synchronization_faults import SynchronizationFault


CONFIG_FILES = (
    "system.yaml",
    "workload.yaml",
    "faults.yaml",
    "detection.yaml",
)


def load_yaml(path):
    """Safely load one YAML file and require a mapping at its root."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")

    return data


def load_experiment_config(config_dir="configs"):
    """Load and validate the four experiment configuration files."""
    config_dir = Path(config_dir)

    config = {
        name.removesuffix(".yaml"): load_yaml(config_dir / name)
        for name in CONFIG_FILES
    }

    validate_experiment_config(config)
    return config


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")


def _require_list(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")


def _require_number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")

    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def validate_experiment_config(config):
    """Validate configuration structure and cross-component references."""
    for section in ("system", "workload", "faults", "detection"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
        _require_mapping(config[section], section)

    system = config["system"]
    workload = config["workload"]
    faults = config["faults"].get("faults", {})

    for section in ("prefill", "decode", "communication", "scheduler"):
        if section not in system:
            raise ValueError(f"Missing system section: {section}")
        _require_mapping(system[section], f"system.{section}")

    prefill_workers = system["prefill"].get("workers")
    decode_workers = system["decode"].get("workers")
    links = system["communication"].get("links")

    _require_list(prefill_workers, "system.prefill.workers")
    _require_list(decode_workers, "system.decode.workers")
    _require_list(links, "system.communication.links")

    worker_ids = set()
    for worker_type, workers in (
        ("prefill", prefill_workers),
        ("decode", decode_workers),
    ):
        for worker in workers:
            _require_mapping(worker, f"{worker_type} worker")
            worker_id = worker.get("id")
            if not isinstance(worker_id, str) or not worker_id.strip():
                raise ValueError(f"{worker_type} worker requires a valid id")
            if worker_id in worker_ids:
                raise ValueError(f"Duplicate component ID: {worker_id}")
            worker_ids.add(worker_id)

            _require_number(
                worker.get("compute_capacity"),
                f"{worker_id}.compute_capacity",
                minimum=0.000001,
            )
            _require_number(
                worker.get("memory_capacity"),
                f"{worker_id}.memory_capacity",
                minimum=0.000001,
            )

    link_ids = set()
    for link in links:
        _require_mapping(link, "communication link")
        link_id = link.get("id")
        if not isinstance(link_id, str) or not link_id.strip():
            raise ValueError("Communication link requires a valid id")
        if link_id in link_ids:
            raise ValueError(f"Duplicate link ID: {link_id}")
        link_ids.add(link_id)

        _require_number(
            link.get("bandwidth_mb_per_ms"),
            f"{link_id}.bandwidth_mb_per_ms",
            minimum=0.000001,
        )
        _require_number(
            link.get("fixed_latency_ms", 0.0),
            f"{link_id}.fixed_latency_ms",
            minimum=0.0,
        )

    scheduler = system["scheduler"]
    routes = scheduler.get("link_routes", {})
    _require_mapping(routes, "system.scheduler.link_routes")

    for route, link_id in routes.items():
        if "->" not in route:
            raise ValueError(f"Invalid route key: {route}")
        prefill_id, decode_id = route.split("->", 1)
        if prefill_id not in {
            worker["id"] for worker in prefill_workers
        }:
            raise ValueError(f"Unknown prefill worker in route: {route}")
        if decode_id not in {
            worker["id"] for worker in decode_workers
        }:
            raise ValueError(f"Unknown decode worker in route: {route}")
        if link_id not in link_ids:
            raise ValueError(f"Unknown link in route: {link_id}")

    workload_section = workload.get("workload")
    _require_mapping(workload_section, "workload.workload")

    _require_number(
        workload_section.get("num_requests"),
        "workload.num_requests",
        minimum=0,
    )

    arrival = workload_section.get("arrival")
    prompt = workload_section.get("prompt_length")
    output = workload_section.get("output_length")

    _require_mapping(arrival, "workload.arrival")
    _require_mapping(prompt, "workload.prompt_length")
    _require_mapping(output, "workload.output_length")

    _require_number(
        arrival.get("rate_per_ms"),
        "workload.arrival.rate_per_ms",
        minimum=0.000001,
    )

    for name, section in (("prompt_length", prompt), ("output_length", output)):
        if section.get("distribution") not in ("uniform", "fixed"):
            raise ValueError(f"Unsupported {name} distribution")
        _require_number(section.get("min_tokens"), f"{name}.min_tokens", minimum=1)
        _require_number(section.get("max_tokens"), f"{name}.max_tokens", minimum=1)

    batch = workload_section.get("batch", {})
    _require_mapping(batch, "workload.batch")
    if batch.get("enabled", False):
        raise ValueError(
            "Batching is not implemented in the current simulator; "
            "set workload.batch.enabled to false"
        )

    if faults.get("enabled", False):
        prefill_ids = {worker["id"] for worker in prefill_workers}
        decode_ids = {worker["id"] for worker in decode_workers}

        for fault in faults.get("compute", []):
            _require_mapping(fault, "compute fault")
            if fault.get("component_id") not in worker_ids:
                raise ValueError("Unknown compute-fault component")
            _validate_fault_interval(fault)

        for fault in faults.get("communication", []):
            _require_mapping(fault, "communication fault")
            if fault.get("component_id") not in link_ids:
                raise ValueError("Unknown communication-fault link")
            _validate_fault_interval(fault)

        for fault in faults.get("synchronization", []):
            _require_mapping(fault, "synchronization fault")
            if fault.get("component_id") not in decode_ids:
                raise ValueError(
                    "Synchronization faults must target decode workers"
                )
            _validate_fault_interval(fault)
            _require_number(fault.get("severity"), "sync severity", minimum=0.0)
            _require_number(
                fault.get("max_additional_delay_ms"),
                "sync max_additional_delay_ms",
                minimum=0.0,
            )

    return True


def _validate_fault_interval(fault):
    _require_number(fault.get("start_ms"), "fault.start_ms", minimum=0.0)
    _require_number(fault.get("end_ms"), "fault.end_ms", minimum=0.000001)
    _require_number(fault.get("severity"), "fault.severity", minimum=0.0)
    if fault["end_ms"] <= fault["start_ms"]:
        raise ValueError("Fault end_ms must be greater than start_ms")
    if fault["severity"] >= 1.0:
        raise ValueError("Fault severity must be less than 1.0")


def build_system(config):
    """Construct a fresh PDSystem from a validated configuration."""
    validate_experiment_config(config)
    system_config = config["system"]

    return PDSystem(
        prefill_config=system_config["prefill"],
        decode_config=system_config["decode"],
        communication_config=system_config["communication"],
        scheduler_config=system_config["scheduler"],
        synchronization_config=system_config.get("synchronization", {}),
    )


def build_fault_injector(config):
    """Construct a fresh typed fault injector from configuration."""
    validate_experiment_config(config)
    fault_config = config["faults"]["faults"]

    injector = FaultInjector()

    if not fault_config.get("enabled", False):
        return injector

    for fault in fault_config.get("compute", []):
        injector.register_compute_fault(
            ComputeFault(
                fault_id=fault["id"],
                worker_id=fault["component_id"],
                start_time_ms=fault["start_ms"],
                end_time_ms=fault["end_ms"],
                severity=fault["severity"],
            )
        )

    for fault in fault_config.get("communication", []):
        injector.register_communication_fault(
            CommunicationFault(
                fault_id=fault["id"],
                link_id=fault["component_id"],
                start_time_ms=fault["start_ms"],
                end_time_ms=fault["end_ms"],
                severity=fault["severity"],
            )
        )

    for fault in fault_config.get("synchronization", []):
        injector.register_synchronization_fault(
            SynchronizationFault(
                fault_id=fault["id"],
                component_id=fault["component_id"],
                start_time_ms=fault["start_ms"],
                end_time_ms=fault["end_ms"],
                severity=fault["severity"],
                max_additional_delay_ms=fault["max_additional_delay_ms"],
            )
        )

    return injector


def build_requests(config, seed=None):
    """Generate reproducible requests from the workload configuration."""
    validate_experiment_config(config)
    workload = config["workload"]["workload"]

    seed = (
        config["system"].get("system", {}).get("random_seed", 42)
        if seed is None
        else seed
    )

    arrival = workload["arrival"]
    prompt = workload["prompt_length"]
    output = workload["output_length"]

    return generate_requests(
        num_requests=int(workload["num_requests"]),
        seed=int(seed),
        arrival_distribution=arrival["distribution"],
        arrival_rate_per_ms=float(arrival["rate_per_ms"]),
        prompt_distribution=prompt["distribution"],
        min_prompt_tokens=int(prompt["min_tokens"]),
        max_prompt_tokens=int(prompt["max_tokens"]),
        fixed_prompt_tokens=prompt.get("fixed_tokens"),
        output_distribution=output["distribution"],
        min_output_tokens=int(output["min_tokens"]),
        max_output_tokens=int(output["max_tokens"]),
        fixed_output_tokens=output.get("fixed_tokens"),
    )
