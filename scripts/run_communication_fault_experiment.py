from pathlib import Path

import pandas as pd

from pd_failslow.simulator.engine import PDSimulationEngine
from pd_failslow.simulator.request import Request

from pd_failslow.faults.injector import FaultInjector
from pd_failslow.faults.communication_faults import CommunicationFault

from pd_failslow.utils.config import load_experiment_config, build_system

from pd_failslow.detection.communication_diagnostic import (
    CommunicationDiagnostic,
)


OUTPUT = Path("results/communication_fault_experiment")
TRACE_DIR = OUTPUT / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

TRIALS = 20
PROMPT_TOKENS = 2000
OUTPUT_TOKENS = 128
TRIAL_SPACING_MS = 1000.0
PAYLOAD_MB = 20.0

SEVERITIES = [0.0, 0.25, 0.50]

FAULT_START_MS = 0.0
FAULT_END_MS = 20000.0

config = load_experiment_config("configs")
summaries = []

for severity in SEVERITIES:

    system = build_system(config)

    # Route all requests through the same shared PCIe link.
    system.scheduler.link_routes = {
        ("P1", "D1"): "L1",
        ("P1", "D2"): "L1",
        ("P2", "D1"): "L1",
        ("P2", "D2"): "L1",
    }

    injector = None

    if severity > 0:
        injector = FaultInjector()
        injector.register_communication_fault(
            CommunicationFault(
                fault_id=f"L1_bandwidth_{int(severity * 100)}pct",
                link_id="L1",
                start_time_ms=FAULT_START_MS,
                end_time_ms=FAULT_END_MS,
                severity=severity,
            )
        )

    engine = PDSimulationEngine(
        system,
        kv_cache_mb_per_prompt_token=(
            PAYLOAD_MB / PROMPT_TOKENS
        ),
        fault_injector=injector,
    )

    # Construct the diagnostic from healthy configured link values.
    diagnostic = CommunicationDiagnostic(system)

    requests = []

    for trial in range(TRIALS):
        arrival = trial * TRIAL_SPACING_MS

        for member in range(2):
            requests.append(
                Request(
                    request_id=(
                        f"severity{int(severity * 100)}"
                        f"_trial{trial}"
                        f"_member{member}"
                    ),
                    arrival_time_ms=arrival,
                    prompt_tokens=PROMPT_TOKENS,
                    output_tokens=OUTPUT_TOKENS,
                )
            )

    trace = engine.run(requests, reset_system=True)

    if not trace["link_id"].eq("L1").all():
        raise RuntimeError("Unexpected link assignment")

    if not pd.to_numeric(
        trace["peak_concurrent_transfers"],
        errors="coerce",
    ).eq(2).all():
        raise RuntimeError(
            "Expected every transfer to experience peak concurrency 2"
        )

    # Apply diagnostic to telemetry only.
    diagnosed = diagnostic.predict(trace)

    name = f"communication_severity_{int(severity * 100)}pct"

    trace.to_csv(TRACE_DIR / f"{name}_trace.csv", index=False)
    diagnosed.to_csv(
        TRACE_DIR / f"{name}_diagnosed.csv",
        index=False,
    )

    actual = pd.to_numeric(
        diagnosed["transfer_time_ms"], errors="coerce"
    )
    expected = pd.to_numeric(
        diagnosed["communication_expected_time_ms"],
        errors="coerce",
    )
    residual = pd.to_numeric(
        diagnosed["communication_residual_ms"],
        errors="coerce",
    )

    summary = {
        "severity": severity,
        "request_count": len(diagnosed),
        "mean_actual_transfer_ms": actual.mean(),
        "mean_expected_transfer_ms": expected.mean(),
        "mean_residual_ms": residual.mean(),
        "median_residual_ms": residual.median(),
        "p95_residual_ms": residual.quantile(0.95),
        "estimated_count": int(
            diagnosed["communication_diagnostic_status"]
            .eq("estimated").sum()
        ),
        "other_status_count": int(
            diagnosed["communication_diagnostic_status"]
            .ne("estimated").sum()
        ),
        "mean_concurrency": diagnosed[
            "average_concurrent_transfers"
        ].mean(),
    }

    summaries.append(summary)

    print(f"\n{name}")
    print(pd.Series(summary).to_string())

summary_df = pd.DataFrame(summaries)

summary_df.to_csv(
    OUTPUT / "communication_fault_summary.csv",
    index=False,
)

print("\nFINAL COMMUNICATION FAULT RESULTS")
print(summary_df.to_string(index=False))

print(f"\nResults saved in {OUTPUT}")
