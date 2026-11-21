from pathlib import Path

import pandas as pd

from pd_failslow.simulator.engine import PDSimulationEngine
from pd_failslow.simulator.request import Request
from pd_failslow.utils.config import load_experiment_config, build_system


OUTPUT = Path("results/controlled_contention")
TRACE_DIR = OUTPUT / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

TRIALS = 20
PROMPT_TOKENS = 2000
OUTPUT_TOKENS = 128
TRIAL_SPACING_MS = 1000.0

PAYLOADS_MB = [20.0, 600.0]
CONCURRENCY_LEVELS = [1, 2]

config = load_experiment_config("configs")

all_summaries = []

for payload_mb in PAYLOADS_MB:
    for concurrency in CONCURRENCY_LEVELS:

        system = build_system(config)

        # Force every worker combination through the same L1 link.
        system.scheduler.link_routes = {
            ("P1", "D1"): "L1",
            ("P1", "D2"): "L1",
            ("P2", "D1"): "L1",
            ("P2", "D2"): "L1",
        }

        engine = PDSimulationEngine(
            system,
            kv_cache_mb_per_prompt_token=(
                payload_mb / PROMPT_TOKENS
            ),
            fault_injector=None,
        )

        requests = []

        for trial in range(TRIALS):
            arrival = trial * TRIAL_SPACING_MS

            for member in range(concurrency):
                requests.append(
                    Request(
                        request_id=(
                            f"payload{int(payload_mb)}"
                            f"_c{concurrency}"
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
            raise RuntimeError(
                f"Unexpected link assignment: "
                f"{trace['link_id'].value_counts().to_dict()}"
            )

        observed_peak = pd.to_numeric(
            trace["peak_concurrent_transfers"],
            errors="coerce",
        )

        if not observed_peak.eq(concurrency).all():
            raise RuntimeError(
                f"Expected concurrency {concurrency}, "
                f"observed peaks: {observed_peak.value_counts().to_dict()}"
            )

        name = f"payload_{int(payload_mb)}MB_concurrency_{concurrency}"
        trace.to_csv(TRACE_DIR / f"{name}.csv", index=False)

        transfer = pd.to_numeric(
            trace["transfer_time_ms"], errors="coerce"
        )

        summary = {
            "condition": name,
            "payload_mb": payload_mb,
            "requested_concurrency": concurrency,
            "request_count": len(trace),
            "mean_transfer_ms": transfer.mean(),
            "median_transfer_ms": transfer.median(),
            "p95_transfer_ms": transfer.quantile(0.95),
            "mean_observed_concurrency": trace[
                "average_concurrent_transfers"
            ].mean(),
            "mean_peak_concurrency": observed_peak.mean(),
            "link": trace["link_id"].unique().tolist(),
        }

        all_summaries.append(summary)

        print(
            f"\n{name}: "
            f"requests={len(trace)}, "
            f"mean transfer={transfer.mean():.3f} ms, "
            f"mean concurrency="
            f"{summary['mean_observed_concurrency']:.2f}"
        )

summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(
    OUTPUT / "controlled_contention_summary.csv",
    index=False,
)

print("\nFINAL CONTROLLED CONTENTION RESULTS")
print(summary_df.to_string(index=False))
print(f"\nResults saved in {OUTPUT}")
