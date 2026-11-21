from pathlib import Path

import numpy as np
import pandas as pd

from pd_failslow.detection.reference import HealthyReferenceModel


TRACE_DIR = Path("results/healthy_characterization/traces")
OUTPUT_DIR = Path("results/healthy_reference_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILES = [
    "stable",
    "long_prompt",
    "long_output",
    "high_arrival",
    "mixed",
]

STAGES = [
    "prefill_service_time_ms",
    "prefill_queue_wait_ms",
    "transfer_time_ms",
    "handoff_wait_ms",
    "admission_wait_ms",
    "decode_service_time_ms",
]


def load_traces(seed):
    frames = []

    for profile in PROFILES:
        path = TRACE_DIR / f"{profile}_seed{seed}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        frame["evaluation_profile"] = profile
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def evaluate(model, trace, dataset_name):
    prediction = model.predict(trace)
    records = []

    for stage in STAGES:
        reference_col = f"{stage}_reference"
        status_col = f"{stage}_status"

        if reference_col not in prediction.columns:
            continue

        actual = pd.to_numeric(
            prediction[stage], errors="coerce"
        )
        estimated = pd.to_numeric(
            prediction[reference_col], errors="coerce"
        )
        status = prediction[status_col]

        valid = actual.notna()
        covered = valid & estimated.notna()
        errors = (actual[covered] - estimated[covered]).abs()

        records.append({
            "dataset": dataset_name,
            "stage": stage,
            "total_requests": int(valid.sum()),
            "reference_coverage_pct": round(
                100 * covered.sum() / max(valid.sum(), 1), 2
            ),
            "group_reference_pct": round(
                100 * (status[valid] == "ok").sum()
                / max(valid.sum(), 1), 2
            ),
            "global_fallback_pct": round(
                100 * (status[valid] == "fallback_global").sum()
                / max(valid.sum(), 1), 2
            ),
            "insufficient_reference_pct": round(
                100 * (status[valid] == "insufficient_reference").sum()
                / max(valid.sum(), 1), 2
            ),
            "mae_ms": round(float(errors.mean()), 4)
            if len(errors) else np.nan,
            "median_absolute_error_ms": round(
                float(errors.median()), 4
            ) if len(errors) else np.nan,
        })

    return records


print("Loading healthy training traces: seed 42")
train = load_traces(42)

print("Loading held-out healthy traces: seed 43")
test = load_traces(43)

print(f"Training requests: {len(train)}")
print(f"Held-out requests: {len(test)}")

model = HealthyReferenceModel(min_samples=20)
model.fit(train)

all_records = []

all_records.extend(
    evaluate(model, test, "held_out_seed43")
)

for profile in PROFILES:
    subset = test[test["evaluation_profile"] == profile]
    all_records.extend(
        evaluate(model, subset, profile)
    )

results = pd.DataFrame(all_records)
output_path = OUTPUT_DIR / "held_out_reference_metrics.csv"
results.to_csv(output_path, index=False)

print("\nHELD-OUT HEALTHY REFERENCE RESULTS")
print(results.to_string(index=False))

print(f"\nSaved results to: {output_path}")
print("\nTraining profiles:", sorted(train["evaluation_profile"].unique()))
print("Evaluation profiles:", sorted(test["evaluation_profile"].unique()))
