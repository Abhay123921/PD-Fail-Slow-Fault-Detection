"""Convert detection and localization evidence into a system-level decision."""

import numpy as np
import pandas as pd

from pd_failslow.detection.preprocessing import STAGE_METRICS


DIRECT_STAGES = {
    "prefill_service_time_ms",
    "decode_service_time_ms",
    "transfer_time_ms",
    "handoff_wait_ms",
}

CONTEXT_STAGES = {
    "prefill_queue_wait_ms",
    "admission_wait_ms",
}


def decide_diagnosis(
    trace: pd.DataFrame,
    candidates: pd.DataFrame,
    localized: pd.DataFrame,
    min_evidence: int = 3,
    min_quality: float = 0.5,
) -> dict:
    """Distinguish workload pressure from component-level fail-slow evidence."""

    if not isinstance(trace, pd.DataFrame):
        raise TypeError("trace must be a DataFrame")
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a DataFrame")
    if not isinstance(localized, pd.DataFrame):
        raise TypeError("localized must be a DataFrame")
    if not isinstance(min_evidence, int) or min_evidence < 1:
        raise ValueError("min_evidence must be a positive integer")
    if not np.isfinite(min_quality) or not 0.0 <= min_quality <= 1.0:
        raise ValueError("min_quality must be between 0 and 1")

    result = {
        "status": "insufficient_evidence",
        "is_fault_suspected": None,
        "fault_type": None,
        "component_type": None,
        "component_id": None,
        "localization_rank": None,
        "evidence_count": 0,
        "mean_evidence_quality_score": None,
        "scored_stage_count": 0,
        "total_stage_count": len(STAGE_METRICS),
        "anomalous_observation_count": 0,
        "direct_anomalous_observation_count": 0,
        "persistent_direct_anomaly_count": 0,
        "context_anomalous_observation_count": 0,
        "persistent_context_evidence_count": 0,
        "candidate_fault_types": [],
        "reason": "",
    }

    if trace.empty:
        result["reason"] = "empty_trace"
        return result

    scored_stages = []
    direct_anomalies = 0
    persistent_direct_anomalies = 0
    context_anomalies = 0

    for metric in STAGE_METRICS:
        status_col = f"{metric}_anomaly_status"
        flag_col = f"{metric}_anomaly"

        if status_col not in trace.columns or flag_col not in trace.columns:
            continue

        scored = trace[status_col].eq("scored")
        if not scored.any():
            continue

        scored_stages.append(metric)
        count = int(trace.loc[scored, flag_col].eq(True).sum())
        persistent_col = f"{metric}_persistent_anomaly"
        persistent_count = 0
        if persistent_col in trace.columns:
            persistent_count = int(
                trace.loc[scored, persistent_col].eq(True).sum()
            )

        if metric in DIRECT_STAGES:
            direct_anomalies += count
            persistent_direct_anomalies += persistent_count
        elif metric in CONTEXT_STAGES:
            context_anomalies += count

    result["scored_stage_count"] = len(scored_stages)
    result["direct_anomalous_observation_count"] = direct_anomalies
    result["persistent_direct_anomaly_count"] = persistent_direct_anomalies
    result["context_anomalous_observation_count"] = context_anomalies
    result["anomalous_observation_count"] = (
        direct_anomalies + context_anomalies
    )

    # Record persistent context-only evidence without attributing blame.
    if not candidates.empty and "evidence_role" in candidates.columns:
        result["persistent_context_evidence_count"] = int(
            candidates["evidence_role"].eq("context_only").sum()
        )

    if not scored_stages:
        result["reason"] = "no_scored_anomaly_observations"
        return result

    # Direct stage anomalies are the only stage anomalies that can
    # initiate a component-level fail-slow suspicion.
    if direct_anomalies > 0:
        if persistent_direct_anomalies < min_evidence:
            result["is_fault_suspected"] = False
            if context_anomalies > 0:
                result["status"] = "workload_pressure"
                result["reason"] = (
                    "context_pressure_without_persistent_direct_evidence"
                )
            else:
                result["status"] = "insufficient_evidence"
                result["reason"] = "direct_anomalies_below_persistence_evidence_threshold"
            return result

        result["is_fault_suspected"] = True

        if localized.empty:
            result["status"] = "suspicious"
            result["reason"] = "direct_anomalies_without_localized_component"
            return result

        required = {
            "fault_type",
            "component_type",
            "component_id",
            "evidence_count",
            "localization_rank",
            "mean_evidence_quality_score",
        }
        missing = required.difference(localized.columns)

        if missing:
            result["status"] = "suspicious"
            result["reason"] = (
                f"localization_missing_columns:{sorted(missing)}"
            )
            return result

        valid = localized.dropna(
            subset=["fault_type", "component_type", "component_id"]
        ).copy()

        if valid.empty:
            result["status"] = "suspicious"
            result["reason"] = "no_valid_direct_localization"
            return result

        fault_types = sorted(valid["fault_type"].unique().tolist())
        result["candidate_fault_types"] = fault_types

        if len(fault_types) != 1:
            result["status"] = "suspicious"
            result["reason"] = "multiple_fault_types_compete"
            return result

        for column in (
            "evidence_count",
            "localization_rank",
            "mean_evidence_quality_score",
        ):
            valid[column] = pd.to_numeric(
                valid[column], errors="coerce"
            )

        top = valid.loc[valid["localization_rank"].eq(1)]

        if top.empty:
            result["status"] = "suspicious"
            result["reason"] = "no_top_ranked_component"
            return result

        candidate = top.iloc[0]
        evidence_count = candidate["evidence_count"]
        quality = candidate["mean_evidence_quality_score"]

        if (
            pd.isna(evidence_count)
            or evidence_count < min_evidence
            or pd.isna(quality)
            or quality < min_quality
        ):
            result["status"] = "suspicious"
            result["reason"] = "localization_evidence_below_threshold"
            return result

        result.update({
            "status": "localized_fault",
            "fault_type": candidate["fault_type"],
            "component_type": candidate["component_type"],
            "component_id": candidate["component_id"],
            "localization_rank": int(candidate["localization_rank"]),
            "evidence_count": int(evidence_count),
            "mean_evidence_quality_score": float(quality),
            "reason": "persistent_direct_evidence_supports_component",
        })
        return result

    # Queue/admission anomalies indicate pressure, not component blame.
    if context_anomalies > 0:
        result["status"] = "workload_pressure"
        result["is_fault_suspected"] = False
        result["reason"] = "context_only_queue_or_admission_anomalies"
        return result

    # A healthy decision requires complete coverage of every monitored stage.
    result["is_fault_suspected"] = False

    if len(scored_stages) != len(STAGE_METRICS):
        result["reason"] = "incomplete_stage_coverage"
        return result

    result["status"] = "healthy"
    result["reason"] = "all_stages_scored_without_anomalies"
    return result
