"""Evidence-based component localization and ranking."""

import pandas as pd


OUTPUT_COLUMNS = [
    "fault_type",
    "component_type",
    "component_id",
    "evidence_count",
    "affected_stages",
    "stage_count",
    "mean_positive_residual_ms",
    "mean_anomaly_excess_ms",
    "mean_evidence_quality_score",
    "last_seen_ms",
    "localization_rank",
]


def rank_localized_components(
    candidates: pd.DataFrame,
    min_evidence: int = 1,
) -> pd.DataFrame:
    """
    Rank suspicious components using persistent direct-attribution evidence.

    Rankings are calculated separately for each fault type.
    This is an evidence ranking, not a calibrated fault probability.

    Ground-truth fault labels are never used.
    """
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")

    if not isinstance(min_evidence, int) or min_evidence < 1:
        raise ValueError("min_evidence must be a positive integer")

    if candidates.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    required = {
        "fault_type",
        "component_type",
        "component_id",
        "evidence_role",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(
            f"Missing required candidate columns: {sorted(missing)}"
        )

    direct = candidates.loc[
        candidates["evidence_role"].eq("direct")
    ].copy()

    direct = direct.dropna(
        subset=["fault_type", "component_type", "component_id"]
    )

    if direct.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    numeric_columns = [
        "positive_residual_ms",
        "anomaly_excess_ms",
        "evidence_quality_score",
        "timestamp_ms",
    ]

    for column in numeric_columns:
        if column in direct.columns:
            direct[column] = pd.to_numeric(
                direct[column], errors="coerce"
            )

    aggregations = {
        "evidence_count": ("component_id", "size"),
    }

    if "stage" in direct.columns:
        aggregations["affected_stages"] = (
            "stage",
            lambda values: sorted(set(values.dropna())),
        )
    else:
        aggregations["affected_stages"] = (
            "component_id",
            lambda values: [],
        )

    if "positive_residual_ms" in direct.columns:
        aggregations["mean_positive_residual_ms"] = (
            "positive_residual_ms", "mean"
        )

    if "anomaly_excess_ms" in direct.columns:
        aggregations["mean_anomaly_excess_ms"] = (
            "anomaly_excess_ms", "mean"
        )

    if "evidence_quality_score" in direct.columns:
        aggregations["mean_evidence_quality_score"] = (
            "evidence_quality_score", "mean"
        )

    if "timestamp_ms" in direct.columns:
        aggregations["last_seen_ms"] = ("timestamp_ms", "max")

    grouped = (
        direct.groupby(
            ["fault_type", "component_type", "component_id"],
            dropna=True,
        )
        .agg(**aggregations)
        .reset_index()
    )

    grouped["stage_count"] = grouped["affected_stages"].map(len)

    for column in OUTPUT_COLUMNS:
        if column not in grouped.columns:
            grouped[column] = pd.NA

    grouped = grouped.loc[
        grouped["evidence_count"] >= min_evidence
    ].copy()

    if grouped.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Rank only within the same fault type. Residual magnitudes from
    # different fault types are not assumed to be directly comparable.
    grouped = grouped.sort_values(
        by=[
            "fault_type",
            "evidence_count",
            "mean_anomaly_excess_ms",
            "mean_evidence_quality_score",
            "component_id",
        ],
        ascending=[True, False, False, False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    grouped["localization_rank"] = (
        grouped.groupby("fault_type").cumcount() + 1
    )

    return grouped[OUTPUT_COLUMNS]
