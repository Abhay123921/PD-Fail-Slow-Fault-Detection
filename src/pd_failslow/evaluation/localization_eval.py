"""Evaluation of fault-type identification and component localization."""

import numpy as np
import pandas as pd


def evaluate_localization(ranking, ground_truth, top_k=(1, 3)):
    """
    Evaluate ranked component localization against fault ground truth.

    Component ranking is evaluated within each ground-truth fault type.
    A component is correct only when fault_type, component_type and
    component_id all match.

    Returns fault-type accuracy, top-k component accuracy, MRR and
    per-fault localization details.
    """
    if not isinstance(ranking, pd.DataFrame):
        raise TypeError("ranking must be a pandas DataFrame")

    if not isinstance(ground_truth, pd.DataFrame):
        raise TypeError("ground_truth must be a pandas DataFrame")

    ranking_required = {
        "fault_type",
        "component_type",
        "component_id",
        "localization_rank",
    }
    gt_required = {"fault_type", "component_type", "component_id"}

    missing_ranking = ranking_required - set(ranking.columns)
    missing_gt = gt_required - set(ground_truth.columns)

    if missing_ranking:
        raise ValueError(
            f"Missing ranking columns: {sorted(missing_ranking)}"
        )

    if missing_gt:
        raise ValueError(
            f"Missing ground-truth columns: {sorted(missing_gt)}"
        )

    top_k = tuple(top_k)

    if not top_k or any(
        isinstance(k, bool) or not isinstance(k, int) or k < 1
        for k in top_k
    ):
        raise ValueError("top_k must contain positive integers")

    if len(set(top_k)) != len(top_k):
        raise ValueError("top_k values must be unique")

    ranked = ranking.copy()
    ranked["localization_rank"] = pd.to_numeric(
        ranked["localization_rank"], errors="coerce"
    )

    if (
        ranked["localization_rank"].isna().any()
        or (ranked["localization_rank"] < 1).any()
    ):
        raise ValueError("localization_rank must contain positive ranks")

    details = []

    for fault in ground_truth.itertuples(index=False):
        candidates = ranked.loc[
            ranked["fault_type"].eq(fault.fault_type)
        ].sort_values(
            ["localization_rank", "component_id"],
            kind="stable",
        )

        fault_type_found = not candidates.empty

        matching = candidates.loc[
            candidates["component_type"].eq(fault.component_type)
            & candidates["component_id"].eq(fault.component_id)
        ]

        if matching.empty:
            correct_rank = None
        else:
            correct_rank = int(matching["localization_rank"].min())

        row = {
            "fault_type": fault.fault_type,
            "ground_truth_component_type": fault.component_type,
            "ground_truth_component_id": fault.component_id,
            "fault_type_found": fault_type_found,
            "correct_component_rank": correct_rank,
            "top_1_correct": correct_rank == 1,
            "reciprocal_rank": (
                1.0 / correct_rank if correct_rank is not None else 0.0
            ),
        }

        for k in top_k:
            row[f"top_{k}_correct"] = (
                correct_rank is not None and correct_rank <= k
            )

        details.append(row)

    details_df = pd.DataFrame(details)
    fault_count = len(details_df)

    if fault_count == 0:
        metrics = {
            "fault_count": 0,
            "fault_type_accuracy": 0.0,
            "component_top_1_accuracy": 0.0,
            "mean_reciprocal_rank": 0.0,
        }
        for k in top_k:
            metrics[f"component_top_{k}_accuracy"] = 0.0
    else:
        metrics = {
            "fault_count": fault_count,
            "fault_type_accuracy": float(
                details_df["fault_type_found"].mean()
            ),
            "component_top_1_accuracy": float(
                details_df["top_1_correct"].mean()
            ),
            "mean_reciprocal_rank": float(
                details_df["reciprocal_rank"].mean()
            ),
        }

        for k in top_k:
            metrics[f"component_top_{k}_accuracy"] = float(
                details_df[f"top_{k}_correct"].mean()
            )

    return {
        "metrics": metrics,
        "details": details_df,
    }
