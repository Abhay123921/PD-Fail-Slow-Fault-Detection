"""Assess the quality of evidence supporting stage-level decisions."""

import numpy as np
import pandas as pd

from pd_failslow.detection.preprocessing import STAGE_METRICS


class EvidenceConfidence:
    """Calculate heuristic evidence quality, not fault probability."""

    def __init__(
        self,
        reference_sample_target=100,
        calibration_sample_target=500,
    ):
        if reference_sample_target < 1:
            raise ValueError("reference_sample_target must be positive")

        if calibration_sample_target < 1:
            raise ValueError("calibration_sample_target must be positive")

        self.reference_sample_target = reference_sample_target
        self.calibration_sample_target = calibration_sample_target

    def predict(self, trace: pd.DataFrame) -> pd.DataFrame:
        """Add stage-specific evidence-quality scores."""

        if not isinstance(trace, pd.DataFrame):
            raise TypeError("trace must be a pandas DataFrame")

        result = trace.copy()

        for metric in STAGE_METRICS:
            output_col = f"{metric}_evidence_quality_score"
            status_col = f"{metric}_evidence_quality_status"

            ref_status_col = f"{metric}_status"
            ref_count_col = f"{metric}_sample_count"
            calibration_count_col = (
                f"{metric}_anomaly_calibration_count"
            )
            anomaly_status_col = f"{metric}_anomaly_status"
            persistence_status_col = (
                f"{metric}_persistence_status"
            )

            required = [
                ref_status_col,
                ref_count_col,
                calibration_count_col,
                anomaly_status_col,
                persistence_status_col,
            ]

            if any(column not in result.columns for column in required):
                result[output_col] = np.nan
                result[status_col] = "missing_inputs"
                continue

            scores = []
            statuses = []

            for _, row in result.iterrows():
                reference_status = row[ref_status_col]
                anomaly_status = row[anomaly_status_col]
                persistence_status = row[persistence_status_col]

                if reference_status not in ("ok", "fallback_global"):
                    scores.append(np.nan)
                    statuses.append("not_comparable")
                    continue

                if anomaly_status != "scored":
                    scores.append(np.nan)
                    statuses.append("insufficient_evidence")
                    continue

                if persistence_status != "evaluated":
                    scores.append(np.nan)
                    statuses.append("insufficient_evidence")
                    continue

                reference_count = pd.to_numeric(
                    pd.Series([row[ref_count_col]]),
                    errors="coerce",
                ).iloc[0]

                calibration_count = pd.to_numeric(
                    pd.Series([row[calibration_count_col]]),
                    errors="coerce",
                ).iloc[0]

                if (
                    pd.isna(reference_count)
                    or pd.isna(calibration_count)
                    or reference_count <= 0
                    or calibration_count <= 0
                ):
                    scores.append(np.nan)
                    statuses.append("insufficient_evidence")
                    continue

                reference_quality = (
                    1.0 if reference_status == "ok" else 0.7
                )

                reference_support = min(
                    float(reference_count)
                    / self.reference_sample_target,
                    1.0,
                )

                calibration_support = min(
                    float(calibration_count)
                    / self.calibration_sample_target,
                    1.0,
                )

                persistence_support = 1.0

                score = (
                    0.40 * reference_quality
                    + 0.20 * reference_support
                    + 0.20 * calibration_support
                    + 0.20 * persistence_support
                )

                scores.append(float(score))
                statuses.append("evidence_assessed")

            result[output_col] = scores
            result[status_col] = statuses

        return result
