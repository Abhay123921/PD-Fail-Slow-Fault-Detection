"""Empirical healthy-calibrated, stage-wise anomaly detection."""

import numpy as np
import pandas as pd

from pd_failslow.detection.preprocessing import STAGE_METRICS
from pd_failslow.detection.comparability import get_comparable_mask


class StageAnomalyDetector:
    """Detect unusually high stage latency using healthy calibration data."""

    LINK_CONDITION_COLUMNS = (
        "link_type",
        "configured_bandwidth_mb_per_ms",
    )

    def __init__(
        self,
        quantile: float = 0.99,
        min_samples: int = 100,
        min_threshold_ms: float = 1e-6,
        link_conditioned_transfer: bool = False,
    ):
        if not 0.5 < quantile < 1.0:
            raise ValueError("quantile must be between 0.5 and 1.0")

        if min_samples < 1:
            raise ValueError("min_samples must be positive")

        if not np.isfinite(min_threshold_ms) or min_threshold_ms < 0:
            raise ValueError(
                "min_threshold_ms must be finite and non-negative"
            )

        self.quantile = quantile
        self.min_samples = min_samples
        self.min_threshold_ms = min_threshold_ms
        self.link_conditioned_transfer = link_conditioned_transfer

        self.thresholds_ = {}
        self.sample_counts_ = {}
        self.link_thresholds_ = {}
        self.link_sample_counts_ = {}
        self.is_fitted_ = False

    def fit(self, healthy_calibration_trace: pd.DataFrame):
        """Calibrate global stage thresholds and optional link thresholds."""

        if not isinstance(healthy_calibration_trace, pd.DataFrame):
            raise TypeError("Calibration data must be a DataFrame")

        if healthy_calibration_trace.empty:
            raise ValueError("Calibration data cannot be empty")

        thresholds = {}
        counts = {}

        for metric in STAGE_METRICS:
            positive_col = f"{metric}_positive_residual_ms"

            if positive_col not in healthy_calibration_trace.columns:
                thresholds[metric] = np.nan
                counts[metric] = 0
                continue

            comparable = get_comparable_mask(
                healthy_calibration_trace, metric
            )

            values = pd.to_numeric(
                healthy_calibration_trace.loc[comparable, positive_col],
                errors="coerce",
            )

            values = values[np.isfinite(values)].clip(lower=0)
            counts[metric] = int(len(values))

            if len(values) < self.min_samples:
                thresholds[metric] = np.nan
                continue

            threshold = float(values.quantile(self.quantile))
            thresholds[metric] = max(
                threshold,
                self.min_threshold_ms,
            )

        self.thresholds_ = thresholds
        self.sample_counts_ = counts

        self.link_thresholds_ = {}
        self.link_sample_counts_ = {}

        if self.link_conditioned_transfer:
            metric = "transfer_time_ms"
            positive_col = f"{metric}_positive_residual_ms"

            missing = [
                column
                for column in self.LINK_CONDITION_COLUMNS
                if column not in healthy_calibration_trace.columns
            ]

            if missing:
                raise ValueError(
                    "Link-conditioned transfer calibration requires "
                    f"columns: {missing}"
                )

            if positive_col not in healthy_calibration_trace.columns:
                raise ValueError(
                    f"Missing calibration column: {positive_col}"
                )

            comparable = get_comparable_mask(
                healthy_calibration_trace, metric
            )

            valid = healthy_calibration_trace.loc[comparable].copy()
            valid[positive_col] = pd.to_numeric(
                valid[positive_col], errors="coerce"
            )

            valid = valid[
                np.isfinite(valid[positive_col])
                & valid[list(self.LINK_CONDITION_COLUMNS)].notna().all(axis=1)
            ]

            for key, group in valid.groupby(
                list(self.LINK_CONDITION_COLUMNS),
                sort=False,
                dropna=True,
            ):
                if not isinstance(key, tuple):
                    key = (key,)

                values = group[positive_col].clip(lower=0)
                count = int(len(values))
                normalized_key = tuple(key)

                self.link_sample_counts_[normalized_key] = count

                if count < self.min_samples:
                    self.link_thresholds_[normalized_key] = np.nan
                    continue

                threshold = float(values.quantile(self.quantile))
                self.link_thresholds_[normalized_key] = max(
                    threshold,
                    self.min_threshold_ms,
                )

        self.is_fitted_ = True
        return self

    def predict(self, trace: pd.DataFrame) -> pd.DataFrame:
        """Attach stage anomaly flags, thresholds and excess residuals."""

        if not self.is_fitted_:
            raise RuntimeError("Fit the anomaly detector before prediction")

        if not isinstance(trace, pd.DataFrame):
            raise TypeError("trace must be a pandas DataFrame")

        result = trace.copy()

        if self.link_conditioned_transfer:
            missing = [
                column
                for column in self.LINK_CONDITION_COLUMNS
                if column not in result.columns
            ]

            if missing:
                raise ValueError(
                    "Link-conditioned transfer prediction requires "
                    f"columns: {missing}"
                )

        for metric in STAGE_METRICS:
            threshold = self.thresholds_.get(metric, np.nan)

            threshold_col = f"{metric}_anomaly_threshold_ms"
            count_col = f"{metric}_anomaly_calibration_count"
            flag_col = f"{metric}_anomaly"
            excess_col = f"{metric}_anomaly_excess_ms"
            status_col = f"{metric}_anomaly_status"

            comparable = get_comparable_mask(result, metric)

            positive_col = f"{metric}_positive_residual_ms"

            if positive_col not in result.columns:
                positive = pd.Series(np.nan, index=result.index)
            else:
                positive = pd.to_numeric(
                    result[positive_col], errors="coerce"
                )

            valid_positive = comparable & np.isfinite(positive)

            if metric == "transfer_time_ms" and self.link_conditioned_transfer:
                keys = result[list(self.LINK_CONDITION_COLUMNS)].apply(
                    lambda row: tuple(row.tolist()), axis=1
                )

                row_thresholds = keys.map(self.link_thresholds_)
                row_counts = keys.map(self.link_sample_counts_)

                row_thresholds = pd.to_numeric(
                    row_thresholds, errors="coerce"
                )
                row_counts = pd.to_numeric(
                    row_counts, errors="coerce"
                )

                result[threshold_col] = row_thresholds
                result[count_col] = row_counts.fillna(0).astype(int)

                calibrated = np.isfinite(row_thresholds)
                valid = valid_positive & calibrated

                flags = pd.Series(
                    pd.NA, index=result.index, dtype="boolean"
                )
                flags.loc[valid] = (
                    positive.loc[valid] > row_thresholds.loc[valid]
                )

                result[flag_col] = flags
                result[excess_col] = (
                    positive - row_thresholds
                ).where(valid).clip(lower=0)

                result[status_col] = np.where(
                    ~comparable,
                    "not_comparable",
                    np.where(
                        ~calibrated,
                        "insufficient_calibration",
                        "scored",
                    ),
                )
                continue

            result[threshold_col] = threshold
            result[count_col] = self.sample_counts_.get(metric, 0)

            if not np.isfinite(threshold):
                result[flag_col] = pd.Series(
                    pd.NA, index=result.index, dtype="boolean"
                )
                result[excess_col] = np.nan
                result[status_col] = "insufficient_calibration"
                continue

            flags = pd.Series(
                pd.NA, index=result.index, dtype="boolean"
            )
            flags.loc[valid_positive] = (
                positive.loc[valid_positive] > threshold
            )

            result[flag_col] = flags
            result[excess_col] = (
                positive - threshold
            ).where(valid_positive).clip(lower=0)

            result[status_col] = np.where(
                valid_positive,
                "scored",
                "not_comparable",
            )

        return result
