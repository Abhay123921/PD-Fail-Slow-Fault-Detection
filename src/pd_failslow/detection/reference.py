"""Workload-conditioned robust references for healthy PD inference traces."""

from __future__ import annotations

import numpy as np
import pandas as pd


STAGE_GROUPS = {
    "prefill_service_time_ms": (
        "prompt_bucket",
        "prefill_worker_class",
    ),
    "prefill_queue_wait_ms": (
        "prompt_bucket",
        "prefill_queue_depth",
    ),
    "transfer_time_ms": (
        "kv_cache_bucket",
        "link_type",
        "configured_bandwidth_mb_per_ms",
        "link_shared",
    ),
    "handoff_wait_ms": (
        "decode_worker_class",
    ),
    "admission_wait_ms": (
        "output_bucket",
        "decode_queue_depth",
    ),
    "decode_service_time_ms": (
        "output_bucket",
        "decode_worker_class",
    ),
}

OOD_FEATURES = {
    "prefill_service_time_ms": ("prompt_tokens",),
    "prefill_queue_wait_ms": ("prompt_tokens",),
    "transfer_time_ms": (
        "kv_cache_size_mb",
        "configured_bandwidth_mb_per_ms",
    ),
    "admission_wait_ms": ("output_tokens",),
    "decode_service_time_ms": ("output_tokens",),
}

NUMERIC_BUCKETS = {
    "prompt_bucket": "prompt_tokens",
    "output_bucket": "output_tokens",
    "kv_cache_bucket": "kv_cache_size_mb",
}


class HealthyReferenceModel:
    """Estimate normal stage latency using workload-conditioned medians.

    Groups with insufficient samples use a stage-wide healthy reference.
    If even the stage-wide reference is insufficient, prediction abstains.
    """

    def __init__(self, min_samples: int = 20, workload_conditioned: bool = True):
        if min_samples < 1:
            raise ValueError("min_samples must be positive")

        self.min_samples = min_samples
        self.workload_conditioned = bool(workload_conditioned)
        self.worker_equivalence_enabled = True
        self.is_fitted_ = False
        self.feature_columns_ = []
        self.bucket_edges_ = {}
        self.stage_models_ = {}
        self.training_ranges_ = {}

    @staticmethod
    def _worker_class(value, prefix):
        if pd.isna(value):
            return "__missing__"

        # P1/P2 and D1/D2 have identical configured capacities
        # in the current simulator. Keep unknown IDs distinct.
        if value in ("P1", "P2") and prefix == "prefill":
            return "prefill_equivalent"
        if value in ("D1", "D2") and prefix == "decode":
            return "decode_equivalent"

        return str(value)

    @staticmethod
    def _normalise(value):
        if pd.isna(value):
            return "__missing__"
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _fit_bucket_edges(self, trace):
        for bucket_name, source_column in NUMERIC_BUCKETS.items():
            if source_column not in trace.columns:
                self.bucket_edges_[bucket_name] = np.array([])
                continue

            values = pd.to_numeric(
                trace[source_column], errors="coerce"
            ).dropna()

            if values.empty:
                edges = np.array([])
            else:
                edges = np.unique(
                    np.quantile(values, [0.25, 0.50, 0.75])
                )

            self.bucket_edges_[bucket_name] = edges

    def _prepare_features(self, trace):
        data = trace.copy()

        for bucket_name, source_column in NUMERIC_BUCKETS.items():
            if source_column not in data.columns:
                data[bucket_name] = "__missing__"
                continue

            values = pd.to_numeric(
                data[source_column], errors="coerce"
            )

            edges = self.bucket_edges_.get(
                bucket_name, np.array([])
            )

            buckets = np.full(len(data), "__missing__", dtype=object)
            valid = values.notna()

            if valid.any():
                buckets[valid.to_numpy()] = np.searchsorted(
                    edges,
                    values[valid].to_numpy(),
                    side="right",
                ).astype(str)

            data[bucket_name] = buckets

        if "prefill_worker_id" in data:
            data["prefill_worker_class"] = data[
                "prefill_worker_id"
            ].map(
                lambda x: self._worker_class(x, "prefill")
            )
        else:
            data["prefill_worker_class"] = "__missing__"

        if "decode_worker_id" in data:
            data["decode_worker_class"] = data[
                "decode_worker_id"
            ].map(
                lambda x: self._worker_class(x, "decode")
            )
        else:
            data["decode_worker_class"] = "__missing__"

        return data

    @staticmethod
    def _group_key(row, columns):
        return tuple(
            HealthyReferenceModel._normalise(row[column])
            for column in columns
        )

    @staticmethod
    def _statistics(values):
        values = pd.to_numeric(
            values, errors="coerce"
        ).dropna()

        if values.empty:
            return {
                "median": np.nan,
                "mad": np.nan,
                "count": 0,
            }

        median = float(values.median())
        mad = float((values - median).abs().median())

        return {
            "median": median,
            "mad": mad,
            "count": int(len(values)),
        }

    def fit(self, healthy_trace: pd.DataFrame):
        """Fit stage references using healthy telemetry only."""

        if (
            healthy_trace is None
            or not isinstance(healthy_trace, pd.DataFrame)
            or healthy_trace.empty
        ):
            raise ValueError("Healthy training data cannot be empty")

        self.feature_columns_ = [
            "prompt_tokens",
            "output_tokens",
            "kv_cache_size_mb",
            "prefill_worker_id",
            "decode_worker_id",
            "prefill_queue_depth",
            "decode_queue_depth",
            "link_type",
            "configured_bandwidth_mb_per_ms",
            "link_shared",
        ]

        self._fit_bucket_edges(healthy_trace)

        self.training_ranges_ = {}
        for column in (
            "prompt_tokens",
            "output_tokens",
            "kv_cache_size_mb",
            "configured_bandwidth_mb_per_ms",
        ):
            if column not in healthy_trace.columns:
                continue

            values = pd.to_numeric(
                healthy_trace[column], errors="coerce"
            ).dropna()

            if not values.empty:
                self.training_ranges_[column] = (
                    float(values.min()),
                    float(values.max()),
                )
        data = self._prepare_features(healthy_trace)
        self.stage_models_ = {}

        for metric, configured_group_columns in STAGE_GROUPS.items():
            group_columns = (
                configured_group_columns if self.workload_conditioned else ()
            )
            if metric not in data.columns:
                continue

            valid = data[metric].notna()
            stage_data = data.loc[valid].copy()

            global_stats = self._statistics(stage_data[metric])
            group_stats = {}

            if not stage_data.empty and group_columns:
                grouped = stage_data.groupby(
                    list(group_columns),
                    dropna=False,
                    sort=False,
                )

                for key, group in grouped:
                    if not isinstance(key, tuple):
                        key = (key,)

                    group_stats[key] = self._statistics(
                        group[metric]
                    )

            self.stage_models_[metric] = {
                "group_columns": group_columns,
                "global": global_stats,
                "groups": group_stats,
            }

        if not self.stage_models_:
            raise ValueError(
                "Training data contains no supported stage metrics"
            )

        self.is_fitted_ = True
        return self

    def predict(self, trace: pd.DataFrame) -> pd.DataFrame:
        """Attach healthy reference statistics to each request."""

        if not self.is_fitted_:
            raise RuntimeError("Fit the reference model before prediction")

        if not isinstance(trace, pd.DataFrame):
            raise TypeError("trace must be a pandas DataFrame")

        result = trace.copy()
        data = self._prepare_features(trace)

        for metric, model in self.stage_models_.items():
            group_columns = model["group_columns"]
            global_stats = model["global"]
            groups = model["groups"]

            references = []
            mads = []
            counts = []
            statuses = []

            for _, row in data.iterrows():
                key = self._group_key(row, group_columns)

                # Do not extrapolate healthy references beyond
                # the observed training range.
                ood_features = OOD_FEATURES.get(metric, ())
                is_ood = False

                for column in ood_features:
                    if column not in self.training_ranges_:
                        continue

                    value = pd.to_numeric(
                        pd.Series([row.get(column)]),
                        errors="coerce",
                    ).iloc[0]

                    if pd.isna(value):
                        continue

                    lower, upper = self.training_ranges_[column]
                    if value < lower or value > upper:
                        is_ood = True
                        break

                stats = groups.get(key)

                if is_ood:
                    selected = {
                        "median": np.nan,
                        "mad": np.nan,
                        "count": 0,
                    }
                    status = "out_of_distribution"

                elif not self.workload_conditioned:
                    if global_stats["count"] >= self.min_samples:
                        selected = global_stats
                        status = "ok"
                    else:
                        selected = {
                            "median": np.nan,
                            "mad": np.nan,
                            "count": global_stats["count"],
                        }
                        status = "insufficient_reference"
                elif stats is not None and stats["count"] >= self.min_samples:
                    selected = stats
                    status = "ok"

                elif global_stats["count"] >= self.min_samples:
                    selected = global_stats
                    status = "fallback_global"

                else:
                    selected = {
                        "median": np.nan,
                        "mad": np.nan,
                        "count": global_stats["count"],
                    }
                    status = "insufficient_reference"

                references.append(selected["median"])
                mads.append(selected["mad"])
                counts.append(selected["count"])
                statuses.append(status)

            result[f"{metric}_reference"] = references
            result[f"{metric}_mad"] = mads
            result[f"{metric}_sample_count"] = counts
            result[f"{metric}_status"] = statuses

        return result
