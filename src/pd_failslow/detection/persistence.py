"""Detect persistent stage anomalies for individual components."""

import numpy as np
import pandas as pd

from pd_failslow.detection.preprocessing import STAGE_METRICS


STAGE_CONTEXT = {
    "prefill_service_time_ms": (
        "prefill_worker_id", "prefill_completion_ms"
    ),
    "prefill_queue_wait_ms": (
        "prefill_worker_id", "prefill_start_ms"
    ),
    "transfer_time_ms": (
        "link_id", "transfer_completion_time_ms"
    ),
    "handoff_wait_ms": (
        "decode_worker_id", "decode_admission_time_ms"
    ),
    "admission_wait_ms": (
        "decode_worker_id", "decode_execution_start_ms"
    ),
    "decode_service_time_ms": (
        "decode_worker_id", "decode_completion_ms"
    ),
}


class PersistenceDetector:
    """Require repeated anomalies within a component-specific window."""

    def __init__(self, window_size=5, min_anomalies=3):
        if window_size < 1:
            raise ValueError("window_size must be positive")

        if not 1 <= min_anomalies <= window_size:
            raise ValueError(
                "min_anomalies must be between 1 and window_size"
            )

        self.window_size = window_size
        self.min_anomalies = min_anomalies

    def predict(self, trace: pd.DataFrame) -> pd.DataFrame:
        """Attach rolling persistence decisions to stage anomaly flags."""

        if not isinstance(trace, pd.DataFrame):
            raise TypeError("trace must be a pandas DataFrame")

        result = trace.copy()

        for metric in STAGE_METRICS:
            flag_col = f"{metric}_anomaly"
            anomaly_status_col = f"{metric}_anomaly_status"

            persistent_col = f"{metric}_persistent_anomaly"
            count_col = f"{metric}_persistence_count"
            status_col = f"{metric}_persistence_status"

            flags = pd.Series(
                pd.NA, index=result.index, dtype="boolean"
            )
            counts = np.full(len(result), np.nan)
            statuses = np.full(
                len(result), "not_scored", dtype=object
            )

            component_col, time_col = STAGE_CONTEXT[metric]

            if flag_col not in result.columns:
                statuses[:] = "missing_anomaly_input"
                result[persistent_col] = flags
                result[count_col] = counts
                result[status_col] = statuses
                continue

            if component_col not in result.columns:
                statuses[:] = "missing_component"
                result[persistent_col] = flags
                result[count_col] = counts
                result[status_col] = statuses
                continue

            if time_col not in result.columns:
                statuses[:] = "missing_timestamp"
                result[persistent_col] = flags
                result[count_col] = counts
                result[status_col] = statuses
                continue

            anomaly_status = (
                result[anomaly_status_col]
                if anomaly_status_col in result.columns
                else pd.Series(
                    "scored", index=result.index
                )
            )

            raw_flags = result[flag_col]

            flag_values = raw_flags.astype("boolean")
            timestamps = pd.to_numeric(
                result[time_col], errors="coerce"
            )

            components = result[component_col]

            for pos in range(len(result)):
                if anomaly_status.iloc[pos] != "scored":
                    statuses[pos] = (
                        str(anomaly_status.iloc[pos])
                    )
                elif pd.isna(components.iloc[pos]):
                    statuses[pos] = "missing_component"
                elif pd.isna(timestamps.iloc[pos]):
                    statuses[pos] = "missing_timestamp"
                elif pd.isna(flag_values.iloc[pos]):
                    statuses[pos] = "missing_anomaly_flag"
                else:
                    statuses[pos] = "insufficient_history"

            work = pd.DataFrame({
                "_position": np.arange(len(result)),
                "_component": components.to_numpy(),
                "_timestamp": timestamps.to_numpy(),
                "_flag": flag_values.to_numpy(
                    dtype=object, na_value=np.nan
                ),
                "_status": anomaly_status.to_numpy(),
            })

            eligible = work[
                work["_status"].eq("scored")
                & work["_component"].notna()
                & work["_timestamp"].notna()
                & work["_flag"].notna()
            ]

            for _, group in eligible.groupby(
                "_component", sort=False
            ):
                group = group.sort_values(
                    ["_timestamp", "_position"],
                    kind="stable",
                )

                positions = group["_position"].to_numpy()
                values = group["_flag"].astype(bool).astype(int)

                rolling_counts = values.rolling(
                    self.window_size,
                    min_periods=self.window_size,
                ).sum()

                for position, count in zip(
                    positions, rolling_counts
                ):
                    if pd.isna(count):
                        continue

                    counts[position] = int(count)
                    flags.iloc[position] = (
                        count >= self.min_anomalies
                    )
                    statuses[position] = "evaluated"

            result[persistent_col] = flags
            result[count_col] = counts
            result[status_col] = statuses

        return result


class TimeAwarePersistenceDetector(PersistenceDetector):
    """Require repeated anomalies within a bounded elapsed-time window."""

    def __init__(
        self,
        window_size=5,
        min_anomalies=3,
        max_window_duration_ms=250.0,
    ):
        super().__init__(
            window_size=window_size,
            min_anomalies=min_anomalies,
        )

        if (
            not np.isfinite(max_window_duration_ms)
            or max_window_duration_ms < 0
        ):
            raise ValueError(
                "max_window_duration_ms must be finite and non-negative"
            )

        self.max_window_duration_ms = float(
            max_window_duration_ms
        )

    def predict(self, trace: pd.DataFrame) -> pd.DataFrame:
        """Attach time-aware persistence decisions to stage anomalies."""

        result = super().predict(trace)

        for metric in STAGE_METRICS:
            component_col, time_col = STAGE_CONTEXT[metric]

            duration_col = (
                f"{metric}_persistence_window_duration_ms"
            )

            duration_values = np.full(
                len(result), np.nan, dtype=float
            )

            flag_col = f"{metric}_anomaly"
            status_col = f"{metric}_anomaly_status"

            if (
                flag_col not in result.columns
                or component_col not in result.columns
                or time_col not in result.columns
            ):
                result[duration_col] = duration_values
                continue

            anomaly_status = (
                result[status_col]
                if status_col in result.columns
                else pd.Series("scored", index=result.index)
            )

            timestamps = pd.to_numeric(
                result[time_col], errors="coerce"
            )

            work = pd.DataFrame({
                "_position": np.arange(len(result)),
                "_component": result[component_col].to_numpy(),
                "_timestamp": timestamps.to_numpy(),
                "_status": anomaly_status.to_numpy(),
            })

            eligible = work[
                work["_status"].eq("scored")
                & work["_component"].notna()
                & work["_timestamp"].notna()
                & result[flag_col].notna().to_numpy()
            ]

            for _, group in eligible.groupby(
                "_component", sort=False
            ):
                group = group.sort_values(
                    ["_timestamp", "_position"],
                    kind="stable",
                )

                positions = group["_position"].to_numpy()
                times = group["_timestamp"].to_numpy()

                if len(group) < self.window_size:
                    continue

                for end in range(
                    self.window_size - 1, len(group)
                ):
                    start = end - self.window_size + 1
                    duration = times[end] - times[start]

                    duration_values[positions[end]] = duration

            result[duration_col] = duration_values

            persistent_col = (
                f"{metric}_persistent_anomaly"
            )

            evaluated = result[
                f"{metric}_persistence_status"
            ].eq("evaluated")

            within_limit = (
                result[duration_col]
                <= self.max_window_duration_ms
            )

            result.loc[
                evaluated & ~within_limit,
                persistent_col,
            ] = False

        return result