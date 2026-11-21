"""Contention-aware analytical diagnostics for KV-cache transfers."""

import numpy as np
import pandas as pd


class CommunicationDiagnostic:
    """Estimate expected transfer time from configured link behaviour.

    Shared links require approximately constant concurrency for an exact
    analytical estimate. Variable-concurrency cases abstain.
    """

    def __init__(self, system, concurrency_tolerance=0.05):
        if concurrency_tolerance < 0:
            raise ValueError("concurrency_tolerance must be non-negative")

        self.concurrency_tolerance = concurrency_tolerance
        self.links = {
            link.link_id: {
                "bandwidth": float(link.bandwidth_mb_per_ms),
                "fixed_latency": float(link.fixed_latency_ms),
                "shared": bool(link.shared),
            }
            for link in system.communication_links
        }

    @staticmethod
    def _number(value):
        try:
            number = float(value)
            return number if np.isfinite(number) else None
        except (TypeError, ValueError):
            return None

    def predict(self, trace):
        """Attach analytical communication estimates to telemetry."""
        if not isinstance(trace, pd.DataFrame):
            raise TypeError("trace must be a pandas DataFrame")

        result = trace.copy()

        expected_times = []
        residuals = []
        concurrency_stability = []
        statuses = []

        for _, row in result.iterrows():
            link_id = row.get("link_id")
            link = self.links.get(link_id)

            expected = np.nan
            residual = np.nan
            stable = None

            if link is None:
                status = "unknown_link"

            else:
                payload = self._number(row.get("kv_cache_size_mb"))
                actual = self._number(row.get("transfer_time_ms"))

                if payload is None or payload <= 0 or actual is None:
                    status = "insufficient_data"

                else:
                    recorded_shared = row.get("link_shared")
                    recorded_bw = self._number(
                        row.get("configured_bandwidth_mb_per_ms")
                    )

                    config_mismatch = False

                    if pd.notna(recorded_shared):
                        config_mismatch |= (
                            bool(recorded_shared) != link["shared"]
                        )

                    if recorded_bw is not None:
                        config_mismatch |= not np.isclose(
                            recorded_bw,
                            link["bandwidth"],
                            rtol=1e-6,
                            atol=1e-9,
                        )

                    if config_mismatch:
                        status = "configuration_mismatch"

                    elif not link["shared"]:
                        expected = (
                            link["fixed_latency"]
                            + payload / link["bandwidth"]
                        )
                        residual = actual - expected
                        stable = None
                        status = "estimated"

                    else:
                        concurrency = self._number(
                            row.get("concurrent_transfers")
                        )
                        average = self._number(
                            row.get("average_concurrent_transfers")
                        )
                        peak = self._number(
                            row.get("peak_concurrent_transfers")
                        )

                        values = (concurrency, average, peak)

                        if any(
                            value is None or value < 1
                            for value in values
                        ):
                            status = "insufficient_data"

                        else:
                            stable = all(
                                np.isclose(
                                    value,
                                    concurrency,
                                    rtol=0.01,
                                    atol=self.concurrency_tolerance,
                                )
                                for value in (average, peak)
                            )

                            if not stable:
                                status = "variable_concurrency"

                            else:
                                expected = (
                                    link["fixed_latency"]
                                    + payload * concurrency
                                    / link["bandwidth"]
                                )
                                residual = actual - expected
                                status = "estimated"

            expected_times.append(expected)
            residuals.append(residual)
            concurrency_stability.append(stable)
            statuses.append(status)

        result["communication_expected_time_ms"] = expected_times
        result["communication_residual_ms"] = residuals
        result["communication_concurrency_stable"] = (
            concurrency_stability
        )
        result["communication_diagnostic_status"] = statuses

        return result
