from pathlib import Path

from pd_failslow.telemetry.schemas import (
    build_trace_row,
    build_ground_truth_row,
    create_trace_dataframe,
    create_ground_truth_dataframe,
    validate_trace_dataframe,
)


class TelemetryCollector:
    """
    Collects simulator observations and ground-truth labels.

    Important:
        The detector should receive only the trace DataFrame.

        Ground-truth labels are stored separately and are used
        exclusively for evaluation.
    """

    def __init__(self):
        self.trace_rows = []
        self.ground_truth_rows = []

    def record_request(
        self,
        request,
        metadata=None,
    ):
        row = build_trace_row(
            request=request,
            metadata=metadata or {},
        )

        self.trace_rows.append(row)

    def record_fault(
        self,
        fault_id,
        fault_type,
        component_type,
        component_id,
        start_time_ms,
        end_time_ms,
        severity,
    ):
        row = build_ground_truth_row(
            fault_id=fault_id,
            fault_type=fault_type,
            component_type=component_type,
            component_id=component_id,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            severity=severity,
        )

        self.ground_truth_rows.append(row)

    def get_trace_dataframe(self):
        dataframe = create_trace_dataframe(
            self.trace_rows
        )

        validate_trace_dataframe(dataframe)

        return dataframe

    def get_ground_truth_dataframe(self):
        return create_ground_truth_dataframe(
            self.ground_truth_rows
        )

    def save(
        self,
        trace_path,
        ground_truth_path=None,
    ):
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        trace_df = self.get_trace_dataframe()

        trace_df.to_csv(
            trace_path,
            index=False,
        )

        if ground_truth_path is not None:
            ground_truth_path = Path(ground_truth_path)
            ground_truth_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            ground_truth_df = self.get_ground_truth_dataframe()

            ground_truth_df.to_csv(
                ground_truth_path,
                index=False,
            )

        return trace_df

    def reset(self):
        self.trace_rows.clear()
        self.ground_truth_rows.clear()