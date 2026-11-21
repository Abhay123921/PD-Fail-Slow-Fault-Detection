import pandas as pd
import pytest

from pd_failslow.diagnosis.localization import rank_localized_components


def test_components_ranked_by_evidence_count():
    candidates = pd.DataFrame({
        "fault_type": ["communication"] * 4,
        "component_type": ["link"] * 4,
        "component_id": ["L1", "L1", "L1", "L2"],
        "evidence_role": ["direct"] * 4,
        "stage": ["transfer_time_ms"] * 4,
        "positive_residual_ms": [2.0, 3.0, 4.0, 5.0],
        "anomaly_excess_ms": [1.0, 2.0, 3.0, 4.0],
        "evidence_quality_score": [0.8, 0.9, 0.9, 0.7],
        "timestamp_ms": [10.0, 20.0, 30.0, 40.0],
    })

    result = rank_localized_components(candidates)

    assert list(result["component_id"]) == ["L1", "L2"]
    assert list(result["localization_rank"]) == [1, 2]
    assert list(result["evidence_count"]) == [3, 1]


def test_aggregation_computes_stage_and_evidence_statistics():
    candidates = pd.DataFrame({
        "fault_type": ["compute", "compute"],
        "component_type": ["worker", "worker"],
        "component_id": ["P1", "P1"],
        "evidence_role": ["direct", "direct"],
        "stage": ["prefill_service_time_ms", "prefill_service_time_ms"],
        "positive_residual_ms": [2.0, 4.0],
        "anomaly_excess_ms": [1.0, 3.0],
        "evidence_quality_score": [0.6, 1.0],
        "timestamp_ms": [10.0, 20.0],
    })

    result = rank_localized_components(candidates)
    row = result.iloc[0]

    assert row["evidence_count"] == 2
    assert row["stage_count"] == 1
    assert row["affected_stages"] == ["prefill_service_time_ms"]
    assert row["mean_positive_residual_ms"] == pytest.approx(3.0)
    assert row["mean_anomaly_excess_ms"] == pytest.approx(2.0)
    assert row["mean_evidence_quality_score"] == pytest.approx(0.8)
    assert row["last_seen_ms"] == 20.0


def test_context_only_evidence_is_excluded():
    candidates = pd.DataFrame({
        "fault_type": ["compute", None],
        "component_type": ["worker", None],
        "component_id": ["P1", None],
        "evidence_role": ["direct", "context_only"],
    })

    result = rank_localized_components(candidates)

    assert len(result) == 1
    assert result.iloc[0]["component_id"] == "P1"


def test_rankings_are_separate_for_each_fault_type():
    candidates = pd.DataFrame({
        "fault_type": ["compute", "compute", "communication"],
        "component_type": ["worker", "worker", "link"],
        "component_id": ["P1", "P2", "L1"],
        "evidence_role": ["direct"] * 3,
        "stage": ["prefill_service_time_ms", "prefill_service_time_ms",
                  "transfer_time_ms"],
        "anomaly_excess_ms": [3.0, 1.0, 10.0],
    })

    result = rank_localized_components(candidates)

    compute = result[result["fault_type"] == "compute"]
    communication = result[result["fault_type"] == "communication"]

    assert list(compute["component_id"]) == ["P1", "P2"]
    assert list(compute["localization_rank"]) == [1, 2]
    assert list(communication["component_id"]) == ["L1"]
    assert list(communication["localization_rank"]) == [1]


def test_minimum_evidence_filter():
    candidates = pd.DataFrame({
        "fault_type": ["compute", "compute", "compute"],
        "component_type": ["worker"] * 3,
        "component_id": ["P1", "P1", "P2"],
        "evidence_role": ["direct"] * 3,
    })

    result = rank_localized_components(candidates, min_evidence=2)

    assert list(result["component_id"]) == ["P1"]
    assert result.iloc[0]["evidence_count"] == 2


def test_empty_candidates_return_empty_result():
    result = rank_localized_components(pd.DataFrame())

    assert result.empty
    assert "localization_rank" in result.columns


def test_invalid_minimum_evidence_raises():
    with pytest.raises(ValueError):
        rank_localized_components(pd.DataFrame(), min_evidence=0)


def test_missing_required_column_raises():
    candidates = pd.DataFrame({"component_id": ["P1"]})

    with pytest.raises(ValueError):
        rank_localized_components(candidates)
