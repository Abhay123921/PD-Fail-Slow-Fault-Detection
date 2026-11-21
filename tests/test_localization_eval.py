import pandas as pd
import pytest

from pd_failslow.evaluation.localization_eval import evaluate_localization


def make_ground_truth():
    return pd.DataFrame([{
        "fault_type": "compute",
        "component_type": "worker",
        "component_id": "P1",
    }])


def make_ranking(component_ids):
    return pd.DataFrame([
        {
            "fault_type": "compute",
            "component_type": "worker",
            "component_id": component_id,
            "localization_rank": rank,
        }
        for rank, component_id in enumerate(component_ids, start=1)
    ])


def test_correct_top_1_localization():
    result = evaluate_localization(
        make_ranking(["P1", "P2", "P3"]),
        make_ground_truth(),
    )

    metrics = result["metrics"]
    assert metrics["component_top_1_accuracy"] == 1.0
    assert metrics["component_top_3_accuracy"] == 1.0
    assert metrics["mean_reciprocal_rank"] == 1.0


def test_correct_component_ranked_third():
    result = evaluate_localization(
        make_ranking(["P2", "P3", "P1"]),
        make_ground_truth(),
    )

    metrics = result["metrics"]
    assert metrics["component_top_1_accuracy"] == 0.0
    assert metrics["component_top_3_accuracy"] == 1.0
    assert metrics["mean_reciprocal_rank"] == pytest.approx(1 / 3)


def test_wrong_component():
    result = evaluate_localization(
        make_ranking(["P2", "P3"]),
        make_ground_truth(),
    )

    metrics = result["metrics"]
    assert metrics["fault_type_accuracy"] == 1.0
    assert metrics["component_top_1_accuracy"] == 0.0
    assert metrics["component_top_3_accuracy"] == 0.0
    assert metrics["mean_reciprocal_rank"] == 0.0


def test_missing_fault_type():
    ranking = pd.DataFrame(columns=[
        "fault_type", "component_type", "component_id",
        "localization_rank",
    ])

    result = evaluate_localization(ranking, make_ground_truth())

    assert result["metrics"]["fault_type_accuracy"] == 0.0
    assert result["metrics"]["component_top_1_accuracy"] == 0.0


def test_empty_ground_truth():
    gt = pd.DataFrame(columns=[
        "fault_type", "component_type", "component_id"
    ])

    result = evaluate_localization(
        make_ranking(["P1"]),
        gt,
    )

    assert result["metrics"]["fault_count"] == 0
    assert result["metrics"]["mean_reciprocal_rank"] == 0.0


def test_invalid_rank_raises_error():
    ranking = make_ranking(["P1"])
    ranking.loc[0, "localization_rank"] = 0

    with pytest.raises(ValueError):
        evaluate_localization(ranking, make_ground_truth())


def test_invalid_top_k_raises_error():
    with pytest.raises(ValueError):
        evaluate_localization(
            make_ranking(["P1"]),
            make_ground_truth(),
            top_k=(1, 0),
        )
