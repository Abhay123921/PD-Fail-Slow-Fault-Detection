import pytest

from pd_failslow.evaluation.metrices import binary_classification_metrics


def test_binary_classification_metrics():
    result = binary_classification_metrics(
        [1, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
    )

    assert result["true_positive"] == 2
    assert result["false_positive"] == 1
    assert result["true_negative"] == 2
    assert result["false_negative"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1_score"] == pytest.approx(2 / 3)
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_perfect_classification():
    result = binary_classification_metrics([1, 0], [1, 0])

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["specificity"] == 1.0


def test_zero_division_handling():
    result = binary_classification_metrics([0, 0], [0, 0])

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1_score"] == 0.0
    assert result["specificity"] == 1.0


def test_mismatched_lengths_raise_error():
    with pytest.raises(ValueError):
        binary_classification_metrics([1, 0], [1])


def test_invalid_labels_raise_error():
    with pytest.raises(ValueError):
        binary_classification_metrics([1, 2], [1, 0])


def test_invalid_zero_division_raises_error():
    with pytest.raises(ValueError):
        binary_classification_metrics([1], [1], zero_division=0.5)
