"""Shared evaluation metrics for fail-slow detection."""

import numpy as np


def binary_classification_metrics(
    y_true,
    y_pred,
    zero_division=0.0,
):
    """
    Calculate binary classification metrics.

    Parameters
    ----------
    y_true : iterable
        Ground-truth binary labels.
    y_pred : iterable
        Predicted binary labels.
    zero_division : float
        Value used when a metric denominator is zero.

    Returns
    -------
    dict
        Confusion counts and standard classification metrics.
    """
    if zero_division not in (0, 1, 0.0, 1.0):
        raise ValueError("zero_division must be 0 or 1")

    def convert(values, name):
        array = np.asarray(list(values))

        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")

        if not np.isin(array, [0, 1, False, True]).all():
            raise ValueError(f"{name} must contain only binary labels")

        return array.astype(bool)

    true = convert(y_true, "y_true")
    pred = convert(y_pred, "y_pred")

    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have equal lengths")

    tp = int(np.sum(true & pred))
    fp = int(np.sum(~true & pred))
    tn = int(np.sum(~true & ~pred))
    fn = int(np.sum(true & ~pred))

    def divide(numerator, denominator):
        if denominator == 0:
            return float(zero_division)
        return float(numerator / denominator)

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    f1 = divide(
        2 * precision * recall,
        precision + recall,
    )
    accuracy = divide(tp + tn, tp + fp + tn + fn)
    false_positive_rate = divide(fp, fp + tn)

    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "accuracy": accuracy,
        "sample_count": len(true),
    }
