import numpy as np
from sklearn.metrics import average_precision_score, recall_score
from sklearn.metrics import brier_score_loss

def pr_auc(y_true, y_prob) -> float:
    return float(average_precision_score(y_true, y_prob))

def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(y_true, y_prob))

def pick_threshold_by_alert_budget(y_prob: np.ndarray, budget: float = 0.20) -> float:
    if not 0 < budget < 1:
        raise ValueError("budget must be between 0 and 1")
    return float(np.quantile(y_prob, 1 - budget))


def recall_at_threshold(y_true, y_prob, threshold: float) -> float:
    """
    Crisis recall at a fixed alert threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return float(recall_score(y_true, y_pred))
