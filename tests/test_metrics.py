import numpy as np
from src.common.metrics import expected_calibration_error, classification_report_dict


def test_ece_in_range():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.2, 0.9])
    ece, _ = expected_calibration_error(y, p, n_bins=2)
    assert 0 <= ece <= 1


def test_classification_report_has_keys():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.2, 0.9])
    r = classification_report_dict(y, p)
    assert {"roc_auc", "pr_auc", "log_loss", "brier", "ece"}.issubset(r.keys())
