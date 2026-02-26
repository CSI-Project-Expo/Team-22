import numpy as np


def detect_and_restore(y_clean, y_poisoned, threshold=0.3):
    """
    Detect poisoned labels based on deviation from clean labels.
    Restore values if deviation exceeds threshold.
    """

    y_defended = y_poisoned.copy()

    deviation = np.abs(y_poisoned - y_clean)

    mask = deviation > threshold

    y_defended[mask] = y_clean[mask]

    return y_defended