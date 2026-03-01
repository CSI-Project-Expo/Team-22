# Import numpy library for numerical operations (arrays, math, etc.)
import numpy as np


# Define a function named detect_and_restore
# y_clean = original correct values
# y_poisoned = values that may contain errors or attacks
# threshold=0.3 means maximum allowed difference
def detect_and_restore(y_clean, y_poisoned, threshold=0.3):

    """
    Detect poisoned labels based on deviation from clean labels.
    Restore values if deviation exceeds threshold.
    """

    # Create a copy of poisoned data to avoid changing original data
    y_defended = y_poisoned.copy()

    # Calculate deviation (difference) between poisoned and clean values
    # np.abs() converts negative differences into positive values
    deviation = np.abs(y_poisoned - y_clean)

    # Create a mask (True/False array)
    # True if deviation is greater than threshold (means value is poisoned)
    mask = deviation > threshold

    # Restore correct values where deviation is greater than threshold
    # Replace poisoned values with clean values
    y_defended[mask] = y_clean[mask]

    # Return the defended (restored) data
    return y_defended