import numpy as np


def poison_training_labels(y_train, poison_rate=0.3, spike_factor=0.6):
    """
    Poison training labels by randomly adding +/- spike_factor
    to poison_rate percentage of samples.
    """

    y_poisoned = y_train.copy()

    n_samples = len(y_poisoned)
    n_poison = int(n_samples * poison_rate)

    indices = np.random.choice(n_samples, n_poison, replace=False)

    for idx in indices:
        direction = np.random.choice([-1, 1])
        y_poisoned[idx] = y_poisoned[idx] * (1 + direction * spike_factor)

    return y_poisoned
