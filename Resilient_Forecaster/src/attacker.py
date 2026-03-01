import numpy as np  # Import NumPy library and give it the short name 'np'

def poison_training_labels(y_train, poison_rate=0.3, spike_factor=0.6):
    """
    Poison training labels by randomly adding +/- spike_factor
    to poison_rate percentage of samples.
    """
    # Function definition:
    # y_train → original training labels (array)
    # poison_rate → percentage of labels to modify (default 30%)
    # spike_factor → amount of increase or decrease (default 60%)

    y_poisoned = y_train.copy()
    # Create a copy of y_train so the original data is not modified

    n_samples = len(y_poisoned)
    # Get total number of samples in the dataset

    n_poison = int(n_samples * poison_rate)
    # Calculate how many samples should be poisoned
    # Example: if 100 samples and poison_rate=0.3 → 30 samples will be poisoned

    indices = np.random.choice(n_samples, n_poison, replace=False)
    # Randomly select 'n_poison' unique indices from total samples
    # replace=False ensures no index is selected twice

    for idx in indices:
        # Loop through each randomly selected index

        direction = np.random.choice([-1, 1])
        # Randomly choose either -1 or +1
        # -1 → decrease value
        # +1 → increase value

        y_poisoned[idx] = y_poisoned[idx] * (1 + direction * spike_factor)
        # Modify the selected label:
        # If direction = 1 → value increases by spike_factor (e.g., +60%)
        # If direction = -1 → value decreases by spike_factor (e.g., -60%)

    return y_poisoned
    # Return the modified (poisoned) labels