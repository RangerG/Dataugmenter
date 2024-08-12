# This is for DA method classification of "Decomposition-Based Methods"

import numpy as np
from PyEMD import EMD


def emd_augmentation(x):
    """
        Perform data augmentation using Empirical Mode Decomposition (EMD).
        Parameters:
        x: Input data, shape (number of samples, time steps)

        Returns:
        augmented_data: Augmented data, shape (number of samples, time steps)
    """
    emd = EMD()
    augmented_data = np.empty_like(x)

    for i in range(x.shape[0]):
        signal = x[i].flatten()  # Flatten the signal into a one-dimensional array
        time_array = np.arange(len(signal))  # Ensure that the time array matches the signal length
        imfs = emd(signal, time_array)
        if imfs.size > 0:
            selected_imfs = imfs[:2]  # Select the first two IMFs
            reconstructed_signal = np.sum(selected_imfs, axis=0)
            augmented_data[i] = reconstructed_signal.reshape(-1, 1)
        else:
            augmented_data[i] = signal.reshape(-1, 1)

    return augmented_data
