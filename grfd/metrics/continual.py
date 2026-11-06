from __future__ import annotations

import numpy as np


def continual_statistics(matrix: np.ndarray, initial: np.ndarray) -> dict:
    """AIA/AF/BWT/FWT/FAA where matrix[t, j] is probe j accuracy after experience t+1 and initial[j] is before training."""
    k = matrix.shape[0]
    diag = np.diag(matrix)
    aia = np.mean([matrix[t, : t + 1].mean() for t in range(k)])
    faa = np.mean(matrix[-1])
    if k > 1:
        af = np.mean([max(0.0, np.max(matrix[j:, j]) - matrix[-1, j]) for j in range(k - 1)])
        bwt = np.mean([matrix[-1, j] - diag[j] for j in range(k - 1)])
        fwt = np.mean([matrix[j - 1, j] - initial[j] for j in range(1, k)])
    else:
        af = bwt = fwt = 0.0
    return {
        "AIA_probe": float(aia),
        "AF_probe": float(af),
        "BWT_probe": float(bwt),
        "FWT_probe": float(fwt),
        "FAA_probe": float(faa),
    }
