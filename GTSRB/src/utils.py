import numpy as np


def l2_norm_of_perturbation(x):
    """
    x: ndarray[width][height][3], x in [0,1]
    """
    return np.linalg.norm(x)

