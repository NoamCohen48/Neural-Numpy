import numpy as np


def int_to_one_hot(x: np.ndarray):
    b = np.zeros((x.size, x.max() + 1))
    b[np.arange(x.size), x] = 1
    return b
