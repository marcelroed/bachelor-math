import numpy as np


def gaussian(z, m):
    """Return e^{-r^2}e^{i m phi}"""
    rs = np.abs(z)
    ths = np.arctan2(z.imag, z.real)

    return np.exp(-rs**2) * np.exp(1.j * m * ths)

