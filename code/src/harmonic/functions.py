import numpy as np
import tensorflow.keras as keras
import tensorflow as tf


def gaussian(z, m):
    """Return e^{-r^2}e^{i m phi}"""
    rs = np.abs(z)
    ths = np.arctan2(z.imag, z.real)

    return np.exp(-rs**2) * np.exp(1.j * m * ths)


class HNonLinearity(keras.layers.Layer):
    """
    Maps z to non_linearity(R + b) z/|z|
    """
    def __init__(self, non_linearity, eps=1e-5):
        super(HNonLinearity, self).__init__()
        self.non_linearity = non_linearity
        self.eps = eps
        self.b = tf.Variable(0, trainable=True)

    def call(self, z, **kwargs):
        """z has shape []"""
