import tensorflow as tf
import tensorflow.keras as keras
from tensorflow import complex64 as c64, float32 as f32
import numpy as np

from typing import Optional


class Conv2DH(keras.layers.Layer):
    def __init__(self, k_size=5, out_orders=(0, ), out_channels=1, fourier_depth=5, **kwargs):
        super(Conv2DH, self).__init__(**kwargs)

        self.k_size = k_size
        self.output_orders = out_orders
        self.out_channels = out_channels
        self.fourier_depth = fourier_depth
        self.fourier_weights: Optional[tf.Variable] = None
        self.in_orders, self.in_channels = None, None

    def build(self, input_shape):
        width, height, self.in_channels, self.in_orders = input_shape
        # Initialize weights
        weight_shape = (
            self.k_size, self.k_size, self.in_channels, self.out_channels, self.in_orders, self.out_orders
        )

        self.fourier_weights = self.add_weight(name='fourier_weights',
                                               shape=weight_shape,
                                               initializer='glorot_normal',
                                               trainable=True)
        super(Conv2DH, self).build(input_shape)

    def call(self, x, **kwargs):
        filters = self.generate_filters()

        convolved = tf.nn.conv2d(x, filters, strides=(1, 1), padding='same')
        return convolved

    def compute_output_shape(self, input_shape):
        # Previous sizes
        width, height, in_channels, in_orders = input_shape

        return width, height, self.out_channels, len(self.out_orders)

    def generate_filters(self):
        # Dictionary for filters from input order to output order
        filter_dict = {}

        for in_m in range(self.in_orders):
            in_dict = {}
            for out_m in range(self.output_orders):
                in_dict[out_m] = self.fourier_weights

            filter_dict[in_m] = in_dict

    def filter_from_weights(self, weights):
        # Weights have shape [channels, fourier_depth]
        ns = tf.constant(alternating_integers(weights.shape[1]))

        linspace = tf.cast(tf.linspace(- self.k_size / 2, self.k_size / 2, self.k_size), c64)
        filter_grid = 1.j * linspace[:, tf.newaxis] + linspace[:, tf.newaxis]

        radii = tf.abs(filter_grid)
        angles = tf.atan2(filter_grid.imag, filter_grid.real)




def alternating_integers(n):
    integers = [1]*n
    for i in range(n):
        integers[i] *= ((i + 1) // 2) if i % 2 else -((i + 1) // 2)
    return integers


if __name__ == '__main__':
    print(alternating_integers(8))
