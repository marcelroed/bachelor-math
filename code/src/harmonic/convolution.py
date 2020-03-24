from itertools import product
from typing import Optional, List
from collections import defaultdict as dd

import tensorflow as tf
from tensorflow import newaxis
import tensorflow.keras as keras
from tensorflow import complex64 as c64, float32 as f32
import numpy as np


class Conv2DH(keras.layers.Layer):
    def __init__(self, k_size=5, out_orders=1, out_channels=1, fourier_depth=5, **kwargs):
        super(Conv2DH, self).__init__(**kwargs)

        self.k_size = k_size
        self.out_orders = out_orders
        self.out_channels = out_channels
        self.fourier_depth = fourier_depth
        self.fourier_weights: Optional[tf.Variable] = None
        self.in_orders, self.in_channels = None, None
        self.delta_ms: Optional[List] = None

    def build(self, input_shape):
        batch_size, width, height, self.in_channels, self.in_orders = input_shape

        self.delta_ms = list(set([out_stream - in_stream
                             for in_stream, out_stream in product(range(self.in_orders), range(self.out_orders))]))

        # Initialize weights
        weight_shape = (
            self.in_channels, self.out_channels, self.fourier_depth,
        )

        self.fourier_weights = self.add_weight(name='fourier_weights',
                                               shape=weight_shape,
                                               initializer='glorot_normal',
                                               trainable=True)
        super(Conv2DH, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        Perform the
        :param x: input tensor of shape [batch_size, height, width, n_channels, n_streams]
        :param kwargs: Additional parameters (not used)
        :return:
        """


        print(f'{self.delta_ms=}')

        filters = self.generate_filters()

        stream_lists = dd(list)
        for in_stream, out_stream in product(range(self.in_orders), range(self.out_orders)):
            delta_m = out_stream - in_stream
            stream_lists[out_stream].append(
                tf.nn.conv2d(x[..., in_stream, newaxis], filters[delta_m], strides=(1, 1), padding='same'))

        streams = {out: tf.concat(l, axis=3) for out, l in stream_lists.items()}
        convolved = tf.concat([streams[k] for k in sorted(streams.keys())], axis=4)

        return convolved

    def compute_output_shape(self, input_shape):
        # Previous sizes
        height, width, in_channels, in_orders = input_shape

        return height, width, self.out_channels, self.out_orders

    def generate_filters(self):
        delta_m_tensor = tf.constant(self.delta_ms, dtype=tf.float32)
        filter_tensor = self.filter_from_weights(self.fourier_weights, delta_m_tensor)

        # Dictionary for filters from input order to output order
        filter_dict = {delta_m: filter_tensor[..., i] for i, delta_m in enumerate(self.delta_ms)}

        return filter_dict

    def filter_from_weights(self, weights: tf.Tensor, ms: tf.Tensor):
        """Make filters from tensor of weights corresponding to rotation orders m. The final value in each weight row
        represents the rotational offset β.

        :param weights: tf.Tensor containing [m_order, in_channels, out_channels, N_weights + 1] trainable weights
        :param ms: tf.Tensor containing [m_order] orders to use

        :return: tf.Tensor of type c64 and shape (width, height)
        """
        print(f'{weights.shape=}, {ms.shape=}')
        # Weights have shape [channels, fourier_depth]
        # ns for Fourier expansion
        ns = tf.constant(alternating_integers(weights.shape[-1] - 1), dtype=tf.float32)[newaxis, newaxis, :]

        print(f'{ns.shape=}')

        linspace = tf.cast(tf.linspace(- self.k_size / 2, self.k_size / 2, self.k_size), c64)
        filter_grid = 1.j * tf.reshape(linspace, (1, -1)) + tf.reshape(linspace, (-1, 1))
        filter_grid = tf.reshape(filter_grid, (self.k_size, self.k_size, ))

        radii = tf.abs(filter_grid.resize)
        angles = tf.atan2(tf.math.imag(filter_grid), tf.math.real(filter_grid))

        exponentials = tf.cast(weights[..., :-1], c64) * tf.exp(1.j * tf.cast(ns * radii, c64))
        filter_values = tf.reduce_sum(exponentials, axis=2) * tf.exp(1.j * tf.cast(ms * angles + weights[:, -1], c64))
        return filter_values


def alternating_integers(n):
    integers = [1] * n
    for i in range(n):
        integers[i] *= ((i + 1) // 2) if i % 2 else -((i + 1) // 2)
    return integers


if __name__ == '__main__':
    conv = Conv2DH()
    conv.filter_from_weights()
