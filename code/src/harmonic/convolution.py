from itertools import product
from typing import Optional, List
from collections import defaultdict as dd

import tensorflow as tf
from tensorflow import newaxis
import tensorflow.keras as keras
from tensorflow import complex64 as c64, float32 as f32
from utils import re, im, alternating_integers, hot_shape as hs
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
            self.in_channels, self.out_channels, len(self.delta_ms), self.fourier_depth,
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
                complex_conv2d(x[..., in_stream], filters[delta_m], strides=(1, 1), padding='SAME'))

        streams = {out: tf.concat(l, axis=3) for out, l in stream_lists.items()}
        convolved = tf.concat([streams[k][tf.newaxis] for k in sorted(streams.keys())], axis=4)

        return convolved

    def compute_output_shape(self, input_shape):
        # Previous sizes
        height, width, in_channels, in_orders = input_shape

        return height, width, self.out_channels, self.out_orders

    def generate_filters(self):
        """
        Get a dictionary mapping d s.t. d[m_order] = Tensor[width, height, in_channels, out_channels]
        :return: Dictionary m_order -> Tensor[width, height, in_channels, out_channels]
        """
        delta_m_tensor = tf.constant(self.delta_ms, dtype=tf.float32)
        filter_tensor = self.filter_from_weights(self.fourier_weights, delta_m_tensor)

        # Dictionary for filters from input order to output order
        filter_dict = {delta_m: filter_tensor[..., i] for i, delta_m in enumerate(self.delta_ms)}

        return filter_dict

    def filter_from_weights(self, weights: tf.Tensor, ms: tf.Tensor):
        """Make filters from tensor of weights corresponding to rotation orders m. The final value in each weight row
        represents the rotational offset β.

        :param weights: tf.Tensor containing (in_channels, out_channels, m_order, fourier_depth) trainable weights
        :param ms: tf.Tensor containing (m_order, ) orders to use

        :return: tf.Tensor of type c64 and shape (k_size, k_size, in_channels, out_channels, m_order)
        """

        # All variables fit shape (k_size, k_size, in_channels, out_channels, m_order, fourier_depth - 1)
        # before the final dimension is summed over

        # Put ms in (1, 1, 1, 1, -1, 1)
        ms = tf.reshape(ms, hs(4, 6))
        weights = tf.reshape(weights, (1, 1, *weights.shape))
        print(f'{ms.shape=}, {weights.shape=}')
        # Weights have shape [in_channels, out_channels, fourier_depth]
        # ns for Fourier expansion
        ns = tf.reshape(tf.constant(alternating_integers(weights.shape[-1] - 1), dtype=tf.float32), hs(5, 6))

        # Make grid and get radii and angles for all points on grid
        linspace = tf.cast(tf.linspace(- self.k_size / 2, self.k_size / 2, self.k_size), c64)
        filter_grid = 1.j * tf.reshape(linspace, hs(1, 6)) + tf.reshape(linspace, hs(0, 6))

        radii = tf.abs(filter_grid)
        angles = tf.atan2(im(filter_grid), re(filter_grid))

        print(f'{radii.shape=}, {ns.shape=}')

        exponentials = tf.cast(weights[..., :-1], c64) * tf.exp(1.j * tf.cast(ns * radii, c64))
        print(f'{exponentials.shape=}, {tf.exp(1.j * tf.cast(ms * angles + weights[..., -1, tf.newaxis], c64)).shape=}')
        filter_values = exponentials * tf.exp(1.j * tf.cast(ms * angles + weights[..., -1, tf.newaxis], c64))
        print(f'{filter_values.shape=}')
        return tf.reduce_sum(filter_values, axis=-1)


def complex_conv2d(x, filters, **kwargs):
    print(f'{x.shape=}, {filters.shape=}')
    conv = lambda v, f: tf.nn.conv2d(v, f, **kwargs)
    return tf.cast(conv(re(x), re(filters)) - conv(im(x), im(filters)), c64) \
           + 1.j * tf.cast(conv(re(x), im(filters)) + conv(im(x), re(filters)), c64)


if __name__ == '__main__':
    conv = Conv2DH(k_size=6, out_orders=7)
    test_input = tf.random.normal((5, 32, 32, 3, 1))
    print(conv(test_input))
