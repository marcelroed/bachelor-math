from typing import Optional, List, Dict
from itertools import product
from collections import defaultdict as dd

from utils import hot_shape as hs, alt_range, complex_conv2d, re, im

import tensorflow as tf
import tensorflow.keras as keras


class Conv2DH(keras.layers.Layer):
    def __init__(self, k_size: int = 5, out_orders: int = 1, out_channels: int = 1, fourier_depth: int = 5, **kwargs):
        super(Conv2DH, self).__init__(**kwargs)
        self.fourier_depth = fourier_depth
        self.out_channels = out_channels
        self.out_orders = out_orders
        self.k_size = k_size

        self.in_orders: Optional[int] = None
        self.in_channels: Optional[int] = None
        self.delta_ms: Optional[List] = None
        self.fourier_weights: Optional[tf.Variable] = None

    def build(self, input_shape):
        """Runs when constructing the parameters of the convolution layer"""

        # batch_size, ..., complex axis
        _, width, height, self.in_channels, self.in_orders, _ = input_shape

        self.delta_ms = sorted(set([out_stream - in_stream
                                    for in_stream, out_stream in
                                    product(range(self.in_orders), range(self.out_orders))]))
        print(self.delta_ms)

        # The shape of the fourier weight matrix
        weight_shape = (
            self.in_channels, self.out_channels, len(self.delta_ms), self.fourier_depth
        )

        self.fourier_weights = self.add_weight(name='fourier_weights',
                                               shape=weight_shape,
                                               initializer='glorot_normal',
                                               trainable=True)

        super(Conv2DH, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        Perform convolution
        :param x: input tensor of shape [batch_size, height, width, in_channels, in_streams, 2 (complex)]
        :param kwargs: Additional parameters
        :return: output tensor of shape [batch_size, height, width, out_channels, out_streams, 2]
        """

        filters = self._make_filters()

        x = 1.j * tf.cast(x[..., 1], tf.complex64) + tf.cast(x[..., 0], tf.complex64)

        stream_lists = dd(list)
        for in_stream, out_stream in product(range(self.in_orders), range(self.out_orders)):
            dm = out_stream - in_stream
            convolved = complex_conv2d(x[..., in_stream], filters[dm], strides=(1, 1), padding='SAME')
            stream_lists[out_stream].append(convolved)

        # Sum to combine elements in the same stream
        streams = {out: tf.reduce_sum(tf.stack(l, axis=4), axis=4) for out, l in stream_lists.items()}

        convolved = tf.stack([streams[k] for k in sorted(streams.keys())], axis=4)
        return tf.stack((re(convolved), im(convolved)), axis=-1)

    def _make_filters(self) -> Dict[int, tf.Tensor]:
        """
        Creates filters from the trained weights of the layer.
        :return: Dictionary from delta_m to complex Tensor of shape (k_size, k_size, in_channels, out_channels)
        """
        print(hs(4, 6))
        dms = tf.reshape(tf.constant(self.delta_ms, dtype=tf.float32), shape=hs(4, 6))

        weights = tf.reshape(self.fourier_weights, (1, 1, *self.fourier_weights.shape))

        ns = tf.reshape(tf.constant(alt_range(weights.shape[-1] - 1), dtype=tf.float32), shape=hs(5, 6))

        linspace = tf.cast(tf.linspace(- self.k_size / 2, self.k_size / 2, self.k_size), tf.complex64)
        filter_grid = 1.j * tf.reshape(linspace, hs(1, 6)) + tf.reshape(linspace, hs(0, 6))

        radii = tf.math.abs(filter_grid)
        angles = tf.math.angle(filter_grid)

        exponentials = tf.cast(weights[..., :-1], tf.complex64) * tf.exp(1.j * tf.cast(ns * radii, tf.complex64))
        filter_values = exponentials * tf.exp(1.j * tf.cast(dms * angles + weights[..., -1, tf.newaxis], tf.complex64))
        filter_tensor = tf.reduce_sum(filter_values, axis=-1)

        filter_dict = {delta_m: filter_tensor[..., i] for i, delta_m in enumerate(self.delta_ms)}

        return filter_dict


if __name__ == '__main__':
    layer = Conv2DH()

    batch_size, height, width, n_channels, n_streams = 10, 20, 20, 4, 3
    shape = (batch_size, height, width, n_channels, n_streams, 2)

    layer.build(input_shape=shape)
    inp = tf.random.normal(shape=shape)

    output = layer(inp)
    assert output.shape == (batch_size, height, width, 1, 1, 2), f'{output.shape} is not the right shape!'
