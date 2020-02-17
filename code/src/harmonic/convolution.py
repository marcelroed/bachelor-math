import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from typing import Optional


class Conv2DH(keras.layers.Layer):
    def __init__(self, k_size=5, out_orders=(0, ), out_channels=1, **kwargs):
        super(Conv2DH, self).__init__(**kwargs)

        self.k_size = k_size
        self.m_orders = np.array(out_orders)
        self.out_channels = out_channels
        self.fourier_weights: Optional[tf.Variable] = None

    def build(self, input_shape):
        width, height, in_channels, in_orders = input_shape
        # Initialize weights
        weight_shape = (
            self.k_size, self.k_size, in_channels, self.out_channels, in_orders, self.out_orders
        )

        self.fourier_weights = self.add_weight(name='fourier_weights',
                                               shape=weight_shape,
                                               initializer='glorot_normal',
                                               trainable=True)
        super(Conv2DH, self).build(input_shape)

    def call(self, x, **kwargs):
        filters = self.generate_filters(self.fourier_weights)
        tf.nn.conv2d

    def compute_output_shape(self, input_shape):
        # Previous sizes
        width, height, in_channels, in_orders = input_shape

        return width, height, self.out_channels, len(self.out_orders)
