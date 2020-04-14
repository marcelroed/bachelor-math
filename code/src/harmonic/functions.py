import numpy as np
import tensorflow.keras as keras
import tensorflow as tf


def gaussian(z, m):
    """Return e^{-r^2}e^{i m phi}"""
    rs = np.abs(z)
    ths = np.arctan2(z.imag, z.real)

    return np.exp(-rs ** 2) * np.exp(1.j * m * ths)


class HNonLinearity(keras.layers.Layer):
    """
    Maps z to non_linearity(R + b) z/|z|
    """

    def __init__(self, non_linearity=tf.nn.relu, eps=1e-10):
        super(HNonLinearity, self).__init__()
        self.non_linearity = non_linearity
        self.eps = eps
        self.bias = None

    def build(self, input_shape):
        super(HNonLinearity, self).build(input_shape)
        batch_size, height, width, channels, orders, _ = input_shape
        bias_shape = [1, 1, 1, 1, orders, 2]
        self.bias = self.add_weight(name='fourier_weights',
                                    shape=bias_shape,
                                    initializer='zeros',
                                    trainable=True)

    def call(self, z, **kwargs):
        radii = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(z), axis=5, keepdims=True), self.eps))

        radii_b = radii + self.bias
        ratio = self.non_linearity(radii_b) / radii
        return ratio * z


class AvgPool2DH(keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.pooling = keras.layers.AvgPool2D(pool_size, strides, padding, data_format, **kwargs)

    # Doesn't need build because no parameters are needed

    def call(self, x, **kwargs):
        """
        Complex mean = mean(real) + i mean(imag)
        :param x:
        :param kwargs:
        :return:
        """
        batch_size, height, width, channels, orders, _ = x.shape
        re_input_shape = [-1, height, width, channels * orders * 2]
        x = tf.reshape(x, re_input_shape)
        x = self.pooling(x)
        print(f're_input_shape = {re_input_shape}')
        out_shape = self.pooling.compute_output_shape([None] + re_input_shape[1:])[:-1] + [channels, orders, 2]
        out_shape = [-1] + list(out_shape)[1:]
        print(f'out_shape = {out_shape}')
        return tf.reshape(x, out_shape)


class HBatchNormalization(keras.layers.Layer):
    def __init__(self, eps=1e-10, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        # The default axis is -1, meaning that each "row" in the final axis will be normalized over
        self.batch_normalization = keras.layers.BatchNormalization()

    def call(self, x, **kwargs):
        batch_size, height, width, channels, orders, _ = x.shape
        print(f'x.shape = {x.shape}')
        x = tf.reshape(tensor=x, shape=(-1, height, width, channels * orders * 2))

        radii = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x), axis=3, keepdims=True), self.eps))
        normalized = self.batch_normalization(radii)
        ratio = normalized / radii

        return tf.reshape(ratio * x, [-1, height, width, channels, orders, 2])


class HFlatten(keras.layers.Layer):
    def call(self, x, **kwargs):
        batch_size, height, width, channels, orders, _ = x.shape
        # TODO: Doesn't NEED to have a sqrt
        radii = tf.sqrt(tf.reduce_sum(tf.square(x), axis=5, keepdims=True))
        return tf.reshape(radii, [-1, height * width * channels * orders])


if __name__ == '__main__':
    # Average pooling
    x = tf.random.normal((5, 10, 10, 2, 2, 2))
    layer = AvgPool2DH()
    print(layer(x).shape)

    # Non-linearity with ReLU
    x = tf.random.normal((1, 4, 4, 1, 1, 2), stddev=0.25)
    layer = HNonLinearity()
    pre = tf.reshape(x, (4, 4, -1))
    non = tf.reshape(layer(x), (4, 4, -1))

    bn = HBatchNormalization()
    print(bn(x))
