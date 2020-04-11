import tensorflow as tf
from tensorflow import complex64 as c64

re = tf.math.real
im = tf.math.imag


def hot_shape(hot, length):
    return tuple(-1 if i == hot else 1 for i in range(length))


def alt_range(n):
    """
    Get a sequence of integers increasing in absolute value
    :param n: How many integers to get
    :return: List of alternating numbers increasing in absolute value
    """
    integers = [1] * n
    for i in range(n):
        integers[i] *= ((i + 1) // 2) if i % 2 else -((i + 1) // 2)
    return integers


def complex_conv2d(x, filters, **kwargs):
    # print(f'{x.shape=}, {filters.shape=}')
    conv = lambda v, f: tf.nn.conv2d(v, f, **kwargs)
    return tf.cast(conv(re(x), re(filters)) - conv(im(x), im(filters)), c64) \
           + 1.j * tf.cast(conv(re(x), im(filters)) + conv(im(x), re(filters)), c64)

