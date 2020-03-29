import tensorflow as tf

re = tf.math.real
im = tf.math.imag


def hot_shape(hot, length):
    return tuple(-1 if i == hot else 1 for i in range(length))


def alternating_integers(n):
    """
    Get a sequence of integers increasing in absolute value
    :param n: How many integers to get
    :return: List of alternating numbers increasing in absolute value
    """
    integers = [1] * n
    for i in range(n):
        integers[i] *= ((i + 1) // 2) if i % 2 else -((i + 1) // 2)
    return integers

