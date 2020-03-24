import pytest
import tensorflow as tf
import tensorflow.keras as keras
from harmonic.convolution import Conv2DH


def test_harmonic_layer():
    # (batch_size, width, height, channels, streams)
    shape = ()
    test_input = tf.random.normal((5, 32, 32, 3, 1))
    model = keras.Sequential(layers=[
        Conv2DH(5, 1, 4),
        keras.layers.AveragePooling2D(),
        keras.layers.BatchNormalization(),
    ])
    model.compile(optimizer='adam', loss='mse')
    model.predict(test_input)

