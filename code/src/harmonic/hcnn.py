import datetime

import tensorflow.keras as keras
import tensorflow as tf
from data.mnist import get_mnist

from kerastuner import HyperModel, HyperParameters, RandomSearch

from harmonic.convolution import Conv2DH
from harmonic.functions import AvgPool2DH, HNonLinearity, HBatchNormalization, HFlatten


class HCNN(HyperModel):
    def __init__(self):
        super(HCNN, self).__init__()

    def build(self, hp: HyperParameters):
        model = keras.models.Sequential([
            keras.layers.Reshape((28, 28, 1, 1)),  # Introduce streams
            keras.layers.Lambda(lambda v: tf.stack((v, tf.zeros_like(v)), axis=-1)),  # Imaginary part initialized to 0
            keras.layers.Lambda(print_return),

            # Block 1: Shape [batch, 28, 28, channels=8, streams=2, 2]
            Conv2DH(out_orders=2, out_channels=8),
            HNonLinearity(),  # Defaults to ReLU
            Conv2DH(out_orders=2, out_channels=8),
            HBatchNormalization(),

            # Block 2: Shape [batch, 14, 14, channels=16, streams=2, 2]
            AvgPool2DH(strides=(2, 2)),
            Conv2DH(out_orders=2, out_channels=16),
            HNonLinearity(),
            Conv2DH(out_orders=2, out_channels=16),
            HBatchNormalization(),

            # Block 3: Shape [batch, 7, 7, channels=35, streams=2, 2]
            AvgPool2DH(),
            Conv2DH(out_orders=2, out_channels=35),
            HNonLinearity(),
            Conv2DH(out_orders=2, out_channels=35),

            # Block 4: Reduce to magnitudes and apply final activation
            HFlatten(),
            keras.layers.Lambda(print_return),
            keras.layers.Dense(10),
            keras.layers.Softmax(),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=10 ** hp.Float('log_learning_rate', -6, -1, step=0.5, default=-3)),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


def print_return(v):
    print(v.shape)
    return v


if __name__ == '__main__':
    # tf.keras.backend.set_floatx('float16')

    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    hypermodel = HCNN()
    train, test = get_mnist()
    tuner = RandomSearch(
        hypermodel,
        objective='accuracy',
        max_trials=40,
        directory='models',
        project_name='H-MNIST-' + dt
    )

    print('Created tuner')
    log_dir = "logs/fit/" + dt
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    tuner.search(*train, epochs=30, validation_data=test, batch_size=32,
                 callbacks=[tensorboard_callback])
