import tensorflow.keras as keras
import tensorflow as tf
from data.mnist import get_mnist

from kerastuner import HyperModel, HyperParameters, RandomSearch

from harmonic.convolution import Conv2DH


class HCNN(HyperModel):
    def __init__(self):
        super(HCNN, self).__init__()

    def build(self, hp: HyperParameters):
        model = keras.models.Sequential([
            keras.layers.Reshape((28, 28, 1, 1)),  # Introduce streams
            keras.layers.Lambda(lambda v: tf.stack((v, tf.zeros_like(v)), axis=-1)),
            keras.layers.Lambda(print_return),
            Conv2DH(out_orders=2, out_channels=8),
            keras.layers.Lambda(print_return),
            keras.layers.AvgPool2D(),

            Conv2DH(out_orders=1, out_channels=10),
            keras.layers.Lambda(lambda v: tf.math.reduce_euclidean_norm(v, axis=-1)),
            keras.layers.Lambda(print_return),
            keras.layers.Softmax(),
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


def print_return(v):
    print(v.shape)
    return v


if __name__ == '__main__':
    hypermodel = HCNN()
    train, test = get_mnist()
    tuner = RandomSearch(
        hypermodel,
        objective='accuracy',
        max_trials=10,
        directory='models',
        project_name='H_mnist'
    )

    print('Created tuner')
    tuner.search(*train, epochs=5, validation_data=test)

"""
        model = keras.Sequential([
            layer for group in [
                [
                    Conv2DH(5, 3),

                ] for i in range(4)
            ] for layer in group
        ] + [

        ])
"""
