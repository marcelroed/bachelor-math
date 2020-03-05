import tensorflow.keras as keras
import tensorflow as tf

from kerastuner import HyperModel, HyperParameters

from harmonic.convolution import Conv2DH

class HCNN(HyperModel):
    def __init__(self):
        super(HCNN, self).__init__()

    def build(self, hp: HyperParameters):
        model = keras.Sequential()
        model.add(Conv2DH(
            k_size=5,
            out_orders=(0, 1),
            out_channels=8
        ))
        model.add(Conv2DH(
            k_size=5,
            out_orders=(0, 1),
            out_channels=8
        ))
        model.add(keras.layers.AveragePooling2D())
        model.add(Conv2DH(
            k_size=5,
            out_orders=(0, 1),
            out_channels=8
        ))
        model.add(Conv2DH(
            k_size=5,
            out_orders=(0, 1),
            out_channels=8
        ))
        model.add(keras.layers.AveragePooling2D())
        model.add(Conv2DH(
            k_size=5,
            out_orders=(0, 1),
            out_channels=8
        ))
        model.add(Conv2DH(
            k_size=5,
            out_orders=(0, ),
            out_channels=10
        ))

        # Reduce to one stream
        model.add()
