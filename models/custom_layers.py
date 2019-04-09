from tensorflow.python.keras.layers import Layer
import tensorflow as tf


class ForkLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(ForkLayer, self).__init__(args, kwargs)

    def call(self, inputs, **kwargs):
        return inputs[:, :, 0:6, :] * 0, inputs[:, :, 6, :] * 1
