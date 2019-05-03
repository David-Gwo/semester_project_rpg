from tensorflow.python.keras.layers import Layer
from utils.algebra import exp_mapping, apply_state_diff
from tensorflow.python.ops.array_ops import expand_dims, concat


class ForkLayerIMUdt(Layer):
    def __init__(self, window_len, name=None):
        super(ForkLayerIMUdt, self).__init__(name=name)
        self.imu_window_len = window_len

    def call(self, inputs, **kwargs):
        return inputs[:, :self.imu_window_len, :6, :], \
               inputs[:, :self.imu_window_len, 6:, :]


class ExponentialRemappingLayer(Layer):
    def __init__(self, name=None):
        super(ExponentialRemappingLayer, self).__init__(name=name, trainable=False)

    def call(self, inputs, **kwargs):
        if not inputs.shape[0]:
            return concat([inputs, expand_dims(inputs[:, 0], axis=1)], axis=1)

        q = exp_mapping(inputs[:, 6:9])
        return concat([inputs[:, :6], q], axis=1)


class DiffConcatenationLayer(Layer):
    def __init__(self, name=None):
        super(DiffConcatenationLayer, self).__init__(name=name, trainable=False)

    def call(self, inputs, **kwargs):
        if not inputs[0].shape[0]:
            return inputs[0]
        return apply_state_diff(inputs[0], inputs[1])
