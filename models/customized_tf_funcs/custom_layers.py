from tensorflow.python.keras.layers import Layer


class ForkLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(ForkLayer, self).__init__(args, kwargs)

    def call(self, inputs, **kwargs):
        return inputs[:, :, 0:6, :] * 1, inputs[:, :, 6, :] * 1


class ForkLayerIMUInt(Layer):
    def __init__(self, window_len, state_len, name=None):
        super(ForkLayerIMUInt, self).__init__(name=name)
        self.imu_window_len = window_len
        self.state_len = state_len

    def call(self, inputs, **kwargs):
        return inputs[:, :self.imu_window_len, :6, :], inputs[:, :self.imu_window_len, 6:, :], \
               inputs[:, self.imu_window_len:, 0, :]

