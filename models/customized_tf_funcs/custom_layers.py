from tensorflow.python.keras.layers import Layer
from tensorflow.python.eager import context
from tensorflow.python.ops import gen_math_ops, math_ops, nn
from tensorflow.python.ops.array_ops import expand_dims, concat
from tensorflow.python.keras import activations, initializers, regularizers
from tensorflow.python.framework import dtypes, tensor_shape, ops
from tensorflow.python.keras import backend as K

from utils.algebra import exp_mapping, apply_state_diff, rotate_quat, rotate_vec

import numpy as np


class ForkLayerIMUdt(Layer):
    def __init__(self, name=None):
        super(ForkLayerIMUdt, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        return inputs[:, :, :6, :], inputs[:, :, 6:, :]


class ReshapeIMU(Layer):
    def __init__(self, name=None):
        super(ReshapeIMU, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        return concat([inputs[:, :, :3, :], inputs[:, :, 6:, :], inputs[:, :, 3:6, :], inputs[:, :, 6:, :]], axis=2)


class PreIntegrationForwardDense(Layer):
    def __init__(self,
                 target_shape,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 name=None):
        super(PreIntegrationForwardDense, self).__init__(name=name)

        if len(target_shape) != 2:
            raise ValueError("The target shape should be 3D")

        self.units = int(target_shape[0])
        self.channels = int(target_shape[1])
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.feature_kernel = None
        self.bias = None
        self.recurrent_mask = None
        self.recurrent_kernel = None

        self.feature_units = None
        self.recurrent_units = None

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())

        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` should be defined. Found `None`.')
        if len(input_shape) != 3:
            raise ValueError('The input shape should be a 3D tensor. Found %s' % (input_shape,))
        if input_shape[1] != self.units:
            raise ValueError('The first dimension of input and output shape must coincide')
        if input_shape[2] < self.channels:
            raise ValueError('The number of channels in the input is smaller than in the output. It should be at least'
                             'the same size')

        self.feature_units = input_shape[1] * (input_shape[2] - self.channels)
        self.recurrent_units = input_shape[1] * self.channels

        self.feature_kernel = self.add_weight(
            'feature_kernel',
            shape=[self.feature_units, self.recurrent_units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)

        self.recurrent_kernel = self.add_weight(
            'recurrent_kernel',
            shape=[self.recurrent_units, self.recurrent_units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)

        recurrent_mask = np.zeros((self.units, self.units))
        for i in range(self.units):
            for j in range(i, self.units):
                recurrent_mask[i, j] = 1
        recurrent_mask = np.tile(recurrent_mask, (self.channels, self.channels))

        self.recurrent_mask = self.add_weight(
            shape=[self.recurrent_units, self.recurrent_units],
            initializer=lambda shape, dtype, partition_info=None: K.variable(recurrent_mask),
            dtype=self.dtype,
            trainable=False)

        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.recurrent_units, ],
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):

        inputs = ops.convert_to_tensor(inputs)

        feature_vec = inputs[:, :, :-self.channels]
        recurrent_vec = inputs[:, :, -self.channels:]

        if not context.executing_eagerly():
            return recurrent_vec

        feature_vec = K.reshape(feature_vec, (feature_vec.shape[0], self.units*feature_vec.shape[2]))
        recurrent_vec = K.reshape(recurrent_vec, (recurrent_vec.shape[0], self.units * self.channels))

        # Cast the inputs to self.dtype, which is the variable dtype. We do not
        # cast if `should_cast_variables` is True, as in that case the variable
        # will be automatically casted to inputs.dtype.
        if not self._mixed_precision_policy.should_cast_variables:
            feature_vec = math_ops.cast(feature_vec, self.dtype)
            recurrent_vec = math_ops.cast(recurrent_vec, self.dtype)

        outputs = gen_math_ops.mat_mul(feature_vec, self.feature_kernel)
        outputs += gen_math_ops.mat_mul(gen_math_ops.mat_mul(recurrent_vec, self.recurrent_kernel), self.recurrent_mask)

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)  # pylint: disable=not-callable

        return K.reshape(outputs, (outputs.shape[0], self.units, self.channels))


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


class FinalPreIntegration(Layer):
    def __init__(self, name=None):
        super(FinalPreIntegration, self).__init__(name=name, trainable=False)

    def call(self, inputs, **kwargs):
        if not inputs[0].shape[0]:
            return K.concatenate([inputs[2][:, 0, :], inputs[1][:, 0, :], K.placeholder([None, 4])])

        return K.concatenate([inputs[2][:, 0, :], inputs[1][:, 0, :], exp_mapping(inputs[0][:, 0, :])])


class IntegratingLayer(Layer):
    def __init__(self, name=None):
        super(IntegratingLayer, self).__init__(name=name, trainable=False)
        # TODO: pass sign as argument
        self.g_vec = np.expand_dims(np.array([0, 0, -9.81]), 0)

    def call(self, inputs, **kwargs):
        if not inputs[0].shape[0]:
            return inputs[0]

        state_in = inputs[0]
        pre_integration = inputs[1]
        total_dt = K.expand_dims(K.sum(K.squeeze(K.squeeze(inputs[2], axis=2), axis=2), axis=1), 1)

        rot_f = rotate_quat(state_in[:, 6:], pre_integration[:, 6:])
        vel_f = state_in[:, 3:6] + gen_math_ops.mat_mul(total_dt, self.g_vec) + \
            rotate_vec(pre_integration[:, 3:6], state_in[:, 6:])
        pos_f = state_in[:, :3] + math_ops.multiply(state_in[:, 3:6], total_dt) + \
            1/2 * gen_math_ops.mat_mul(total_dt ** 2, self.g_vec) + rotate_vec(pre_integration[:, :3], state_in[:, 6:])

        return concat([pos_f, vel_f, rot_f], axis=1)
