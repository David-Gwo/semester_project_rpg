from tensorflow.python.keras.layers import Layer
from tensorflow.python.eager import context
from tensorflow.python.ops import gen_math_ops, math_ops, nn
from tensorflow.python.ops.array_ops import expand_dims, concat
from tensorflow.python.keras import activations, initializers, regularizers
from tensorflow.python.framework import dtypes, tensor_shape, ops
from tensorflow.python.keras import backend as K

from utils.algebra import exp_mapping, apply_state_diff, rotate_quat, rotate_vec

import numpy as np
import tensorflow as tf


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

        self.target_shape = target_shape
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.bias = None
        self.mask = None
        self.kernel = None

        self.flat_length = None

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())

        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` should be defined. Found `None`.')
        if len(input_shape) != 3:
            raise ValueError('The input shape should be a 3D tensor. Found %s' % (input_shape,))
        if input_shape[1:3] != self.target_shape:
            raise ValueError('The input and output shapes must coincide. Got {0} and {1}'.format(
                input_shape[1:3], self.target_shape))

        self.flat_length = np.prod(input_shape[1:3])
        output_channels = self.target_shape[-1]

        self.kernel = self.add_weight(
            'feature_kernel',
            shape=[self.flat_length, self.flat_length],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)

        recurrent_mask = np.zeros((input_shape[1], input_shape[1]))
        for i in range(input_shape[1]):
            for j in range(i, input_shape[1]):
                recurrent_mask[i, j] = 1
        recurrent_mask = np.tile(recurrent_mask[::-1, :], (output_channels, output_channels))

        self.mask = self.add_weight(
            shape=[self.flat_length, self.flat_length],
            initializer=lambda shape, dtype, partition_info=None: K.variable(recurrent_mask),
            dtype=self.dtype,
            trainable=False)

        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.flat_length, ],
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True)

        self.built = True

    # @tf.function
    def call(self, inputs, **kwargs):

        inputs = ops.convert_to_tensor(inputs)

        if not inputs.shape[0]:
            return K.expand_dims(inputs, axis=3)

        vectorized_input = K.reshape(inputs, (inputs.shape[0], self.flat_length))

        # Cast the inputs to self.dtype, which is the variable dtype. We do not
        # cast if `should_cast_variables` is True, as in that case the variable
        # will be automatically casted to inputs.dtype.
        if not self._mixed_precision_policy.should_cast_variables:
            vectorized_input = math_ops.cast(vectorized_input, self.dtype)

        outputs = gen_math_ops.mat_mul(vectorized_input, tf.math.multiply(self.kernel, self.mask))

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)  # pylint: disable=not-callable

        return K.reshape(outputs, (outputs.shape[0], self.target_shape[0], self.target_shape[1], 1))


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
        self.g_vec = np.expand_dims(np.array([0, 0, 9.81]), 0)

    def call(self, inputs, **kwargs):
        if not inputs[0].shape[0]:
            return inputs[0]

        state_in = inputs[0]
        pre_integration = inputs[1]
        total_dt = K.expand_dims(K.sum(K.squeeze(K.squeeze(inputs[2], axis=2), axis=2), axis=1), 1) / 1000

        rot_f = rotate_quat(state_in[:, 6:], pre_integration[:, 6:])
        vel_f = state_in[:, 3:6] + gen_math_ops.mat_mul(total_dt, self.g_vec) + \
            rotate_vec(pre_integration[:, 3:6], state_in[:, 6:])
        pos_f = state_in[:, :3] + math_ops.multiply(state_in[:, 3:6], total_dt) + \
            1/2 * gen_math_ops.mat_mul(total_dt ** 2, self.g_vec) + rotate_vec(pre_integration[:, :3], state_in[:, 6:])

        return concat([pos_f, vel_f, rot_f], axis=1)
