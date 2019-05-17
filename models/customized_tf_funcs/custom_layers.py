from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops import gen_math_ops, math_ops, nn
from tensorflow.python.ops.array_ops import expand_dims, concat, ops
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.keras import backend as K

from utils.algebra import exp_mapping, apply_state_diff, rotate_quat, rotate_vec

import numpy as np
import tensorflow as tf


class ForkLayerIMUdt(Layer):
    def __init__(self, name=None):
        super(ForkLayerIMUdt, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        return inputs[:, :, :6, :], inputs[:, :, 6:, :]


class PreProcessIMU(Layer):
    def __init__(self, name=None):
        super(PreProcessIMU, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        return concat([inputs[:, :, :3, :], inputs[:, :, 6:, :]], axis=2), \
               concat([inputs[:, :, 3:6, :], inputs[:, :, 6:, :]], axis=2), \
               inputs[:, :, 6:, :]


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
        self.recurrent_mask = None
        self.recurrent_kernel = None

        self.total_recurrent_units = None

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())

        # Input sanity checks
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point dtype %s' % (dtype,))

        if isinstance(input_shape, tf.TensorShape):
            sample_input_shape = input_shape
            if len(input_shape) != 3:
                raise ValueError('The first input shape should be a 3D tensor. Found %s' % (input_shape, ))
            if input_shape[1:] != self.target_shape:
                raise ValueError('The input and output shapes must coincide. Got {0} instead of {1}'.format(
                    input_shape[1:], self.target_shape))
        else:
            raise TypeError("The input should be a single tensor or a list of tensors")

        self.total_recurrent_units = np.prod(sample_input_shape[1:])

        output_channels = self.target_shape[-1]

        self.recurrent_kernel = self.add_weight(
            'feature_kernel',
            shape=[self.total_recurrent_units, self.total_recurrent_units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)

        recurrent_mask = np.zeros((sample_input_shape[1], sample_input_shape[1]))
        for i in range(sample_input_shape[1]):
            for j in range(i, sample_input_shape[1]):
                recurrent_mask[i, j] = 1
        recurrent_mask = np.tile(recurrent_mask, (output_channels, output_channels))

        self.recurrent_mask = self.add_weight(
            shape=[self.total_recurrent_units, self.total_recurrent_units],
            initializer=lambda shape, dtype, partition_info=None: K.variable(recurrent_mask),
            dtype=self.dtype,
            trainable=False)

        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.total_recurrent_units, ],
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):

        if not inputs.shape[0]:
            return inputs

        recurrent_input = ops.convert_to_tensor(inputs)

        if not self._mixed_precision_policy.should_cast_variables:
            recurrent_input = math_ops.cast(recurrent_input, self.dtype)

        batch_size = recurrent_input.shape[0]

        # Flatten last two dimensions, but along dimension [2]
        flat_recurrent = K.reshape(K.permute_dimensions(recurrent_input, (0, 2, 1)), (batch_size, -1))
        outputs = gen_math_ops.mat_mul(flat_recurrent, tf.math.multiply(self.recurrent_kernel, self.recurrent_mask))

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)

        # Transform back outputs to original shape
        outputs = K.reshape(K.transpose(outputs), (self.target_shape[0], self.target_shape[1], batch_size))
        outputs = K.reshape(outputs, (self.target_shape[1], self.target_shape[0], batch_size))
        outputs = K.permute_dimensions(outputs, (2, 1, 0))

        return outputs


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


class IntegratingLayer(Layer):
    def __init__(self, name=None):
        super(IntegratingLayer, self).__init__(name=name, trainable=False)
        # TODO: pass sign as argument
        self.g_vec = np.expand_dims(np.array([0, 0, 9.81]), 0)

    def call(self, inputs, **kwargs):
        if not inputs[0].shape[0]:
            return inputs[0]

        state_in = inputs[0]
        pre_int_vecs = inputs[1]
        pre_int_rot = exp_mapping(pre_int_vecs[0][:, -1, :])
        pre_int_vel = pre_int_vecs[1][:, -1, :]
        pre_int_pos = pre_int_vecs[2][:, -1, :]

        total_dt = K.expand_dims(K.sum(K.squeeze(K.squeeze(inputs[2], axis=2), axis=2), axis=1), 1) / 1000

        rot_f = rotate_quat(state_in[:, 6:], pre_int_rot)
        vel_f = state_in[:, 3:6] + gen_math_ops.mat_mul(total_dt, self.g_vec) + \
            rotate_vec(pre_int_vel, state_in[:, 6:])
        pos_f = state_in[:, :3] + math_ops.multiply(state_in[:, 3:6], total_dt) + \
            1/2 * gen_math_ops.mat_mul(total_dt ** 2, self.g_vec) + rotate_vec(pre_int_pos, state_in[:, 6:])

        return concat([pos_f, vel_f, rot_f], axis=1)
