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
        self.kernel = None
        self.recurrent_mask = None
        self.recurrent_kernel = None

        self.total_recurrent_units = None
        self.total_kernel_units = None

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())

        # Input sanity checks
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point dtype %s' % (dtype,))
        if len(input_shape) != 2:
            raise ValueError('There should only be two inputs to this layer. Got %s' % (len(input_shape)))

        input_shape_1 = None
        input_shape_2 = tensor_shape.TensorShape(input_shape[1])
        n_recurrencies = 1

        if isinstance(input_shape[0], tuple):
            n_recurrencies = len(input_shape[0])
            for input_shape_1 in input_shape[0]:
                if tensor_shape.dimension_value(input_shape[0][-1]) is None:
                    raise ValueError('The last dimension of the inputs to `Dense` should be defined. Found `None`.')
                if len(input_shape_1) != 3:
                    raise ValueError('The first input shape should be a 3D tensor. Found %s' % (input_shape_1,))
                if input_shape_1[1:3] != self.target_shape:
                    raise ValueError('The input and output shapes must coincide. Got {0} and {1}'.format(
                        input_shape_1[1:3], self.target_shape))
        elif isinstance(input_shape[0], tf.TensorShape):
            input_shape_1 = input_shape[0]
            if tensor_shape.dimension_value(input_shape[0][-1]) is None:
                raise ValueError('The last dimension of the inputs to `Dense` should be defined. Found `None`.')
            if len(input_shape_1) != 3:
                raise ValueError('The first input shape should be a 3D tensor. Found %s' % (input_shape_1,))
            if input_shape_1[1:3] != self.target_shape:
                raise ValueError('The input and output shapes must coincide. Got {0} and {1}'.format(
                    input_shape_1[1:3], self.target_shape))
        else:
            raise TypeError("The first input should be a tensor or a tuple of tensors")

        self.total_recurrent_units = np.prod(input_shape_1[1:3])
        self.total_kernel_units = np.prod(input_shape_2[1:])

        output_channels = self.target_shape[-1]

        self.recurrent_kernel = self.add_weight(
            'feature_kernel',
            shape=[self.total_recurrent_units, self.total_recurrent_units, n_recurrencies],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)

        self.kernel = self.add_weight(
            'kernel',
            shape=[self.total_kernel_units, self.total_recurrent_units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)

        recurrent_mask = np.zeros((input_shape_1[1], input_shape_1[1]))
        for i in range(input_shape_1[1]):
            for j in range(i, input_shape_1[1]):
                recurrent_mask[i, j] = 1
        recurrent_mask = np.tile(recurrent_mask[::-1, :], (output_channels, output_channels))

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

        recurrent_input, feature_input = inputs

        if isinstance(recurrent_input, tuple):
            if not recurrent_input[0].shape[0]:
                return K.expand_dims(recurrent_input[0], axis=3)

            recurrent_input = K.concatenate([K.expand_dims(rec_in, 3) for rec_in in recurrent_input], axis=3)

        else:
            if not recurrent_input.shape[0]:
                return K.expand_dims(recurrent_input, axis=3)

            recurrent_input = K.expand_dims(recurrent_input, axis=3)

        recurrent_input = ops.convert_to_tensor(recurrent_input)
        feature_input = ops.convert_to_tensor(feature_input)

        if not self._mixed_precision_policy.should_cast_variables:
            recurrent_input = math_ops.cast(recurrent_input, self.dtype)
            feature_input = math_ops.cast(feature_input, self.dtype)

        vectorized_features = K.reshape(feature_input, (feature_input.shape[0], self.total_kernel_units))
        outputs = gen_math_ops.mat_mul(vectorized_features, self.kernel)

        for i in range(self.recurrent_kernel.shape[-1]):
            vectorized_recurrent = K.reshape(recurrent_input[:, :, :, i], (recurrent_input.shape[0], self.total_recurrent_units))
            outputs += gen_math_ops.mat_mul(vectorized_recurrent, tf.math.multiply(self.recurrent_kernel[:, :, i], self.recurrent_mask))

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)

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
