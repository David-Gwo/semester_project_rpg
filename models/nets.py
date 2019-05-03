import tensorflow as tf
from tensorflow.python.keras import regularizers, Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D, Conv1D, MaxPooling2D, \
    BatchNormalization, Concatenate, Reshape
from tensorflow.python.keras.layers.merge import add

from models.customized_tf_funcs.custom_layers import ExponentialRemappingLayer, ForkLayerIMUdt, DiffConcatenationLayer

##########################################
# ADD HERE YOUR NETWORK                  #
# BUILD IT WITH PURE TENSORFLOW OR KERAS #
##########################################

# NOTE: IF USING KERAS, YOU MIGHT HAVE PROBLEMS WITH BATCH-NORMALIZATION


def vel_cnn():
    model = Sequential()

    model.add(Conv2D(filters=60, kernel_size=(3, 6), padding='same', activation='relu', input_shape=(200, 6, 1),
                     name="convolution_layer_1"))
    model.add(Conv2D(filters=120, kernel_size=(3, 6), padding='same', activation='relu', name="convolution_layer_2"))
    model.add(Conv2D(filters=240, kernel_size=(3, 1), padding='valid', activation='relu', name="convolution_layer_3"))
    model.add(MaxPooling2D(pool_size=(10, 1), strides=(6, 1), name="pooling_layer"))
    model.add(Flatten(name="flattening_layer"))
    model.add(Dense(400, activation='relu', name="dense_layer_1"))
    model.add(Dense(100, activation='relu', name="dense_layer_2"))
    model.add(Dense(1, name="output_layer"))

    return model


def imu_so3_integration_net(window_len, ):

    imu_final_channels = 5
    input_state_len = 10
    output_state_len = 9

    conv_kernel_width = min([window_len, 2])

    # Input layers
    net_in = Input((window_len + input_state_len, 7, 1), name="imu_input")
    state_0 = Input((input_state_len, 1), name="state_input")

    imu_stack, time_diff_imu = ForkLayerIMUdt(window_len, name="forking_layer")(net_in)

    imu_conv_1 = Conv2D(filters=15, kernel_size=(conv_kernel_width, 3), strides=(1, 3), padding='same',
                        activation='relu', name='imu_conv_layer_1')(imu_stack)
    imu_conv_2 = Conv2D(filters=30, kernel_size=(conv_kernel_width, 1), padding='same',
                        activation='relu', name='imu_conv_layer_2')(imu_conv_1)
    imu_conv_3 = Conv2D(filters=60, kernel_size=(conv_kernel_width, 1), padding='same',
                        activation='relu', name='imu_conv_layer_3')(imu_conv_2)

    imu_conv_reduced = Conv2D(filters=imu_final_channels, kernel_size=(conv_kernel_width, 1), padding='same',
                              activation='tanh', name='imu_1x_conv_layer')(imu_conv_3)
    imu_conv_flat = Reshape((window_len, imu_final_channels * 2, 1), name='reshape_layer')(imu_conv_reduced)
    re_stacked_imu = Concatenate(name='imu_concatenating_layer', axis=2)([imu_conv_flat, time_diff_imu])

    imu_ts_conv = Conv2D(filters=1+2*imu_final_channels, kernel_size=(conv_kernel_width, 1+2*imu_final_channels),
                         padding='valid', activation='relu', name='imu_final_conv_layer')(re_stacked_imu)

    flatten_conv_imu = Flatten(name='imu_flattening_layer')(imu_ts_conv)
    flatten_state_0 = Flatten(name='state_input_flattening_layer')(state_0)

    stacked = Concatenate(name='data_concatenating_layer')([flatten_conv_imu, flatten_state_0])

    dense_1 = Dense(400, name='dense_layer_1')(stacked)
    activation_1 = Activation('relu', name='activation_1')(dense_1)

    dense_2 = Dense(200, name='dense_layer_2')(activation_1)
    activation_2 = Activation('relu', name='activation_2')(dense_2)

    dense_3 = Dense(100, name='dense_layer_3')(activation_2)
    activation_3 = Activation('relu', name='activation_3')(dense_3)

    net_out = Dense(output_state_len, name='state_output')(activation_3)

    state_out = ExponentialRemappingLayer(name='remapped_state_output')(net_out)
    return Model(inputs=net_in, outputs=net_out), Model(inputs=net_in, outputs=state_out)


def split_imu_integration_net(window_len, input_state_len, diff_state_len, output_state_len):

    imu_final_channels = 5

    conv_kernel_width = min([window_len, 2])
    imu_in = Input((window_len, 7, 1), name="imu_input")
    imu_stack, time_diff_imu = ForkLayerIMUdt(window_len, name="IMU_dt_fork_layer")(imu_in)

    imu_conv_1 = Conv2D(filters=15, kernel_size=(conv_kernel_width, 3), strides=(1, 3), padding='same',
                        activation='relu', name='imu_conv_layer_1')(imu_stack)
    imu_conv_2 = Conv2D(filters=30, kernel_size=(conv_kernel_width, 1), padding='same',
                        activation='relu', name='imu_conv_layer_2')(imu_conv_1)
    imu_conv_3 = Conv2D(filters=60, kernel_size=(conv_kernel_width, 1), padding='same',
                        activation='relu', name='imu_conv_layer_3')(imu_conv_2)

    imu_conv_reduced = Conv2D(filters=imu_final_channels, kernel_size=(conv_kernel_width, 1), padding='same',
                              activation='tanh', name='imu_1x_conv_layer')(imu_conv_3)

    imu_conv_flat = Reshape((window_len, imu_final_channels * 2, 1), name='reshape_layer')(imu_conv_reduced)
    re_stacked_imu = Concatenate(name='imu_concatenating_layer', axis=2)([imu_conv_flat, time_diff_imu])

    imu_ts_conv = Conv2D(filters=1 + 2 * imu_final_channels,
                         kernel_size=(conv_kernel_width, 1 + 2 * imu_final_channels),
                         padding='valid', activation='relu', name='imu_final_conv_layer')(re_stacked_imu)

    flatten_conv_imu = Flatten(name='imu_flattening_layer')(imu_ts_conv)

    dense_1 = Dense(200, name='dense_layer_1', activation='relu')(flatten_conv_imu)
    diff_output = Dense(diff_state_len, name='diff_output', activation='relu')(dense_1)

    diff_su2 = ExponentialRemappingLayer(name='exponential_mapping_layer')(diff_output)

    state_in = Input((input_state_len, 1), name="state_input")

    state_out = DiffConcatenationLayer(name="state_output")((state_in, diff_su2))

    return Model(inputs=(imu_in, state_in), outputs=(state_out, diff_output))
