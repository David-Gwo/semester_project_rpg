from tensorflow.python.keras import regularizers, Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers

from models.customized_tf_funcs import custom_layers
import tensorflow as tf


def vel_cnn():
    model = Sequential()

    model.add(layers.Conv2D(filters=60, kernel_size=(3, 6), padding='same', activation='relu', input_shape=(200, 6, 1)))
    model.add(layers.Conv2D(filters=120, kernel_size=(3, 6), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=240, kernel_size=(3, 1), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(10, 1), strides=(6, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(400, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(1))

    return model


def imu_so3_integration_net(window_len):

    imu_final_channels = 5
    input_state_len = 10
    output_state_len = 9

    conv_kernel_width = min([window_len, 2])

    # Input layers
    net_in = layers.Input((window_len + input_state_len, 7, 1), name="imu_input")
    state_0 = layers.Input((input_state_len, 1), name="state_input")

    imu_stack, time_diff_imu = custom_layers.ForkLayerIMUdt(name="forking_layer")(net_in)

    imu_conv_1 = layers.Conv2D(filters=15, kernel_size=(conv_kernel_width, 3), strides=(1, 3), padding='same',
                               activation='relu', name='imu_conv_layer_1')(imu_stack)
    imu_conv_2 = layers.Conv2D(filters=30, kernel_size=(conv_kernel_width, 1), padding='same',
                               activation='relu', name='imu_conv_layer_2')(imu_conv_1)
    imu_conv_3 = layers.Conv2D(filters=60, kernel_size=(conv_kernel_width, 1), padding='same',
                               activation='relu', name='imu_conv_layer_3')(imu_conv_2)

    imu_conv_reduced = layers.Conv2D(filters=imu_final_channels, kernel_size=(conv_kernel_width, 1), padding='same',
                                     activation='tanh', name='imu_1x_conv_layer')(imu_conv_3)
    imu_conv_flat = layers.Reshape((window_len, imu_final_channels * 2, 1), name='reshape_layer')(imu_conv_reduced)
    re_stacked_imu = layers.Concatenate(name='imu_concatenating_layer', axis=2)([imu_conv_flat, time_diff_imu])

    imu_ts_conv = layers.Conv2D(filters=1+2*imu_final_channels, kernel_size=(conv_kernel_width, 1+2*imu_final_channels),
                                padding='valid', activation='relu', name='imu_final_conv_layer')(re_stacked_imu)

    flatten_conv_imu = layers.Flatten(name='imu_flattening_layer')(imu_ts_conv)
    flatten_state_0 = layers.Flatten(name='state_input_flattening_layer')(state_0)

    stacked = layers.Concatenate(name='data_concatenating_layer')([flatten_conv_imu, flatten_state_0])

    dense_1 = layers.Dense(400, name='dense_layer_1')(stacked)
    activation_1 = layers.Activation('relu', name='activation_1')(dense_1)

    dense_2 = layers.Dense(200, name='dense_layer_2')(activation_1)
    activation_2 = layers.Activation('relu', name='activation_2')(dense_2)

    dense_3 = layers.Dense(100, name='dense_layer_3')(activation_2)
    activation_3 = layers.Activation('relu', name='activation_3')(dense_3)

    net_out = layers.Dense(output_state_len, name='state_output')(activation_3)

    state_out = custom_layers.ExponentialRemappingLayer(name='remapped_state_output')(net_out)
    return Model(inputs=net_in, outputs=net_out), Model(inputs=net_in, outputs=state_out)


def pre_integration_net(args):

    window_len = args[0]
    n_iterations = args[1]

    # Define parameters for model
    kernel_width = min([window_len, 3])
    pooling_width = min([window_len, 2])

    input_state_shape = (10,)
    pre_int_shape = (window_len, 3)
    imu_input_shape = (window_len, 7, 1)

    # Input layers. Don't change names
    imu_in = layers.Input(imu_input_shape, name="imu_input")
    state_in = layers.Input(input_state_shape, name="state_input")

    # Pre-processing
    gyro, acc, dt_vec = custom_layers.PreProcessIMU()(imu_in)

    #############################
    # ##  TRAINABLE NETWORK  ## #
    #############################

    def norm_activate(inputs, activation, name=None, number=None):
        if name and number:
            inputs = layers.BatchNormalization(name=name + "_batchNorm_" + number)(inputs)
            inputs = layers.Activation(name=name + "_activation_" + number, activation=activation)(inputs)
        return inputs

    # Convolution layers
    def down_scaling_loop(x1, iterations, i, conv_channels):
        x_shrink = pre_int_shape[0] - (2 ** n_iterations) * round(pre_int_shape[0] / (2 ** n_iterations)) + 1
        if i == 0:
            x1 = layers.Conv2D(1, kernel_size=(x_shrink, 1), strides=(1, 1))(x1)
            
        conv_kernel = (kernel_width, 4)

        x2 = layers.Conv2D(conv_channels[i], kernel_size=conv_kernel, padding='same')(x1)
        x2 = norm_activate(x2, 'relu')
        x3 = layers.Conv2D(conv_channels[i], kernel_size=conv_kernel, padding='same')(x2)
        x3 = norm_activate(x3, 'relu')

        if iterations > 0:

            x_up = layers.MaxPooling2D(pool_size=(pooling_width, 1))(x3)
            x4 = layers.Conv2D(channels[-1], kernel_size=(1, 4))(x1)
            x4 = norm_activate(x4, 'relu')

            x_up = down_scaling_loop(x_up, iterations - 1, i + 1, conv_channels)
            x_up = layers.UpSampling2D(size=(pooling_width, 1))(x_up)

            x4 = tf.add(x4, x_up)

        else:
            x4 = layers.Conv2D(conv_channels[i], kernel_size=(1, 4))(x3)
            x4 = norm_activate(x4, 'relu')

        if i == 0:
            # Recover original shape
            x4 = layers.Conv2DTranspose(pre_int_shape[0], (x_shrink, 1))(x4)
            x4 = tf.squeeze(x4, axis=2)

        return x4

    channels = [2**i for i in range(4, 4 + n_iterations + 1)]
    gyro_feat_vec = down_scaling_loop(gyro, n_iterations, 0, channels)
    # acc_feat_vec = down_scaling_loop(acc, n_iterations, 0, channels)

    small_kernel = (pre_int_shape[1], pre_int_shape[1])
    
    # Pre-integrated rotation
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True), merge_mode='concat')(gyro_feat_vec)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True), merge_mode='concat')(x)
    x = layers.TimeDistributed(layers.Dense(pre_int_shape[1]))(x)
    pre_integrated_rot = layers.Flatten(name="pre_integrated_R")(x)

    # x = layers.Conv2D(10, kernel_size=small_kernel, padding='same', name='Rot_Branch')(feat_vec)
    # x = norm_activate(x, 'relu', "Rot_Branch", '1')
    # x = layers.Conv2D(1, kernel_size=(1, 1))(x)
    # x = norm_activate(x, 'relu', "Rot_Branch", '2')
    # pre_integrated_rot = tf.squeeze(x, axis=3, name="pre_integrated_R")
    #
    # # # Pre-integrated velocity
    # x = layers.Conv2D(window_len, kernel_size=small_kernel, padding='same', name="v_Branch")(feat_vec)
    # x = norm_activate(x, 'relu', 'v_Branch', '1')
    # x = layers.Conv2D(5, kernel_size=small_kernel, padding='same')(x)
    # x = norm_activate(x, 'relu', 'v_Branch', '2')
    # y = PreIntegrationForwardDense(pre_int_shape)([pre_integrated_rot, x])
    # y = norm_activate(y, 'relu', 'v_Branch', '3')
    # pre_integrated_v = tf.squeeze(y, axis=3, name="pre_integrated_v")
    #
    # # Pre-integrated position
    # x = layers.Conv2D(window_len, kernel_size=small_kernel, padding='same', name="p_Branch")(x)
    # x = norm_activate(x, 'relu', 'p_Branch', '1')
    # x = layers.Conv2D(5, kernel_size=small_kernel, padding='same')(x)
    # x = norm_activate(x, 'relu', 'p_Branch', '2')
    # y = PreIntegrationForwardDense(pre_int_shape)([(pre_integrated_rot, pre_integrated_v), x])
    # y = norm_activate(y, 'relu', 'p_Branch', '3')
    # pre_integrated_p = tf.squeeze(y, axis=3, name="pre_integrated_p")
    #
    # #################################
    # # ##  NON-TRAINABLE NETWORK  ## #
    # #################################
    #
    # x = FinalPreIntegration()([pre_integrated_rot, pre_integrated_v, pre_integrated_p])
    #
    # state_out = IntegratingLayer(name="state_output")([state_in, x, dt_vec])
    #
    # return Model(inputs=(imu_in, state_in), outputs=(pre_integrated_rot, pre_integrated_v, pre_integrated_p, state_out))

    return Model(inputs=(imu_in, state_in), outputs=(pre_integrated_rot))
