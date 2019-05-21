from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as k_b
from models.customized_tf_funcs import custom_layers
import tensorflow as tf


def vel_cnn(window_len):
    input_s = (window_len, 6, 1)
    inputs = layers.Input(input_s, name="imu_input")
    x = layers.Conv2D(filters=60, kernel_size=(3, 6), padding='same', activation='relu', input_shape=input_s)(inputs)
    x = layers.Conv2D(filters=120, kernel_size=(3, 6), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=240, kernel_size=(3, 1), padding='valid', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(10, 1), strides=(6, 1))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(400, activation='relu')(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(1, name="state_output")(x)

    model = Model(inputs, x)

    return model


def pre_integration_net(args):

    window_len = args[0]
    n_iterations = args[1]

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

    channels = [2**i for i in range(3, 3 + n_iterations + 1)]
    final_shape = (pre_int_shape[0], pre_int_shape[1], channels[-1])

    gyro_feat_vec = down_scaling_loop(gyro, n_iterations, 0, channels, window_len, final_shape, n_iterations)
    acc_feat_vec = down_scaling_loop(acc, n_iterations, 0, channels, window_len, final_shape, n_iterations)
    feat_vec = layers.Concatenate()([gyro_feat_vec, acc_feat_vec])

    # Pre-integrated rotation
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), merge_mode='concat')(feat_vec)
    rot_prior = layers.LSTM(3, return_sequences=True)(x)
    pre_integrated_rot_flat = layers.Flatten(name="pre_integrated_R")(rot_prior)

    # Pre-integrated velocity
    x = layers.Conv2D(32, kernel_size=(2, 1), dilation_rate=(2, 1), padding='same')(k_b.expand_dims(rot_prior, axis=3))
    x = norm_activate(x, 'relu')
    x = layers.Conv2D(64, kernel_size=(2, 1), dilation_rate=(2, 1), padding='same')(x)
    x = norm_activate(x, 'relu')
    x = layers.Conv2D(1, kernel_size=(2, 1), dilation_rate=(2, 1), padding='same')(x)
    x = norm_activate(x, 'relu')
    x = layers.Reshape(pre_int_shape)(x)
    x = layers.Concatenate(axis=2)([x, feat_vec])
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), merge_mode='concat')(x)
    v_prior = layers.LSTM(3, return_sequences=True)(x)
    # v_prior = layers.TimeDistributed(layers.Dense(pre_int_shape[1]))(x)
    pre_integrated_v_flat = layers.Flatten(name="pre_integrated_v")(v_prior)

    # Pre-integrated position
    x = layers.Concatenate()([k_b.expand_dims(rot_prior, axis=3), k_b.expand_dims(v_prior, axis=3)])
    x = layers.Conv2D(32, kernel_size=(2, 1), dilation_rate=(2, 1), padding='same')(x)
    x = norm_activate(x, 'relu')
    x = layers.Conv2D(64, kernel_size=(2, 1), dilation_rate=(2, 1), padding='same')(x)
    x = norm_activate(x, 'relu')
    x = layers.Conv2D(1, kernel_size=(2, 1), dilation_rate=(2, 1), padding='same')(x)
    x = norm_activate(x, 'relu')
    x = layers.Reshape(pre_int_shape)(x)
    x = layers.Concatenate(axis=2)([x, feat_vec])
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), merge_mode='concat')(x)
    p_prior = layers.LSTM(3, return_sequences=True)(x)
    # p_prior = layers.TimeDistributed(layers.Dense(pre_int_shape[1]))(x)
    pre_integrated_p_flat = layers.Flatten(name="pre_integrated_p")(p_prior)

    #
    # #################################
    # # ##  NON-TRAINABLE NETWORK  ## #
    # #################################
    #
    # x = FinalPreIntegration()([pre_integrated_rot, pre_integrated_v, pre_integrated_p])
    #
    # state_out = IntegratingLayer(name="state_output")([state_in, x, dt_vec])
    #
    return Model(inputs=(imu_in, state_in), outputs=(pre_integrated_rot_flat, pre_integrated_v_flat, pre_integrated_p_flat))


def cnn_rnn_pre_int_net(window_len, n_iterations):
    input_state_shape = (10,)
    pre_int_shape = (window_len, 3)
    imu_input_shape = (window_len, 7, 1)
    b_norm = True

    # Input layers. Don't change names
    imu_in = layers.Input(imu_input_shape, name="imu_input")
    state_in = layers.Input(input_state_shape, name="state_input")

    gyro, acc, dt_vec = custom_layers.PreProcessIMU()(imu_in)

    # Convolution features
    channels = [2**i for i in range(2, 2 + n_iterations + 1)]
    final_shape = (pre_int_shape[0], pre_int_shape[1], channels[-1])

    gyro_feat_vec = down_scaling_loop(gyro, n_iterations, 0, channels, window_len, final_shape, n_iterations, b_norm)
    acc_feat_vec = down_scaling_loop(acc, n_iterations, 0, channels, window_len, final_shape, n_iterations, b_norm)

    # Pre-integrated rotation
    x = layers.GRU(64, return_sequences=True)(gyro_feat_vec)
    x = layers.TimeDistributed(layers.Dense(50, activation='relu'))(x)
    rot_prior = layers.TimeDistributed(layers.Dense(pre_int_shape[1]), name="pre_integrated_R")(x)

    # Pre-integrated velocity
    x = custom_layers.PreIntegrationForwardDense(pre_int_shape)(rot_prior)
    rot_contrib = norm_activate(x, 'leakyRelu', b_norm)
    v_feat_vec = layers.Concatenate()([gyro_feat_vec, acc_feat_vec, rot_contrib])
    x = layers.GRU(64, return_sequences=True)(v_feat_vec)
    x = layers.TimeDistributed(layers.Dense(50, activation='relu'))(x)
    v_prior = layers.TimeDistributed(layers.Dense(pre_int_shape[1]), name="pre_integrated_v")(x)

    # Pre-integrated position
    x = custom_layers.PreIntegrationForwardDense(pre_int_shape)(rot_prior)
    rot_contrib = norm_activate(x, 'leakyRelu', b_norm)
    x = custom_layers.PreIntegrationForwardDense(pre_int_shape)(v_prior)
    vel_contrib = norm_activate(x, 'leakyRelu', b_norm)
    pos_in = layers.Concatenate()([gyro_feat_vec, acc_feat_vec, rot_contrib, vel_contrib])
    x = layers.GRU(64, return_sequences=True)(pos_in)
    x = layers.TimeDistributed(layers.Dense(50, activation='relu'))(x)
    p_prior = layers.TimeDistributed(layers.Dense(pre_int_shape[1]), name="pre_integrated_p")(x)

    # return Model(inputs=(imu_in, state_in), outputs=(rot_prior, v_prior, p_prior))

    # slice tensors
    delta_rot = tf.squeeze(tf.slice(rot_prior, begin=[0, window_len-1, 0], size=[-1, 1, -1]), axis=1)
    delta_v = tf.squeeze(tf.slice(v_prior, begin=[0, window_len - 1, 0], size=[-1, 1, -1]), axis=1)
    delta_p = tf.squeeze(tf.slice(p_prior, begin=[0, window_len - 1, 0], size=[-1, 1, -1]), axis=1)
    delta_t = tf.reduce_sum(tf.squeeze(dt_vec), axis=1) / 1000

    state_out = custom_layers.IntegratingLayer(name="state_output")([state_in, delta_rot, delta_v, delta_p, delta_t])

    return Model(inputs=(imu_in, state_in), outputs=(rot_prior, v_prior, p_prior, state_out))


def fully_connected_net(args):
    window_len = args[0]

    input_state_shape = (10,)
    pre_int_shape = (window_len, 3)
    imu_input_shape = (window_len, 7, 1)

    # Input layers. Don't change names
    imu_in = layers.Input(imu_input_shape, name="imu_input")
    state_in = layers.Input(input_state_shape, name="state_input")

    _, _, dt_vec = custom_layers.PreProcessIMU()(imu_in)

    x = layers.Flatten()(imu_in)
    x = layers.Dense(200)(x)
    x = norm_activate(x, 'relu')
    x = layers.Dense(400)(x)
    x = norm_activate(x, 'relu')
    x = layers.Dense(400)(x)
    feat_vec = norm_activate(x, 'relu')

    r_flat = layers.Dense(tf.reduce_prod(pre_int_shape))(x)
    rot_prior = layers.Reshape(pre_int_shape, name="pre_integrated_R")(r_flat)

    x = layers.Concatenate()([feat_vec, r_flat])
    v_flat = layers.Dense(tf.reduce_prod(pre_int_shape))(x)
    v_prior = layers.Reshape(pre_int_shape, name="pre_integrated_v")(v_flat)

    x = layers.Concatenate()([feat_vec, r_flat, v_flat])
    p_flat = layers.Dense(tf.reduce_prod(pre_int_shape))(x)
    p_prior = layers.Reshape(pre_int_shape, name="pre_integrated_p")(p_flat)

    return Model(inputs=(imu_in, state_in), outputs=(rot_prior, v_prior, p_prior))


def norm_activate(inputs, activation, do_norm=True, name=None):
    if do_norm:
        inputs = layers.BatchNormalization()(inputs)
    if activation == 'leakyRelu':
        inputs = layers.LeakyReLU(name=name)(inputs)
    else:
        inputs = layers.Activation(name=name, activation=activation)(inputs)
    return inputs


# Convolution layers
def down_scaling_loop(x1, iterations, i, conv_channels, window_len, final_shape, max_iterations, b_norm):

    # Define parameters for model
    kernel_width = min([window_len, 3])
    pooling_width = min([window_len, 2])

    x_shrink = final_shape[0] - (2 ** max_iterations) * round(final_shape[0] / (2 ** max_iterations)) + 1
    if i == 0:
        x1 = layers.Conv2D(1, kernel_size=(x_shrink, 1), strides=(1, 1))(x1)

    conv_kernel = (kernel_width, 1)

    x2 = custom_layers.ReflectionPadding2D(padding=(0, 2))(x1)
    x2 = layers.Conv2D(conv_channels[i], kernel_size=conv_kernel, dilation_rate=(2, 1))(x2)
    x2 = norm_activate(x2, 'relu', b_norm)
    x2 = custom_layers.ReflectionPadding2D(padding=(0, 2))(x2)
    x3 = layers.Conv2D(conv_channels[i], kernel_size=conv_kernel, dilation_rate=(2, 1))(x2)
    x3 = norm_activate(x3, 'relu', b_norm)

    if iterations > 0:

        # Downscale
        x_down = layers.Conv2D(conv_channels[i], kernel_size=(2, 1), strides=(2, 1), padding='same')(x3)
        x4 = layers.Conv2D(final_shape[-1], kernel_size=(1, 4))(x1)
        x4 = norm_activate(x4, 'relu', b_norm)

        x_down = down_scaling_loop(x_down, iterations - 1, i + 1, conv_channels, window_len, final_shape, max_iterations, b_norm)
        x_up = layers.Conv2DTranspose(conv_channels[-1], kernel_size=conv_kernel, strides=(pooling_width, 1),
                                      padding='same')(x_down)

        x4 = tf.add(x4, x_up)

    else:
        x4 = layers.Conv2D(conv_channels[i], kernel_size=(1, 4))(x3)
        x4 = norm_activate(x4, 'relu', b_norm)

    if i == 0:
        # Recover original shape
        x4 = layers.Conv2DTranspose(final_shape[0], kernel_size=(x_shrink, 1))(x4)
        x4 = k_b.squeeze(x4, axis=2)

    return x4


def imu_integration_net(window_len, output_state_len):

    imu_final_channels = 5
    input_state_len = 10

    conv_kernel_width = min([window_len, 2])
    net_in = layers.Input((window_len, 7, 1), name="imu_input")
    state_0 = layers.Input((input_state_len, ), name="state_input")

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
    re_stacked_imu = layers.Concatenate(name='imu_concatenating_layer', axis=2)([imu_stack, imu_conv_flat, time_diff_imu])

    imu_ts_conv = layers.Conv2D(filters=7+2*imu_final_channels, kernel_size=(conv_kernel_width, 7+2*imu_final_channels),
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

    return Model(inputs=(net_in, state_0), outputs=net_out)
