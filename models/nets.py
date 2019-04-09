import tensorflow as tf
from tensorflow.python.keras import regularizers, Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D, Conv1D, MaxPooling2D, \
    BatchNormalization, Bidirectional, LSTM, Concatenate, Reshape
from tensorflow.python.keras.layers.merge import add

from models.custom_layers import ForkLayer

##########################################
# ADD HERE YOUR NETWORK                  #
# BUILD IT WITH PURE TENSORFLOW OR KERAS #
##########################################

# NOTE: IF USING KERAS, YOU MIGHT HAVE PROBLEMS WITH BATCH-NORMALIZATION


def resnet8(img_input, output_dim, l2_reg_scale, scope='Prediction', reuse=False, log=False):
    """
    Define model architecture in Keras.

    # Arguments
       img_input: Batch of input images
       output_dim: Number of output dimensions (cardinality of classification)
       scope: Variable scope in which all variables will be saved
       reuse: Whether to reuse already initialized variables

    # Returns
       logits: Logits on output trajectories
    """

    img_input = Input(tensor=img_input)
    with tf.compat.v1.Variable.variable_scope(scope, reuse=reuse):
        x1 = Conv2D(32, (5, 5), strides=[2, 2], padding='same')(img_input)
        x1 = MaxPooling2D(pool_size=(3, 3), strides=[2, 2])(x1)

        # First residual block
        x2 = Activation('relu')(x1)
        x2 = Conv2D(32, (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x2)

        x2 = Activation('relu')(x2)
        x2 = Conv2D(32, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x2)

        x1 = Conv2D(32, (1, 1), strides=[2, 2], padding='same')(x1)
        x3 = add([x1, x2])

        # Second residual block
        x4 = Activation('relu')(x3)
        x4 = Conv2D(64, (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x4)

        x4 = Activation('relu')(x4)
        x4 = Conv2D(64, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x4)

        x3 = Conv2D(64, (1, 1), strides=[2, 2], padding='same')(x3)
        x5 = add([x3, x4])

        # Third residual block
        x6 = Activation('relu')(x5)
        x6 = Conv2D(128, (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x6)

        x6 = Activation('relu')(x6)
        x6 = Conv2D(128, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x6)

        x5 = Conv2D(128, (1, 1), strides=[2, 2], padding='same')(x5)
        x7 = add([x5, x6])

        x = Flatten()(x7)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # Output channel
        logits = Dense(output_dim)(x)

    if log:
        model = Model(inputs=[img_input], outputs=[logits])
        print(model.summary())

    return logits


def resnet8_model(input_shape, output_dim, l2_reg_scale, log=False):
    """
    Define model architecture in Keras.

    # Arguments
       img_input: Batch of input images
       output_dim: Number of output dimensions (cardinality of classification)
       scope: Variable scope in which all variables will be saved
       reuse: Whether to reuse already initialized variables

    # Returns
       logits: Logits on output trajectories
    """

    print(input_shape)
    with tf.name_scope("InputNode"):
        img_input = Input(shape=input_shape)
        x1 = Conv2D(32, (5, 5), strides=[2, 2], padding='same', )(img_input)
        x1 = MaxPooling2D(pool_size=(3, 3), strides=[2, 2])(x1)

    # First residual block
    with tf.name_scope("ResNode_1"):
        x2 = Activation('relu')(x1)
        x2 = Conv2D(32, (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x2)

        x2 = Activation('relu')(x2)
        x2 = Conv2D(32, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x2)

        x1 = Conv2D(32, (1, 1), strides=[2, 2], padding='same')(x1)
        x3 = add([x1, x2])

    # Second residual block
    with tf.name_scope("ResNode_2"):
        x4 = Activation('relu')(x3)
        x4 = Conv2D(64, (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x4)

        x4 = Activation('relu')(x4)
        x4 = Conv2D(64, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x4)

        x3 = Conv2D(64, (1, 1), strides=[2, 2], padding='same')(x3)
        x5 = add([x3, x4])

    # Third residual block
    with tf.name_scope("ResNode_3"):
        x6 = Activation('relu')(x5)
        x6 = Conv2D(128, (3, 3), strides=[2, 2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x6)

        x6 = Activation('relu')(x6)
        x6 = Conv2D(128, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_reg_scale))(x6)

        x5 = Conv2D(128, (1, 1), strides=[2, 2], padding='same')(x5)
        x7 = add([x5, x6])

    with tf.name_scope("OutputNode"):
        x = Flatten()(x7)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # Output channel
        logits = Dense(output_dim)(x)

    model = Model(inputs=[img_input], outputs=[logits])
    if log:
        print(model.summary())

    return model


def mnist_cnn():
    model = Sequential()

    # Must define the input shape in the first layer of the neural network
    with tf.name_scope("Conv_Node_1"):
        model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))

    with tf.name_scope("Conv_Node_2"):
        model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))

    with tf.name_scope("Flatten_Node"):
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

    with tf.name_scope("Logits_Node"):
        model.add(Dense(10, activation='softmax'))

    return model


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


def one_step_vel_net(imu_len):

    conv_kernel_width = min([imu_len, 3])
    pool_width = min([imu_len, 10])
    pool_strides = min([imu_len, 6])

    net_in = Input((imu_len, 7, 1), name="input_layer")
    imu_stack, v_vec = ForkLayer()(net_in)

    imu_conv_1 = Conv2D(filters=15, kernel_size=(conv_kernel_width, 1), padding='valid', activation='relu',
                        name='imu_conv_layer_1')(imu_stack)
    imu_conv_2 = Conv2D(filters=30, kernel_size=(conv_kernel_width, 1), padding='valid', activation='relu',
                        name='imu_conv_layer_2')(imu_conv_1)
    imu_conv_3 = Conv2D(filters=60, kernel_size=(conv_kernel_width, 3), strides=(1, 3), padding='valid',
                        activation='tanh', name='imu_conv_layer_3')(imu_conv_2)

    v_conv_1 = Conv1D(filters=15, kernel_size=conv_kernel_width, padding='valid', activation='relu',
                      name='vel_conv_layer_1')(v_vec)
    v_conv_2 = Conv1D(filters=30, kernel_size=conv_kernel_width, padding='valid', activation='relu',
                      name='vel_conv_layer_2')(v_conv_1)
    v_conv_3 = Conv1D(filters=60, kernel_size=conv_kernel_width, padding='valid', activation='tanh',
                      name='vel_conv_layer_3')(v_conv_2)

    flatten_imu = Flatten(name='imu_flattening_layer')(imu_conv_3)
    flatten_vel = Flatten(name='vel_flattening_layer')(v_conv_3)
    squeezed_vel = Reshape(name='reshape_vel_layer', target_shape=[imu_len])(v_vec)

    imu_conv_branch = Dropout(0.3)(flatten_imu)
    vel_conv_branch = Dropout(0.3)(flatten_vel)
    vel_branch = Dropout(0.3)(squeezed_vel)

    stacked = Concatenate(name='concatenation_layer')([imu_conv_branch, vel_conv_branch, vel_branch])

    dense_1 = Dense(100, name='dense_layer_1')(stacked)
    b_norm_1 = BatchNormalization(name='b_norm_1')(dense_1)
    activation_1 = Activation('relu', name='activation_1')(b_norm_1)

    dense_2 = Dense(100, name='dense_layer_2')(activation_1)
    b_norm_2 = BatchNormalization(name='b_norm_2')(dense_2)
    activation_2 = Activation('relu', name='activation_2')(b_norm_2)

    net_out = Dense(1, name='output_layer')(activation_2)

    model = Model(net_in, net_out)

    return model
