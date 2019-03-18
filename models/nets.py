import tensorflow as tf
from tensorflow.python.keras import regularizers, Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D, MaxPooling2D, Reshape, \
    MaxPooling1D
from tensorflow.python.keras.layers.merge import add

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
        x1 = Conv2D(32, (5, 5), strides=[2, 2], padding='same')(img_input)
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

    with tf.name_scope("Convolution1"):
        model.add(Conv2D(filters=10, kernel_size=(20, 6), use_bias=False, padding='same', activation='relu', input_shape=(200, 6, 1)))

    with tf.name_scope("Convolution2"):
        model.add(Conv2D(filters=20, kernel_size=(20, 6), use_bias=False, padding='same', activation='relu'))

    with tf.name_scope("Convolution3"):
        model.add(Conv2D(filters=40, kernel_size=(20, 6), use_bias=False, padding='valid', activation='relu'))
        # model.add(MaxPooling2D(pool_size=(10, 1), strides=(20, 1)))
        model.add(Flatten())

    with tf.name_scope("Dense"):
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))

    with tf.name_scope("Output"):
        model.add(Dense(1))

    return model
