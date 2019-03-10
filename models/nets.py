import tensorflow as tf
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv2D, MaxPooling2D
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
