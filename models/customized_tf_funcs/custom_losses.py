import tensorflow as tf
import numpy as np
from utils.algebra import quaternion_error
from tensorflow.python.keras.losses import mean_squared_error


def l1_loss(y_true, y_pred):
    return tf.reduce_sum(tf.abs(tf.math.subtract(tf.cast(y_true, tf.float32), y_pred)), axis=1)


def state_loss(y_true, y_pred):

    pos_vel_contrib = mean_squared_error(y_true[:, :3], y_pred[:, :3]) + \
                      mean_squared_error(y_true[:, 3:6], y_pred[:, 3:6])
    att_contrib = 0
    try:
        # att_contrib += [np.sin(q.angle) for q in quaternion_error(y_true[:, 6:10], y_pred[:, 6:10])]
        att_contrib += mean_squared_error(y_true[:, 10:], y_pred[:, 10:])
    except TypeError:
        att_contrib = pos_vel_contrib

    return pos_vel_contrib + att_contrib
