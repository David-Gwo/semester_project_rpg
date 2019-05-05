import tensorflow as tf
from tensorflow.python.keras.losses import mean_squared_error
from utils.algebra import quaternion_error
import numpy as np


def l1_loss(y_true, y_pred):
    abs_diff = tf.abs(tf.math.subtract(tf.cast(y_true, tf.float32), y_pred))

    while len(abs_diff.shape) > 1:
        abs_diff = tf.reduce_sum(abs_diff, axis=1)

    return abs_diff


def l2_loss(y_true, y_pred):
    abs_diff = tf.sqrt(tf.math.subtract(tf.cast(y_true, tf.float32), y_pred)**2)

    while len(abs_diff.shape) > 1:
        abs_diff = tf.reduce_sum(abs_diff, axis=1)

    return abs_diff


def mock_loss(y_true, _):
    return y_true * 0


def so3_loss_func(y_true, y_pred):

    pos_vel_contrib = l1_loss(y_true[:, :3], y_pred[:, :3]) + l1_loss(y_true[:, 3:6], y_pred[:, 3:6])
    att_contrib = 0
    try:
        att_contrib += mean_squared_error(y_true[:, 6:9], y_pred[:, 6:9])
    except TypeError:
        att_contrib = pos_vel_contrib

    return pos_vel_contrib + att_contrib


def state_loss(y_true, y_pred):

    pos_vel_contrib = l1_loss(y_true[:, :3], y_pred[:, :3]) + l1_loss(y_true[:, 3:6], y_pred[:, 3:6])
    try:
        att_contrib = [np.sin(q.angle) for q in quaternion_error(y_true[:, 6:], y_pred[:, 6:])]
    except TypeError:
        att_contrib = pos_vel_contrib

    return pos_vel_contrib + att_contrib
