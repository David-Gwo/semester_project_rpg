import tensorflow as tf
import numpy as np
from utils.algebra import quaternion_error


def l2_loss(y_true, y_pred):
    return tf.reduce_sum(tf.abs(tf.math.subtract(tf.cast(y_true, tf.float32), y_pred)), axis=1)


def state_loss(y_true, y_pred):

    pos_vel_contrib = l2_loss(y_true[:, :3], y_pred[:, :3]) + l2_loss(y_true[:, 3:6], y_pred[:, 3:6])
    try:
        att_contrib = [np.sin(q.angle) for q in quaternion_error(y_true[:, 6:], y_pred[:, 6:])]
    except TypeError:
        att_contrib = pos_vel_contrib

    return pos_vel_contrib + att_contrib
