import tensorflow as tf
from tensorflow.python.keras.losses import mean_squared_error
from utils.algebra import quaternion_error


def l1_loss(y_true, y_pred):
    abs_diff = tf.abs(tf.math.subtract(tf.cast(y_true, tf.float32), y_pred))

    while len(abs_diff.shape) > 1:
        abs_diff = tf.reduce_sum(abs_diff, axis=1)

    return abs_diff


def l2_loss(y_true, y_pred):
    l2_diff = tf.math.subtract(tf.cast(y_true, tf.float32), y_pred)**2

    return l2_diff


def pre_int_loss(weight):
    def loss_fn(y_true, y_pred):
        loss = tf.concat((mean_squared_error(y_true[:, :-1], y_pred[:, :-1]),
                          tf.expand_dims(mean_squared_error(y_true[:, -1], y_pred[:, -1]) * (1 + weight), axis=1)),
                         axis=-1)
        return loss
    return loss_fn


def mock_loss(y_true, _):
    if not y_true.shape[0]:
        return y_true
    return tf.zeros(y_true.shape[0])


def so3_loss_func(y_true, y_pred):

    pos_vel_contrib = l1_loss(y_true[:, :3], y_pred[:, :3]) + l1_loss(y_true[:, 3:6], y_pred[:, 3:6])
    att_contrib = 0
    try:
        att_contrib += mean_squared_error(y_true[:, 6:9], y_pred[:, 6:9])
    except TypeError:
        att_contrib = pos_vel_contrib

    return pos_vel_contrib + att_contrib


def state_loss(y_true, y_pred):
    pos_vel_contrib = l2_loss(y_true[:, :3], y_pred[:, :3]) + l2_loss(y_true[:, 3:6], y_pred[:, 3:6])

    try:
        att_contrib = [tf.abs(tf.math.sin(q.angle) for q in quaternion_error(y_true[:, 6:], y_pred[:, 6:]))]
    except TypeError:
        att_contrib = pos_vel_contrib

    return pos_vel_contrib + att_contrib
