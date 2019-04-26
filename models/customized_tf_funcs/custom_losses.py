import tensorflow as tf
from tensorflow.python.keras.losses import mean_squared_error


def l1_loss(y_true, y_pred):
    return tf.reduce_sum(tf.abs(tf.math.subtract(tf.cast(y_true, tf.float32), y_pred)), axis=1)


def net_loss_fx(y_true, y_pred):

    pos_vel_contrib = mean_squared_error(y_true[:, :3], y_pred[:, :3]) + \
                      mean_squared_error(y_true[:, 3:6], y_pred[:, 3:6])
    att_contrib = 0
    try:
        att_contrib += mean_squared_error(y_true[:, 6:9], y_pred[:, 6:9])
    except TypeError:
        att_contrib = pos_vel_contrib

    return pos_vel_contrib + att_contrib


def state_loss_fx(_, y_pred):
    """
    Dummy loss function

    :param _:
    :param y_pred:
    :return:
    """

    if not y_pred.shape[0]:
        return y_pred

    return tf.zeros(y_pred.shape)
