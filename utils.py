import numpy as np
import os
import re
from tensorflow.python.keras.utils import Progbar
from pyquaternion import Quaternion


def get_checkpoint_file_list(checkpoint_dir, name):
    regex = name + r"_[0-9]"
    files = [f for f in os.listdir(checkpoint_dir) if re.match(regex, f)]
    files.sort(key=str.lower)
    return files


def imu_integration(data, window_len):

    # TODO: get a better comparison

    # Get features of training set. Discard final labels
    x = [np.squeeze(x_ds) for (x_ds, y_ds) in data]
    x_flat = np.array([item for sublist in x for item in sublist])

    samples = len(x_flat)
    output_dim = np.shape(x_flat)[1] - window_len

    out = np.zeros((samples, output_dim))

    imu_v, t_diff_v, x_0_v = x_flat[:, :window_len, :6], x_flat[:, :window_len, 6:], x_flat[:, window_len:, 0]

    # Convert time diff to seconds
    t_diff_v = np.squeeze(np.stack(t_diff_v/1000))

    bar = Progbar(samples)

    for sample in range(samples):
        bar.update(sample)

        t_diff = t_diff_v[sample, :]

        # Get initial states (world frame)
        x_i = x_0_v[sample, :3]
        v_i = x_0_v[sample, 3:6]
        q_i = Quaternion(x_0_v[sample, 6:]).unit

        for i in range(window_len):

            dt = t_diff[i]

            # Rotation body -> world
            w_R_b = q_i.inverse

            # Rotate angular velocity to world frame
            w_w = w_R_b.rotate(imu_v[sample, i, :3])
            # Rotate acceleration to world frame
            w_a = w_R_b.rotate(imu_v[sample, i, 3:]) + [0, 0, 9.81]

            # Integrate attitude (world frame)
            q_i.integrate(w_w, dt)

            # Integrate velocity
            v_i += w_a*dt

            # Integrate position
            x_i += v_i*dt + 1/2*w_a*(dt**2)

        out[sample, 0:3] = x_i
        out[sample, 3:6] = v_i
        out[sample, 6:] = q_i.elements

    return out


def rotate_quat(q1, q2):
    return q2 * q1.inverse


def quaternion_error(quat_1, quat_2, normalize=True):
    """
    Calculates the quaternion that rotates quaternion quat_1 to quaternion quat_2, or element-wise if given two lists of
    quaternions

    :param quat_1: initial quaternion (or list of quaternions) in numpy array format
    :param quat_2: target quaternion (or list of quaternions) in numpy array format
    :param normalize: whether quaternions should be normalized prior to the error calculation
    :return: the quaternion (or lists of quaternions) that transforms quat_1 to quat_2
    """

    if len(np.shape(quat_1)) == len(np.shape(quat_1)) == 2:
        if normalize:
            q_pred_e = [rotate_quat(Quaternion(quat_1[i]).unit, Quaternion(quat_2[i]).unit) for i in range(len(quat_1))]
        else:
            q_pred_e = [rotate_quat(Quaternion(quat_1[i]), Quaternion(quat_2[i])) for i in range(len(quat_1))]
    elif len(np.shape(quat_1)) == len(np.shape(quat_1)) == 1:
        if normalize:
            q_pred_e = rotate_quat(Quaternion(quat_1).unit, Quaternion(quat_2).unit)
        else:
            q_pred_e = rotate_quat(Quaternion(quat_1), Quaternion(quat_2))
    else:
        raise TypeError("quat_1 and quat_2 must be the same dimensions")

    return q_pred_e
