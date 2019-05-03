import numpy as np
from pyquaternion import Quaternion
from tensorflow.python.keras.utils import Progbar
import tensorflow as tf


def imu_integration(imu_data, window_len, track_progress=True):

    # TODO: get a better comparison

    samples = len(imu_data)
    output_dim = np.shape(imu_data)[1] - window_len

    out = np.zeros((samples, output_dim))

    imu_v, t_diff_v, x_0_v = imu_data[:, :window_len, :6], imu_data[:, :window_len, 6:], imu_data[:, window_len:, 0]

    # Convert time diff to seconds
    t_diff_v = np.squeeze(np.stack(t_diff_v/1000), axis=2)

    bar = Progbar(samples)

    for sample in range(samples):
        if track_progress:
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


def inv_rotate_quat(q1, q2):
    return q2 * q1.inverse


def rotate_quat(q1, q2):
    """
    Applies the rotation described by q2 to q1

    :param q1: Initial quaternion
    :param q2: Rotation quaternion
    :return: The rotated quaternion
    """

    if len(np.shape(q1)) == 2:
        if len(np.shape(q2)) == 2:
            return np.array([Quaternion(q2_i)*Quaternion(q1_i) for q1_i, q2_i in zip(q1, q2)])
        elif len(np.shape(q2)) == 1 and len(q2) == len(q1):
            np.array([Quaternion(q2)*Quaternion(q1_i) for q1_i in q1])
        else:
            raise TypeError("If the initial quaternion is a matrix, there must only be 1 rotation quaternion, "
                            "or exactly as many rotation quaternions as initial quaternions")

    elif len(np.shape(q1)) == 1:
        if len(np.shape(q2)) == 1:
            return Quaternion(q2)*Quaternion(q1)
        else:
            raise TypeError("If there is only an initial quaternion, there must only be one rotation quaternion")
    else:
        raise TypeError("The initial quaternion must be a vector or a 2D array")


def rotate_vec(v, q):
    if len(np.shape(v)) == 2:
        if len(np.shape(q)) == 2:
            return np.array([Quaternion(q_i).unit.rotate(v_i) for q_i, v_i in zip(q, v)])
        elif len(np.shape(q)) == 1 and len(q) == len(v):
            np.array([Quaternion(q).unit.rotate(v_i) for v_i in v])
        else:
            raise TypeError("If the vector is a matrix, there must only be 1 quaternion, "
                            "or exactly as many quaternions as vectors")

    elif len(np.shape(v)) == 1:
        if len(np.shape(q)) == 1:
            return Quaternion(q).unit.rotate(v)
        else:
            raise TypeError("If there is only a vector, there must only be one quaternion")
    else:
        raise TypeError("v should be a vector or a 2D array")


def unit_quat(q):
    if len(np.shape(q)) == 2:
        return [Quaternion(q_i).unit for q_i in q]
    elif len(np.shape(q)) == 1:
        return Quaternion(q).unit
    else:
        raise TypeError("input should be a 4 component array or an nx4 numpy matrix")


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
            q_pred_e = [inv_rotate_quat(unit_quat(quat_1[i]), unit_quat(quat_2[i])) for i in range(len(quat_1))]
        else:
            q_pred_e = [inv_rotate_quat(Quaternion(quat_1[i]), Quaternion(quat_2[i])) for i in range(len(quat_1))]
    elif len(np.shape(quat_1)) == len(np.shape(quat_1)) == 1:
        if normalize:
            q_pred_e = inv_rotate_quat(unit_quat(quat_1), unit_quat(quat_2))
        else:
            q_pred_e = inv_rotate_quat(Quaternion(quat_1), Quaternion(quat_2))
    else:
        raise TypeError("quat_1 and quat_2 must be the same dimensions")

    return q_pred_e


def correct_quaternion_flip(q_vec):
    """
    Makes sure that the quaternion is always positive inside a quaternion sequence

    :param q_vec: quaternion sequence (n quaternions)
    :return: the same quaternion sequence but all quaternions are positive rotations
    """

    as_numpy = isinstance(q_vec, np.ndarray)

    q_vec = unit_quat(q_vec)

    if as_numpy:
        q_vec = np.array([q.elements if q.w >= 0 else (-q).elements for q in q_vec])
    else:
        q_vec = np.array([q if q.w >= 0 else -q for q in q_vec])
    return q_vec


def log_mapping(q_vec):
    """
    Computes the Lie algebra so3 of the quaternion group SU2, or array of quaternions, via the logarithmic mapping

    :param q_vec: quaternion (or list of quaternions) in numpy array format
    :return: the Lie algebra so3 of the quaternion or array of quaternions
    """

    w = np.array([[0.0, 0.0, 0.0] if all(np.isclose(q.elements, [1.0, 0, 0, 0]))
                  else 2 * np.arccos(q.w) / np.linalg.norm(q.imaginary) * q.imaginary for q in unit_quat(q_vec)])

    return w


def exp_mapping(w_vec):
    """
    Computes the quaternion representation of the Lie algebra vector so3, or array of vectors, by exponential mapping

    :param w_vec: 3 component vector or array of vectors
    :return: the Lie group SU2 (in quaternion format) of the so3 Lie algebra
    """
    
    q_vec = np.array(
        [Quaternion().elements if all(np.isclose(list(w), [0, 0, 0]))
         else np.append(np.cos(np.linalg.norm(w)/2), np.sin(np.linalg.norm(w)/2)/np.linalg.norm(w)*w)
         for w in w_vec])

    return q_vec


def apply_state_diff(state, diff):
    """
    Applies a state differential to a state vector. State must have 10 dimensions (3 for position, 3 for velocity and 4
    for attitude quaternion), and can be a single vector or an array of them, along axis 0

    :param state: 10-dimensional initial state
    :param diff: 10-dimensional state differential

    :return: the new 10-dimensional state
    """

    assert np.shape(diff) == np.shape(state), "The state and the diff must be the same shape"

    state_out = None

    if len(np.shape(diff)) == 2:
        assert np.shape(diff)[1] == 10, "The state must be of length 10 (3 pos + 3 vel + 4 quaternion)"
        state_out = state[:, :6] + diff[:, :6]
        new_att = np.array([(q1 * q2).elements for q1, q2 in zip(unit_quat(diff[:, 6:]), unit_quat(state[:, 6:]))])
        state_out = tf.concat((state_out, new_att), axis=1)

    elif len(np.shape(diff)) == 1:
        assert len(diff) == 10, "The state must be of length 10 (3 pos + 3 vel + 4 quaternion)"
        state_out = state[:6] + diff[:6]
        state_out = tf.concat((state_out, np.array((unit_quat(diff[6:])*unit_quat(state[6.])).elements)))

    return state_out
