import numpy as np
from .utils.data_utils import save_train_and_test_datasets


def window_imu_data(imu_vec, window_len):

    window_channels = np.shape(imu_vec)[1]

    # Initialize x data. Will be sequence of IMU measurements of size (imu_len x 6)
    imu_img_tensor = np.zeros((len(imu_vec), window_len, window_channels, 1))

    for i in range(len(imu_vec) - window_len):
        imu_img = np.zeros((window_len, window_channels))

        # The first imu_x_len data vectors will not be full of data (not enough acquisitions to fill it up yet)
        if i < window_len:
            imu_img[window_len - i - 1:window_len, :] = imu_vec[0:i + 1, :]
        else:
            imu_img = imu_vec[i:i + window_len, :]

        # TODO: Should the elapsed time be included in the data?

        imu_img_tensor[i, :, :, :] = np.expand_dims(imu_img, 2)

    return imu_img_tensor


def imu_img_dataset(imu_vec, gt_vec, imu_len):
    """
    Generates a dataset of imu images to regress linear speed. An imu image is defined as a stack of matrices of
    dimensions <imu_len x 6>, where 6 are the 6 dimensions of the IMU readings (3 gyro + 3 acc), and the number of
    rows are the number of used imu samples to regress the speed corresponding to the last sample.
    :param imu_vec: vector of ordered IMU readings. Shape: <n, 2, 3>, n = number of acquisitions, imu_vec[:, 0, :]
    corresponds to the three gyro readings and imu_vec[:, 1, :] to the tree accelerometer readings
    :param gt_vec: ground truth velocity data. Shape: <n, 3>, n = number of acquisitions, and each acquisition is a
    3-dimensional vector with x,y,z velocities
    :param imu_len: number of columns of the imu_image
    :return: the constructed dataset following the above indications, in the format imu_img_tensor, gt_tensor
    """

    # Initialize y data. Will be the absolute ground truth value of the speed of the drone
    gt_v_tensor = np.linalg.norm(gt_vec, axis=1)

    imu_img_tensor = window_imu_data(imu_vec, imu_len)

    return imu_img_tensor, gt_v_tensor


def imu_gt_vel_dataset(imu_vec, gt_vec, imu_len):
    """
    Generates a dataset of combine imu and ground truth velocity data to do one-step speed prediction. The training data
    is defined as a stack of matrices of dimensions <imu_len x 7>, where the first 6 of the 7 columns are the IMU
    readings (3 gyro + 3 acc), and the 7th is the corresponding ground truth velocity. The number of rows correspond to
    the number of used imu samples to predict the speed for next timestamp.
    :param imu_vec: vector of ordered IMU readings. Shape: <n, 2, 3>, n = number of acquisitions, imu_vec[:, 0, :]
    corresponds to the three gyro readings and imu_vec[:, 1, :] to the tree accelerometer readings
    :param gt_vec: ground truth velocity data. Shape: <n, 3>, n = number of acquisitions, and each acquisition is a
    3-dimensional vector with x,y,z velocities
    :param imu_len: number of columns of the imu_image
    :return: the constructed dataset following the above indications, in the format imu_img_tensor, gt_tensor
    """

    seq_len = len(imu_vec)

    # Initialize x data. Will be sequence of IMU measurements of size (imu_len x 6)
    imu_img_tensor = np.zeros((seq_len - 1, imu_len, 7, 1))
    # Initialize y data. Will be the absolute ground truth value of the speed of the drone
    gt_v_tensor = np.zeros(seq_len - 1)

    gt_vec = np.expand_dims(np.linalg.norm(gt_vec, axis=1), axis=1)
    imu_vec = np.append(imu_vec, np.zeros((imu_len - 1, 2, 3)), axis=0)

    for i in range(seq_len - imu_len - 1):
        imu_img = np.append(imu_vec[i:i + imu_len, :, :].reshape(imu_len, 6), gt_vec[i:i + imu_len], axis=1)

        imu_img_tensor[i] = np.expand_dims(imu_img, 2)

        gt_v_tensor[i] = gt_vec[i + imu_len]

    return imu_img_tensor, gt_v_tensor


def reformat_data(compact_data):

    data_channels = np.shape(compact_data)[1] - 1

    # Reshape to 2D array
    flattened_data = np.concatenate([np.stack(compact_data[:, i], axis=0) for i in range(data_channels)], axis=1)
    flattened_data = np.append(flattened_data, np.expand_dims(compact_data[:, data_channels], axis=1), axis=1)

    # Calculate difference between timestamps, and change units to ms
    flattened_data[1:, -1] = np.diff(flattened_data[:, -1]) / 1000
    flattened_data[0, -1] = 0

    return flattened_data


def windowed_imu_integration_dataset(raw_imu, gt, window_len):

    raw_imu = reformat_data(raw_imu)
    gt = reformat_data(gt)

    imu_channels = np.shape(raw_imu)[1] - 1
    n_samples = len(raw_imu)

    # Keep only position, attitude, velocity information (remove angular velocity, acceleration and timestamp)
    gt = np.delete(gt, np.s_[10:17], axis=1)
    kept_channels = np.shape(gt)[1]

    # Copy the first row window_len times at the beginning of the dataset
    gt_augmented = np.append(np.ones((window_len, 1))*np.expand_dims(gt[0, :], axis=0), gt, axis=0)

    # Add the initial state of the window at the beginning of each training sequence
    zero_padded_gt = np.zeros((n_samples, (imu_channels + 1) * kept_channels))
    zero_padded_gt[:, :kept_channels] = gt
    zero_padded_gt = zero_padded_gt.reshape((n_samples, kept_channels, imu_channels + 1, 1), order='F')
    imu_window_with_initial_state = np.zeros((n_samples, window_len + kept_channels, imu_channels + 1, 1))
    imu_window_with_initial_state[:, 0:window_len, :, :] = window_imu_data(raw_imu, window_len)
    imu_window_with_initial_state[:, window_len:, :, :] = zero_padded_gt

    # The ground truth data to be predicted is the state at the end of the window
    return imu_window_with_initial_state, gt_augmented[window_len:, :]


def generate_dataset(raw_imu, gt, ds_dir, train_file_name, test_file_name, dataset_type, *args):
    """
    Generates training and testing datasets, and saves a copy of them

    :param raw_imu: 3D array of IMU measurements (n_samples x 2 <gyro, acc> x 3 <x, y, z>)
    :param gt: list of 3D arrays with the ground truth measurements
    :param ds_dir: root directory of the dataset
    :param train_file_name: Name of the preprocessed training dataset
    :param test_file_name: Name of the preprocessed testing dataset
    :param dataset_type: Type of dataset to be generated
    :param args: extra arguments for dataset generation
    """

    if dataset_type == "imu_img_gt_vel":
        imu_img_tensor, gt_tensor = imu_gt_vel_dataset(raw_imu, gt, *args)
    elif dataset_type == "windowed_imu_integration":
        imu_img_tensor, gt_tensor = windowed_imu_integration_dataset(raw_imu, gt, *args)
    else:
        imu_img_tensor, gt_tensor = imu_img_dataset(raw_imu, gt, *args)

    euroc_training_ds = "{0}{1}".format(ds_dir, train_file_name)
    euroc_testing_ds = "{0}{1}".format(ds_dir, test_file_name)
    save_train_and_test_datasets(euroc_training_ds, euroc_testing_ds, imu_img_tensor, gt_tensor)
