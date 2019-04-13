import numpy as np
from .utils.data_utils import save_train_and_test_datasets


def generate_imu_img_dataset(imu_vec, gt_vec, imu_len):
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

    # Initialize x data. Will be sequence of IMU measurements of size (imu_len x 6)
    imu_img_tensor = np.zeros((len(imu_vec), imu_len, 6, 1))
    # Initialize y data. Will be the absolute ground truth value of the speed of the drone
    gt_v_tensor = np.zeros(len(imu_vec))

    for i in range(len(imu_vec) - imu_len):
        imu_img = np.zeros((imu_len, 6))

        # The first imu_x_len data vectors will not be full of data (not enough acquisitions to fill it up yet)
        if i < imu_len:
            imu_img[imu_len - i - 1:imu_len, :] = imu_vec[0:i + 1, :, :].reshape(i + 1, 6)
        else:
            imu_img = imu_vec[i:i + imu_len, :, :].reshape(imu_len, 6)

        # TODO: Should the elapsed time be included in the data?

        imu_img_tensor[i, :, :, :] = np.expand_dims(imu_img, 2)
        gt_v_tensor[i] = np.linalg.norm(gt_vec[i])

        return imu_img_tensor, gt_v_tensor


def generate_imu_gt_vel_dataset(imu_vec, gt_vec, imu_len):
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


def generate_dataset(raw_imu, gt_v, ds_dir, train_file_name, test_file_name, dataset_type, *args):
    """
    Generates training and testing datasets, and saves a copy of them

    :param raw_imu: 3D array of IMU measurements (n_samples x 2 <gyro, acc> x 3 <x, y, z>)
    :param gt_v: list of 3D arrays with the decomposed velocity ground truth measurements
    :param ds_dir: root directory of the dataset
    :param train_file_name: Name of the preprocessed training dataset
    :param test_file_name: Name of the preprocessed testing dataset
    :param dataset_type: Type of dataset to be generated
    :param args: extra arguments for dataset generation
    """

    if dataset_type == "imu_img_gt_vel":
        imu_img_tensor, gt_v_tensor = generate_imu_gt_vel_dataset(raw_imu, gt_v, *args)
    else:
        imu_img_tensor, gt_v_tensor = generate_imu_img_dataset(raw_imu, gt_v, *args)

    euroc_training_ds = "{0}{1}".format(ds_dir, train_file_name)
    euroc_testing_ds = "{0}{1}".format(ds_dir, test_file_name)
    save_train_and_test_datasets(euroc_training_ds, euroc_testing_ds, imu_img_tensor, gt_v_tensor)
