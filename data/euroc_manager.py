import csv
import yaml
import numpy as np
import tensorflow as tf
import os
from scipy.signal import butter as butterworth_filter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from data.inertial_dataset_manager import IMU, GT
from data.utils.data_utils import load_mat_data, filter_with_coeffs
from data.imu_dataset_generators import generate_dataset

SCALER_GYRO_FILE = "scaler_gyro.save"
SCALER_ACC_FILE = "scaler_acc.save"
SCALER_DIR_FILE = '/scaler_files_dir.txt'

# TODO: refactor as class, such as blackbird_manager.py
# TODO: move common functions to more general class (e.g. inertial_dataset_manager.py) and inherit from it


def read_euroc_dataset(euroc_dir):
    imu_file = euroc_dir + 'mav0/imu0/data.csv'
    imu_yaml_file = euroc_dir + 'mav0/imu0/sensor.yaml'
    ground_truth_file = euroc_dir + 'mav0/state_groundtruth_estimate0/data.csv'

    imu_yaml_data = dict()
    raw_imu_data = []
    ground_truth_data = []

    try:
        # Read IMU data
        with open(imu_file, 'rt') as csv_file:
            header_line = 1
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                if header_line:
                    header_line = 0
                    continue

                imu = IMU()
                imu.read(row)
                raw_imu_data.append(imu)

        # Read IMU sensor yaml file
        with open(imu_yaml_file, 'r') as stream:
            imu_yaml_data = yaml.load(stream)

        # Read ground truth data
        with open(ground_truth_file, 'rt') as csv_file:
            header_line = 1
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                if header_line:
                    header_line = 0
                    continue

                gt = GT()
                gt.read(row)
                ground_truth_data.append(gt)

    except IOError:
        print("Dataset file not found")

    except yaml.YAMLError as exc:
        print(exc)

    return [imu_yaml_data, raw_imu_data, ground_truth_data]





def pre_process_data(raw_imu_data, gt_interp, euroc_dir):
    """
    Pre-process euroc dataset (apply low-pass filter and minmax scaling)

    :param raw_imu_data: 1D array of IMU objects with IMU measurements
    :param gt_interp: list of 3D arrays with the decomposed velocity ground truth measurements
    :param euroc_dir: root directory of the euroc data
    :return: the filtered dataset
    """

    # Transform the data to numpy matrices
    imu_unroll = np.array([(imu_s.unroll()) for imu_s in raw_imu_data])
    gt_unroll = np.array([(gt_meas.unroll()) for gt_meas in gt_interp])

    # Get number of channels per data type (we subtract 1 because timestamp is not a channel we want to filter)
    imu_channels = np.shape(imu_unroll)[1] - 1
    gt_channels = np.shape(gt_unroll)[1] - 1

    # TODO: pass sampling frequency as param

    # Design butterworth filter
    fs = 100.0  # Sample frequency (Hz)
    f0 = 10.0  # Frequency to be removed from signal (Hz)
    w0 = f0 / (fs / 2)  # Normalized Frequency
    [b_bw, a_bw] = butterworth_filter(10, w0, output='ba')

    filt_imu_vec = np.stack([filter_with_coeffs(a_bw, b_bw, imu_unroll[:, i], fs) for i in range(imu_channels)], axis=1)
    filt_gt_vec = np.stack([filter_with_coeffs(a_bw, b_bw, gt_unroll[:, i], fs) for i in range(gt_channels)], axis=1)

    scale_g = MinMaxScaler()
    scale_g.fit(np.stack(filt_imu_vec[:, 0]))
    scale_a = MinMaxScaler()
    scale_a.fit(np.stack(filt_imu_vec[:, 1]))

    joblib.dump(scale_g, euroc_dir + SCALER_GYRO_FILE)
    joblib.dump(scale_a, euroc_dir + SCALER_ACC_FILE)

    # Add back the timestamps to the data matrix and return
    filt_imu_vec = np.append(filt_imu_vec, np.expand_dims(imu_unroll[:, -1], axis=1), axis=1)
    filt_gt_vec = np.append(filt_gt_vec, np.expand_dims(gt_unroll[:, -1], axis=1), axis=1)

    return filt_imu_vec, filt_gt_vec


def generate_tf_imu_train_ds(euroc_dir, euroc_train, batch_s, trained_model_dir, window_len):
    """
    Read the processed euroc dataset from the saved file. Generate the tf-compatible train/validation datasets

    :param euroc_dir: root directory of the euroc data
    :param euroc_train: Name of the preprocessed euroc training dataset
    :param batch_s: (mini)-batch size of datasets
    :param trained_model_dir: Name of the directory where trained model will be stored
    :param window_len: length of the sampling window
    :return: the tf-compatible training and validation datasets, and their respective lengths
    """

    seed = 8901

    train_filename = euroc_dir + euroc_train

    imu_img_tensor, gt_tensor = load_mat_data(train_filename)

    file = open(trained_model_dir + SCALER_DIR_FILE, "r")
    scaler_dir = file.read()

    scale_g = joblib.load(scaler_dir + SCALER_GYRO_FILE)
    scale_a = joblib.load(scaler_dir + SCALER_ACC_FILE)

    for i in range(window_len):
        imu_img_tensor[:, i, 0:3, 0] = scale_g.transform(imu_img_tensor[:, i, 0:3, 0])
        imu_img_tensor[:, i, 3:6, 0] = scale_a.transform(imu_img_tensor[:, i, 3:6, 0])

    full_ds_len = len(gt_tensor)
    val_ds_len = np.ceil(full_ds_len * 0.1)
    train_ds_len = full_ds_len - val_ds_len

    val_ds_indexes = np.random.choice(range(full_ds_len), int(val_ds_len), replace=False)
    val_ds_imu_vec = imu_img_tensor[val_ds_indexes]
    val_ds_v_vec = gt_tensor[val_ds_indexes]

    imu_img_tensor = np.delete(imu_img_tensor, val_ds_indexes, axis=0)
    gt_tensor = np.delete(gt_tensor, val_ds_indexes, axis=0)

    full_train_ds = tf.data.Dataset.from_tensor_slices((imu_img_tensor, gt_tensor)).shuffle(batch_s, seed=seed)
    val_ds = tf.data.Dataset.from_tensor_slices((val_ds_imu_vec, val_ds_v_vec)).batch(batch_s)
    train_ds = full_train_ds.batch(batch_s).repeat()

    return train_ds, val_ds, (train_ds_len, val_ds_len)


def generate_tf_imu_test_ds(euroc_dir, euroc_test, batch_s, trained_model_dir, window_len, normalize=True,
                            full_batches=False):
    """
    Read the preprocessed euroc dataset from saved file. Generate the tf-compatible testing dataset

    :param euroc_dir: root directory of the euroc data
    :param euroc_test:
    :param batch_s: (mini)-batch size of datasets
    :param trained_model_dir: Name of the directory where trained model is stored
    :param window_len: length of the sampling window
    :param normalize: whether data should be normalized using scaler factors
    :param full_batches: whether batches should all be the same size
    :return: the tf-compatible testing dataset, and its length
    """

    test_filename = euroc_dir + euroc_test

    imu_img_tensor, gt_v_tensor = load_mat_data(test_filename)

    if normalize:
        file = open(trained_model_dir + SCALER_DIR_FILE, "r")
        scaler_dir = file.read()

        scale_g = joblib.load(scaler_dir + SCALER_GYRO_FILE)
        scale_a = joblib.load(scaler_dir + SCALER_ACC_FILE)

        for i in range(window_len):
            imu_img_tensor[:, i, 0:3, 0] = scale_g.transform(imu_img_tensor[:, i, 0:3, 0])
            imu_img_tensor[:, i, 3:6, 0] = scale_a.transform(imu_img_tensor[:, i, 3:6, 0])

    ds_len = len(gt_v_tensor)

    test_ds = tf.data.Dataset.from_tensor_slices((imu_img_tensor, gt_v_tensor))
    test_ds = test_ds.batch(batch_s, drop_remainder=full_batches)

    return test_ds, ds_len


def add_scaler_ref_to_training_dir(root, destiny):
    """
    Adds a txt file at the training directory with the location of the scaler functions used to transform the data that
    created the model for the first time

    :param root: Directory of scaler objects
    :param destiny: Training directory
    """

    if not os.path.exists(destiny):
        os.mkdir(destiny)
    if not os.path.exists(destiny + SCALER_DIR_FILE):
        os.mknod(destiny + SCALER_DIR_FILE)

        file = open(destiny + SCALER_DIR_FILE, 'w')

        file.write(root)
        file.close()


def load_euroc_dataset(euroc_dir, batch_size, imu_seq_len, euroc_train, euroc_test, processed_ds_available,
                       trained_model_dir):
    """
    Read, interpolate, pre-process and generate euroc dataset for speed regression in tensorflow.

    :param euroc_dir: root directory of the EuRoC dataset
    :param batch_size: mini-batch size
    :param imu_seq_len: Number of IMU measurements in the x vectors. Is a function of the sampling frequency of the IMU
    :param euroc_train: Name of the file where to store the preprocessed euroc training dataset
    :param euroc_test: Name of the file where to store the preprocessed euroc testing dataset
    :param processed_ds_available: Whether there is already a processed dataset file available to load from
    :param trained_model_dir: Name of the directory where trained model will be stored
    :return: the tf-compatible training and validation datasets, and their respective lengths
    """

    if not processed_ds_available:
        imu_yaml_data, raw_imu_data, ground_truth_data = read_euroc_dataset(euroc_dir)

        raw_imu_data, gt_interp = interpolate_ground_truth(raw_imu_data, ground_truth_data)

        # TODO: fix gt_interp -> gt_v_interp

        processed_imu, processed_v = pre_process_data(raw_imu_data, gt_v_interp, euroc_dir)

        generate_dataset(processed_imu, processed_v, euroc_dir, euroc_train, euroc_test, "imu_img_gt_vel", imu_seq_len)

    add_scaler_ref_to_training_dir(euroc_dir, trained_model_dir)

    return generate_tf_imu_train_ds(euroc_dir, euroc_train, batch_size, trained_model_dir)


def expand_dataset_region(filt_imu_vec, filt_gt_v_interp):
    # TODO: remove this function
    # Add more flat region so avoid model from learning average value
    flat_region = filt_imu_vec[6000:7000, :, :]
    flat_region_v = filt_gt_v_interp[6000:7000, :]
    flat_region = np.repeat(np.concatenate((flat_region, flat_region[::-1, :, :])), [4], axis=0)
    flat_region_v = np.repeat(np.concatenate((flat_region_v, flat_region_v[::-1])), [4], axis=0)
    filt_imu_vec = np.concatenate((filt_imu_vec[0:6000, :, :], flat_region, filt_imu_vec[7000:, :, :]))
    filt_gt_v_interp = np.concatenate((filt_gt_v_interp[0:6000, :], flat_region_v, filt_gt_v_interp[7000:, :]))

    return filt_imu_vec, filt_gt_v_interp
