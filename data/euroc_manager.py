import csv
import yaml
import numpy as np
import tensorflow as tf
import os
from scipy.signal import butter as butterworth_filter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from data.utils.data_utils import load_mat_data, interpolate_ts, filter_with_coeffs
from data.imu_dataset_generators import generate_dataset


class IMU:
    def __init__(self):
        self.timestamp = 0.0
        self.gyro = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])

    def read(self, data):
        data = np.array(data)
        data = data.astype(np.float)
        self.timestamp = data[0]
        self.gyro = data[1:4]
        self.acc = data[4:7]

    def unroll(self):
        return self.gyro, self.acc, self.timestamp


class GT:
    def __init__(self):
        self.timestamp = 0.0
        self.pos = np.array([0.0, 0.0, 0.0])
        self.att = np.array([0.0, 0.0, 0.0, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.ang_vel = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])

    def read(self, data):
        data = np.array(data)
        data = data.astype(np.float)
        self.timestamp = data[0]
        self.pos = data[1:4]
        self.att = data[4:8]
        self.vel = data[8:11]
        self.ang_vel = data[11:14]
        self.acc = data[14:17]

    def read_from_tuple(self, data):
        self.pos = data[0]
        self.vel = data[1]
        self.att = data[2]
        self.ang_vel = data[3]
        self.acc = data[4]
        self.timestamp = data[5]
        return self

    def unroll(self):
        return self.pos, self.vel, self.att, self.ang_vel, self.acc, self.timestamp


SCALER_GYRO_FILE = "scaler_gyro.save"
SCALER_ACC_FILE = "scaler_acc.save"
SCALER_DIR_FILE = '/scaler_files_dir.txt'

# TODO: refactor as class, such as blackbird_manager.py
# TODO: move common functions to more general class (e.g. dataset_manager.py) and inherit from it


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


def interpolate_ground_truth(raw_imu_data, ground_truth_data):
    """
    Interpolates the data of the ground truth so that it matches the timestamps of the raw imu data

    :param raw_imu_data: IMU data (an array of IMU objects)
    :param ground_truth_data: ground truth velocity data (an array of GT objects)
    :return: the original imu data, and the interpolated ground truth data
    """
    raw_imu_data = np.array(raw_imu_data)

    imu_timestamps = np.array([imu_meas.timestamp for imu_meas in raw_imu_data])
    gt_unroll = np.array([(gt_meas.unroll()) for gt_meas in ground_truth_data])

    gt_pos = np.stack(gt_unroll[:, 0])
    gt_vel = np.stack(gt_unroll[:, 1])
    gt_att = np.stack(gt_unroll[:, 2])
    gt_ang_vel = np.stack(gt_unroll[:, 3])
    gt_acc = np.stack(gt_unroll[:, 4])
    gt_timestamps = gt_unroll[:, 5]

    # Only keep imu data that is within the ground truth time span
    raw_imu_data = raw_imu_data[(imu_timestamps > gt_timestamps[0]) * (imu_timestamps < gt_timestamps[-1])]
    imu_timestamps = np.array([imu_meas.timestamp for imu_meas in raw_imu_data])

    # Interpolate Ground truth to match IMU time acquisitions
    gt_pos_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_pos)
    gt_vel_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_vel)
    gt_att_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_att, is_quaternion=True)
    gt_ang_vel_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_ang_vel)
    gt_acc_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_acc)

    return [raw_imu_data, (gt_pos_interp, gt_vel_interp, gt_att_interp, gt_ang_vel_interp, gt_acc_interp, imu_timestamps)]


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


def generate_tf_imu_test_ds(euroc_dir, euroc_test, batch_s, trained_model_dir, window_len, normalize=True):
    """
    Read the preprocessed euroc dataset from saved file. Generate the tf-compatible testing dataset

    :param euroc_dir: root directory of the euroc data
    :param euroc_test:
    :param batch_s: (mini)-batch size of datasets
    :param trained_model_dir: Name of the directory where trained model is stored
    :param window_len: length of the sampling window
    :param normalize: whether data should be normalized using scaler factors
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
    test_ds = test_ds.batch(batch_s)

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


def plot_all_data(imu_vec, gt_vec, title="", from_numpy=False, show=False):
    """
    Plots the imu and ground truth data in two separate figures

    :param imu_vec: vector of imu data (either vector of IMU, or 2d numpy array)
    :param gt_vec: vector of ground truth data (either vector of GT, or 2d numpy array)
    :param title: title of the plot
    :param from_numpy: format of the input data
    :param show: whether to show plot or not
    :return:
    """

    if from_numpy:
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(np.stack(imu_vec[:, 0]))
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(np.stack(imu_vec[:, 1]))
        fig.suptitle(title)

        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        ax.plot(np.stack(gt_vec[:, 0]))
        ax = fig.add_subplot(2, 2, 2)
        ax.plot(np.stack(gt_vec[:, 1]))
        ax = fig.add_subplot(2, 2, 3)
        ax.plot(np.stack(gt_vec[:, 2]))
        ax = fig.add_subplot(2, 2, 4)
        ax.plot(np.stack(gt_vec[:, 3]))
        fig.suptitle(title)

    else:
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.plot([imu.gyro for imu in imu_vec])
        ax = fig.add_subplot(2, 1, 2)
        ax.plot([imu.acc for imu in imu_vec])
        fig.suptitle(title)

        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        ax.plot([gt.pos for gt in gt_vec])
        ax = fig.add_subplot(2, 2, 2)
        ax.plot([gt.vel for gt in gt_vec])
        ax = fig.add_subplot(2, 2, 3)
        ax.plot([gt.att for gt in gt_vec])
        ax = fig.add_subplot(2, 2, 4)
        ax.plot([gt.ang_vel for gt in gt_vec])
        fig.suptitle(title)

    if show:
        plt.show()


def plot_prediction(gt, prediction, manual_pred):

    if manual_pred is not None:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(3, 1, 1)
        ax2 = fig1.add_subplot(3, 1, 2)
        ax3 = fig1.add_subplot(3, 1, 3)

        ax1.plot(gt[:, 0], 'b')
        ax1.plot(prediction[:, 0], 'r')
        ax1.plot(manual_pred[:, 0], 'k')
        ax2.plot(gt[:, 1], 'b')
        ax2.plot(prediction[:, 1], 'r')
        ax2.plot(manual_pred[:, 1], 'k')
        ax3.plot(gt[:, 2], 'b')
        ax3.plot(prediction[:, 2], 'r')
        ax3.plot(manual_pred[:, 2], 'k')
        ax1.set_title('pos_x')
        ax2.set_title('pos_y')
        ax3.set_title('pos_z')

        fig2 = plt.figure()
        ax1 = fig2.add_subplot(3, 1, 1)
        ax2 = fig2.add_subplot(3, 1, 2)
        ax3 = fig2.add_subplot(3, 1, 3)

        ax1.plot(gt[:, 3], 'b')
        ax1.plot(prediction[:, 3], 'r')
        ax1.plot(manual_pred[:, 3], 'k')
        ax2.plot(gt[:, 4], 'b')
        ax2.plot(prediction[:, 4], 'r')
        ax2.plot(manual_pred[:, 4], 'k')
        ax3.plot(gt[:, 5], 'b')
        ax3.plot(prediction[:, 5], 'r')
        ax3.plot(manual_pred[:, 5], 'k')
        ax1.set_title('vel_x')
        ax2.set_title('vel_y')
        ax3.set_title('vel_z')

        fig3 = plt.figure()
        ax1 = fig3.add_subplot(4, 1, 1)
        ax2 = fig3.add_subplot(4, 1, 2)
        ax3 = fig3.add_subplot(4, 1, 3)
        ax4 = fig3.add_subplot(4, 1, 4)

        ax1.plot(gt[:, 6], 'b')
        ax1.plot(prediction[:, 6], 'r')
        ax1.plot(manual_pred[:, 6], 'k')
        ax2.plot(gt[:, 7], 'b')
        ax2.plot(prediction[:, 7], 'r')
        ax2.plot(manual_pred[:, 7], 'k')
        ax3.plot(gt[:, 8], 'b')
        ax3.plot(prediction[:, 8], 'r')
        ax3.plot(manual_pred[:, 8], 'k')
        ax4.plot(gt[:, 9], 'b')
        ax4.plot(prediction[:, 9], 'r')
        ax4.plot(manual_pred[:, 9], 'k')
        ax1.set_title('att_w')
        ax2.set_title('att_y')
        ax3.set_title('att_z')
        ax4.set_title('att_x')

        return fig1, fig2, fig3

    else:
        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        ax.plot(gt[:, 0:3], 'b')
        ax.plot(prediction[:, 0:3], 'r')
        ax.set_title('position')
        ax = fig.add_subplot(3, 1, 2)
        ax.plot(gt[:, 3:6], 'b')
        ax.plot(prediction[:, 3:6], 'r')
        ax.set_title('velocity')
        ax = fig.add_subplot(3, 1, 3)
        ax.plot(gt[:, 6:10], 'b')
        ax.plot(prediction[:, 6:10], 'r')
        ax.set_title('attitude (quat)')

        return fig


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
