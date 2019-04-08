import csv
import yaml
import numpy as np
import tensorflow as tf
import os
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from data.data_utils import generate_imu_img_dataset, save_processed_dataset_files, load_mat_data


class IMU:
    def __init__(self):
        self.timestamp = 0.0
        self.gyro = [0.0, 0.0, 0.0]
        self.acc = [0.0, 0.0, 0.0]

    def read(self, data):
        data = np.array(data)
        data = data.astype(np.float)
        self.timestamp = data[0]
        self.gyro = data[1:4]
        self.acc = data[4:7]


class GT:
    def __init__(self):
        self.timestamp = 0.0
        self.pos = [0.0, 0.0, 0.0]
        self.att = [0.0, 0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.ang_vel = [0.0, 0.0, 0.0]
        self.acc = [0.0, 0.0, 0.0]

    def read(self, data):
        data = np.array(data)
        data = data.astype(np.float)
        self.timestamp = data[0]
        self.pos = data[1:4]
        self.att = data[4:8]
        self.vel = data[8:11]
        self.ang_vel = data[11:14]
        self.acc = data[14:17]


SCALER_GYRO_FILE = "scaler_gyro.save"
SCALER_ACC_FILE = "scaler_acc.save"
SCALER_DIR_FILE = '/scaler_files_dir.txt'


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

    :param raw_imu_data: IMU data
    :param ground_truth_data: ground truth velocity data
    :return: the original imu data, and the interpolated velocity data
    """
    raw_imu_data = np.array(raw_imu_data)

    imu_timestamps = np.array([imu_meas.timestamp for imu_meas in raw_imu_data])
    gt_timestamps = np.array([gt_meas.timestamp for gt_meas in ground_truth_data])
    gt_velocity = np.array([gt_meas.vel for gt_meas in ground_truth_data])

    # Only keep imu data that is within the ground truth time span
    raw_imu_data = raw_imu_data[(imu_timestamps > gt_timestamps[0]) * (imu_timestamps < gt_timestamps[-1])]

    imu_timestamps = np.array([imu_meas.timestamp for imu_meas in raw_imu_data])

    # Interpolate Ground truth velocities to match IMU time acquisitions
    v_x_interp = interp1d(gt_timestamps, gt_velocity[:, 0])
    v_y_interp = interp1d(gt_timestamps, gt_velocity[:, 1])
    v_z_interp = interp1d(gt_timestamps, gt_velocity[:, 2])

    # Initialize array of interpolated Ground Truth velocities
    v_interp = [GT() for _ in range(len(imu_timestamps))]

    # Fill in array
    for i, imu_timestamp in enumerate(imu_timestamps):
        v_interp[i] = np.array([v_x_interp(imu_timestamp), v_y_interp(imu_timestamp), v_z_interp(imu_timestamp)])

    return [raw_imu_data, v_interp]


def pre_process_data(raw_imu_data, gt_v_interp, euroc_dir):
    """
    Pre-process euroc dataset (apply low-pass filter and minmax scaling)

    :param raw_imu_data: 1D array of IMU objects with IMU measurements
    :param gt_v_interp: list of 3D arrays with the decomposed velocity ground truth measurements
    :param euroc_dir: root directory of the euroc data
    :return: the filtered dataset
    """

    imu_vec = np.array([(imu_s.gyro, imu_s.acc) for imu_s in raw_imu_data])
    filt_imu_vec = np.copy(imu_vec)

    fs = 200.0  # Sample frequency (Hz)
    f0 = 10.0  # Frequency to be removed from signal (Hz)
    w0 = f0 / (fs / 2)  # Normalized Frequency

    # Design butterworth filter
    b_bw, a_bw = signal.butter(10, w0, output='ba')

    plt.figure()
    f, t, stft = signal.stft(imu_vec[:, 0, 1], 200)
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t, f, np.abs(stft))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    filt_imu_vec[:, 0, :] = signal.lfilter(b_bw, a_bw, imu_vec[:, 0, :], axis=0)
    filt_imu_vec[:, 1, :] = signal.lfilter(b_bw, a_bw, imu_vec[:, 1, :], axis=0)
    filt_gt_v_interp = signal.lfilter(b_bw, a_bw, gt_v_interp, axis=0)

    # Add more flat region so avoid model from learning average value
    # flat_region = filt_imu_vec[6000:7000, :, :]
    # flat_region_v = filt_gt_v_interp[6000:7000, :]
    # flat_region = np.repeat(np.concatenate((flat_region, flat_region[::-1, :, :])), [4], axis=0)
    # flat_region_v = np.repeat(np.concatenate((flat_region_v, flat_region_v[::-1])), [4], axis=0)
    # filt_imu_vec = np.concatenate((filt_imu_vec[0:6000, :, :], flat_region, filt_imu_vec[7000:, :, :]))
    # filt_gt_v_interp = np.concatenate((filt_gt_v_interp[0:6000, :], flat_region_v, filt_gt_v_interp[7000:, :]))

    f, t, stft = signal.stft(filt_imu_vec[:, 0, 1], 200)

    plt.subplot(2, 1, 2)
    plt.pcolormesh(t, f, np.abs(stft))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    y1 = np.squeeze(gt_v_interp)
    y2 = np.squeeze(filt_gt_v_interp)
    x1 = imu_vec.reshape((-1, 6))
    x2 = filt_imu_vec.reshape((-1, 6))

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(x1[:, 0:3])
    plt.title('Gyro')
    plt.subplot(3, 2, 3)
    plt.plot(x1[:, 3:6])
    plt.title('Acc')
    plt.subplot(3, 2, 5)
    plt.plot(np.linalg.norm(y1, axis=1))
    plt.title('Ground-truth speed')

    plt.subplot(3, 2, 2)
    plt.plot(x2[:, 0:3])
    plt.title('Filt Gyro')
    plt.subplot(3, 2, 4)
    plt.plot(x2[:, 3:6])
    plt.title('Filt Acc')
    plt.subplot(3, 2, 6)
    plt.plot(np.linalg.norm(y2, axis=1))
    plt.title('Filt Ground-truth speed')
    plt.show()

    scale_g = MinMaxScaler()
    scale_g.fit(filt_imu_vec[:, 0, :].reshape(-1, 1))
    scale_a = MinMaxScaler()
    scale_a.fit(filt_imu_vec[:, 1, :].reshape(-1, 1))

    joblib.dump(scale_g, euroc_dir + SCALER_GYRO_FILE)
    joblib.dump(scale_a, euroc_dir + SCALER_ACC_FILE)

    return filt_imu_vec, filt_gt_v_interp


def generate_speed_regression_ds(imu_len, raw_imu, gt_v, ds_dir, train_file_name, test_file_name):
    """

    :param imu_len: number of IMU acquisitions in the input (length)
    :param raw_imu: 3D array of IMU measurements (n_samples x 2 <gyro, acc> x 3 <x, y, z>)
    :param gt_v: list of 3D arrays with the decomposed velocity ground truth measurements
    :param ds_dir: root directory of the dataset
    :param train_file_name: Name of the preprocessed training dataset
    :param test_file_name: Name of the preprocessed testing dataset
    """

    imu_img_tensor, gt_v_tensor = generate_imu_img_dataset(raw_imu, gt_v, imu_len)
    euroc_training_ds = ds_dir + train_file_name
    euroc_testing_ds = ds_dir + test_file_name
    save_processed_dataset_files(euroc_training_ds, euroc_testing_ds, imu_img_tensor, gt_v_tensor)


def generate_tf_imu_train_ds(euroc_dir, euroc_train, batch_s, trained_model_dir):
    """
    Read the processed euroc dataset from the saved file. Generate the tf-compatible train/validation datasets

    :param euroc_dir: root directory of the euroc data
    :param euroc_train: Name of the preprocessed euroc training dataset
    :param batch_s: (mini)-batch size of datasets
    :param trained_model_dir: Name of the directory where trained model will be stored
    :return: the tf-compatible training and validation datasets, and their respective lengths
    """

    seed = 8901

    train_filename = euroc_dir + euroc_train

    imu_img_tensor, gt_v_tensor = load_mat_data(train_filename)

    file = open(trained_model_dir + SCALER_DIR_FILE, "r")
    scaler_dir = file.read()

    scale_g = joblib.load(scaler_dir + SCALER_GYRO_FILE)
    scale_a = joblib.load(scaler_dir + SCALER_ACC_FILE)

    for i in range(3):
        imu_img_tensor[:, :, i, 0] = scale_g.transform(imu_img_tensor[:, :, i, 0])
        imu_img_tensor[:, :, i+3, 0] = scale_a.transform(imu_img_tensor[:, :, i+3, 0])

    full_ds_len = len(gt_v_tensor)
    val_ds_len = np.ceil(full_ds_len * 0.1)
    train_ds_len = full_ds_len - val_ds_len

    val_ds_indexes = np.random.choice(range(full_ds_len), int(val_ds_len), replace=False)
    val_ds_imu_vec = imu_img_tensor[val_ds_indexes]
    val_ds_v_vec = gt_v_tensor[val_ds_indexes]

    imu_img_tensor = np.delete(imu_img_tensor, val_ds_indexes, axis=0)
    gt_v_tensor = np.delete(gt_v_tensor, val_ds_indexes)

    full_train_ds = tf.data.Dataset.from_tensor_slices((imu_img_tensor, gt_v_tensor)).shuffle(batch_s, seed=seed)
    val_ds = tf.data.Dataset.from_tensor_slices((val_ds_imu_vec, val_ds_v_vec)).batch(batch_s)
    train_ds = full_train_ds.batch(batch_s).repeat()

    return train_ds, val_ds, (train_ds_len, val_ds_len)


def generate_cnn_testing_dataset(euroc_dir, euroc_test, batch_s, trained_model_dir):
    """
    Read the preprocessed euroc dataset from saved file. Generate the tf-compatible testing dataset

    :param euroc_dir: root directory of the euroc data
    :param euroc_test:
    :param batch_s: (mini)-batch size of datasets
    :param trained_model_dir: Name of the directory where trained model is stored
    :return: the tf-compatible testing dataset, and its length
    """

    test_filename = euroc_dir + euroc_test

    imu_img_tensor, gt_v_tensor = load_mat_data(test_filename)

    file = open(trained_model_dir + SCALER_DIR_FILE, "r")
    scaler_dir = file.read()

    scale_g = joblib.load(scaler_dir + SCALER_GYRO_FILE)
    scale_a = joblib.load(scaler_dir + SCALER_ACC_FILE)

    for i in range(3):
        imu_img_tensor[:, :, i, 0] = scale_g.transform(imu_img_tensor[:, :, i, 0])
        imu_img_tensor[:, :, i+3, 0] = scale_a.transform(imu_img_tensor[:, :, i+3, 0])

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

        raw_imu_data, gt_v_interp = interpolate_ground_truth(raw_imu_data, ground_truth_data)

        processed_imu, processed_v = pre_process_data(raw_imu_data, gt_v_interp, euroc_dir)

        generate_speed_regression_ds(imu_seq_len, processed_imu, processed_v, euroc_dir, euroc_train, euroc_test)

    add_scaler_ref_to_training_dir(euroc_dir, trained_model_dir)

    return generate_tf_imu_train_ds(euroc_dir, euroc_train, batch_size, trained_model_dir)
