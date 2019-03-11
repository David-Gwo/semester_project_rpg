import csv
import yaml
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d


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


def read_euroc_dataset(euroc_dir):
    imu_file = euroc_dir + 'imu0/data.csv'
    imu_yaml_file = euroc_dir + 'imu0/sensor.yaml'
    ground_truth_file = euroc_dir + 'state_groundtruth_estimate0/data.csv'

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
    raw_imu_data = np.array(raw_imu_data)

    imu_timestamps = np.array([imu_meas.timestamp for imu_meas in raw_imu_data])
    gt_timestsamps = np.array([gt_meas.timestamp for gt_meas in ground_truth_data])
    gt_velocity = np.array([gt_meas.vel for gt_meas in ground_truth_data])

    # Only keep imu data that is within the ground truth time span
    raw_imu_data = raw_imu_data[(imu_timestamps > gt_timestsamps[0]) * (imu_timestamps < gt_timestsamps[-1])]

    imu_timestamps = np.array([imu_meas.timestamp for imu_meas in raw_imu_data])

    # Interpolate Ground truth velocities to match IMU time acquisitions
    v_x_interp = interp1d(gt_timestsamps, gt_velocity[:, 0])
    v_y_interp = interp1d(gt_timestsamps, gt_velocity[:, 1])
    v_z_interp = interp1d(gt_timestsamps, gt_velocity[:, 2])

    # Initialize array of interpolated Ground Truth velocities
    v_interp = [GT() for _ in range(len(imu_timestamps))]

    # Fill in array
    for i, imu_timestamp in enumerate(imu_timestamps):
        v_interp[i] = np.array([v_x_interp(imu_timestamp), v_y_interp(imu_timestamp), v_z_interp(imu_timestamp)])

    return [raw_imu_data, v_interp]


def generate_cnn_train_data(imu_len, raw_imu, gt_v, batch_s):
    """

    :param imu_len: number of IMU acquisitions in the input (length)
    :param raw_imu: 1D array of IMU objects with all the IMU measurements
    :param gt_v: list of 3D arrays with the decomposed velocity ground truth measurements
    :param batch_s: size of mini-batch
    :return:
    """

    # Initialize data tensors #
    # Initialize x data. Will be sequence of IMU measurements of size (imu_len x 6)
    imu_img_tensor = np.zeros((len(raw_imu), imu_len, 6))
    # Initialize y data. Will be the absolute ground truth value of the speed of the drone
    gt_v_tensor = np.zeros(len(raw_imu))

    # for i, _ in enumerate(raw_imu[0:len(raw_imu)-imu_x_len]):
    for i in range(len(raw_imu) - imu_len):
        imu_img = np.zeros((imu_len, 6))

        # The first imu_x_len data vectors will not be full of data (not enough acquisitions to fill it up yet)
        if i < imu_len:
            imu_img[imu_len - i - 1:imu_len, :] = \
                np.array([list(imu_s.gyro) + list(imu_s.acc) for imu_s in raw_imu[0:i+1]])
        else:
            imu_img = np.array([list(imu_s.gyro) + list(imu_s.acc) for imu_s in raw_imu[i:i + imu_len]])

        # TODO: Should the elapsed time be included in the data?

        imu_img_tensor[i, :, :] = imu_img
        gt_v_tensor[i] = np.linalg.norm(gt_v[i])

    full_ds_len = len(gt_v_tensor)
    val_ds_len = np.ceil(full_ds_len * 0.1)
    train_ds_len = full_ds_len - val_ds_len

    full_train_ds = tf.data.Dataset.from_tensor_slices((imu_img_tensor, gt_v_tensor)).shuffle(batch_s)
    val_ds = full_train_ds.take(val_ds_len)
    train_ds = full_train_ds.skip(val_ds_len)

    val_ds.batch(batch_s).repeat()
    train_ds.batch(batch_s).repeat()

    return [train_ds, val_ds, (train_ds_len, val_ds_len)]


def load_euroc_dataset():
    euroc_dir = './data/EuRoC_dataset/mav0/'
    batch_size = 10
    imu_seq_len = 200  # Number of IMU measurements in the x vectors. Is a function of the sampling frequency of the IMU

    [imu_yaml_data, raw_imu_data, ground_truth_data] = read_euroc_dataset(euroc_dir)

    [raw_imu_data, gt_v_interp] = interpolate_ground_truth(raw_imu_data, ground_truth_data)

    return generate_cnn_train_data(imu_seq_len, raw_imu_data, gt_v_interp, batch_size)
