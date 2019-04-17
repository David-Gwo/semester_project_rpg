import csv
import yaml
import numpy as np

from data.inertial_ABCs import IMU, GT

# TODO: ADAPT TO NEW inertial_dataset_manager.py


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


# TODO: fix fix fix fix
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

        processed_imu, processed_v = pre_process_data(raw_imu_data, gt_v_interp, euroc_dir)

        generate_dataset(processed_imu, processed_v, euroc_dir, euroc_train, euroc_test, imu_seq_len)

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


class EurocIMU(IMU):
    def __init__(self):
        super(EurocIMU, self).__init__()
        self.gyro_indx = [14, 15, 16]
        self.acc_indx = [19, 20, 21]

    def read(self, data):
        data = np.array(data)
        data = data.astype(np.float)
        self.timestamp = data[0]
        self.gyro = data[1:4]
        self.acc = data[4:7]


class EurocGT(GT):
    def __init__(self):
        super(EurocGT, self).__init__()

    def read(self, data):
        data = np.array(data)
        data = data.astype(np.float)
        self.timestamp = data[0]
        self.pos = data[1:4]
        self.att = data[4:8]
        self.vel = data[8:11]
        self.ang_vel = data[11:14]
        self.acc = data[14:17]

    def integrate(self, gt_old):
        """
        Integrates position and attitude to obtain velocity and angular velocity. Saves integrated values to current
        BBGT object

        :param gt_old: BBGT from previous timestamp
        """
        # TODO: review angular velocity integration -> https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/

        dt = (self.timestamp - gt_old.timestamp) * 10e-6
        self.vel = (self.pos - gt_old.pos) / dt
        att_q = q.quaternion(self.att[0], self.att[1], self.att[2], self.att[3])
        att = q.as_euler_angles(att_q)
        old_att_q = q.quaternion(gt_old.att[0], gt_old.att[1], gt_old.att[2], gt_old.att[3])
        old_att = q.as_euler_angles(old_att_q)
        self.ang_vel = (att - old_att) / dt

