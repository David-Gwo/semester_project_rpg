import numpy as np
import matplotlib.pyplot as plt

from data.imu_dataset_generators import imu_gt_vel_dataset, windowed_imu_integration_dataset, imu_img_dataset
from data.utils.data_utils import save_train_and_test_datasets, interpolate_ts
from data.blackbird_manager import BlackbirdDSManager


class DatasetManager:
    def __init__(self, batch_size, prepared_train_data_file, prepared_test_data_file, prepared_file_available,
                 trained_model_dir, dataset_name):

        self.batch_size = batch_size
        self.train_data_file = prepared_train_data_file
        self.test_data_file = prepared_test_data_file
        self.files_ready = prepared_file_available
        self.training_dir = trained_model_dir

        if dataset_name == 'blackbird':
            self.dataset = BlackbirdDSManager(self)
        elif dataset_name == 'euroc':
            return load_euroc_dataset(self.config.train_dir, self.config.batch_size, self.config.window_length,
                                      self.config.prepared_train_data_file, self.config.prepared_test_data_file,
                                      self.config.prepared_file_available, self.trained_model_dir)
        else:
            raise NameError("Invalid dataset name")

    def get_dataset(self, *args):

        if not self.dataset.is_file_ready(args):
            self.dataset.make_dataset(args)

    def generate_dataset(self, x_data, y_data, ds_dir, train_file_name, test_file_name, dataset_type, *args, shuffle=True):
        """
        Generates training and testing datasets, and saves a copy of them

        :param x_data: 3D array of IMU measurements (n_samples x 2 <gyro, acc> x 3 <x, y, z>)
        :param y_data: list of 3D arrays with the ground truth measurements
        :param ds_dir: root directory of the dataset
        :param train_file_name: Name of the preprocessed training dataset
        :param test_file_name: Name of the preprocessed testing dataset
        :param dataset_type: Type of dataset to be generated
        :param args: extra arguments for dataset generation
        :param shuffle: whether datasets should be shuffled
        """

        if dataset_type == "imu_img_gt_vel":
            train_data_tensor, gt_tensor = imu_gt_vel_dataset(x_data, y_data, *args)
        elif dataset_type == "windowed_imu_integration":
            train_data_tensor, gt_tensor = windowed_imu_integration_dataset(x_data, y_data, *args)
        else:
            train_data_tensor, gt_tensor = imu_img_dataset(x_data, y_data, *args)

        storage_train_ds_file = "{0}{1}".format(ds_dir, train_file_name)
        storage_test_ds_file = "{0}{1}".format(ds_dir, test_file_name)
        save_train_and_test_datasets(storage_train_ds_file, storage_test_ds_file, train_data_tensor, gt_tensor, shuffle)

    @staticmethod
    def interpolate_ground_truth(x_data, gt_data):
        """
        Interpolates the data of the ground truth so that it matches the timestamps of the raw imu data

        :param x_data: feature data (an array of IMU objects)
        :param gt_data: ground truth velocity data (an array of GT objects)
        :return: the original imu data, and the interpolated ground truth data
        """
        x_data = np.array(x_data)

        imu_timestamps = np.array([imu_meas.timestamp for imu_meas in x_data])
        gt_unroll = np.array([(gt_meas.unroll()) for gt_meas in gt_data])

        gt_pos = np.stack(gt_unroll[:, 0])
        gt_vel = np.stack(gt_unroll[:, 1])
        gt_att = np.stack(gt_unroll[:, 2])
        gt_ang_vel = np.stack(gt_unroll[:, 3])
        gt_acc = np.stack(gt_unroll[:, 4])
        gt_timestamps = gt_unroll[:, 5]

        # Only keep imu data that is within the ground truth time span
        x_data = x_data[(imu_timestamps > gt_timestamps[0]) * (imu_timestamps < gt_timestamps[-1])]
        imu_timestamps = np.array([imu_meas.timestamp for imu_meas in x_data])

        # Interpolate Ground truth to match IMU time acquisitions
        gt_pos_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_pos)
        gt_vel_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_vel)
        gt_att_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_att, is_quaternion=True)
        gt_ang_vel_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_ang_vel)
        gt_acc_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_acc)

        return [x_data,
                (gt_pos_interp, gt_vel_interp, gt_att_interp, gt_ang_vel_interp, gt_acc_interp, imu_timestamps)]

    @staticmethod
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
