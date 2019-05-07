import joblib
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter as butterworth_filter

from data.utils.data_utils import filter_with_coeffs, interpolate_ts


class IMU:
    def __init__(self):
        self.timestamp = 0.0
        self.gyro = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])

    @abstractmethod
    def read(self, data):
        ...

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

    @abstractmethod
    def read(self, data):
        ...

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


class InertialDataset(ABC):
    @abstractmethod
    def __init__(self):
        self.imu_data = None
        self.gt_data = None
        self.sampling_freq = None
        self.ds_local_dir = None
        ...

    @abstractmethod
    def get_raw_ds(self):
        ...
    
    def get_ds_directory(self):
        assert self.ds_local_dir is not None, "Directory has not yet been set"
        return self.ds_local_dir

    def pre_process_data(self, gyro_scale_file, acc_scale_file, filter_freq):
        """
        Pre-process euroc dataset (apply low-pass filter and minmax scaling)

        :param gyro_scale_file: file to save pre-processing functions for gyroscope
        :param acc_scale_file: file to save pre-processing functions for accelerometer
        :param filter_freq: frequency used for low-pass filter
        :return: the filtered datasets in compressed (numpy) format
        """
        
        assert self.imu_data is not None and self.gt_data is not None and self.sampling_freq is not None, \
            "Data cannot be processed because there is no data yet."

        # Transform the data to numpy matrices
        imu_unroll = np.array([(imu_s.unroll()) for imu_s in self.imu_data])
        gt_unroll = np.array([(gt_meas.unroll()) for gt_meas in self.gt_data])

        # Get number of channels per data type (we subtract 1 because timestamp is not a channel we want to filter)
        imu_channels = np.shape(imu_unroll)[1] - 1
        gt_channels = np.shape(gt_unroll)[1] - 1

        # Design butterworth filter
        fs = self.sampling_freq  # Sample frequency (Hz)
        f0 = filter_freq  # Frequency to be removed from signal (Hz)
        w0 = f0 / (fs / 2)  # Normalized Frequency
        [b_bw, a_bw] = butterworth_filter(10, w0, output='ba')

        imu_data = np.stack([filter_with_coeffs(a_bw, b_bw, imu_unroll[:, i], fs) for i in range(imu_channels)],
                            axis=1)
        gt_data = np.stack([filter_with_coeffs(a_bw, b_bw, gt_unroll[:, i], fs) for i in range(gt_channels)],
                           axis=1)

        scale_g = MinMaxScaler()
        scale_g.fit(np.stack(imu_data[:, 0]))
        scale_a = MinMaxScaler()
        scale_a.fit(np.stack(imu_data[:, 1]))

        joblib.dump(scale_g, self.get_ds_directory() + gyro_scale_file)
        joblib.dump(scale_a, self.get_ds_directory() + acc_scale_file)

        # Add back the timestamps to the data matrix and return
        # Careful -> data from now on is in numpy format, instead of GT and IMU format
        self.imu_data = np.append(imu_data, np.expand_dims(imu_unroll[:, -1], axis=1), axis=1)
        self.gt_data = np.append(gt_data, np.expand_dims(gt_unroll[:, -1], axis=1), axis=1)

        return self.imu_data, self.gt_data
    
    def interpolate_ground_truth(self):
        """
        Interpolates the data of the ground truth so that it matches the timestamps of the raw imu data

        :return: the original imu data, and the interpolated ground truth data
        """
        x_data = np.array(self.imu_data)

        imu_timestamps = np.array([imu_meas.timestamp for imu_meas in x_data])
        gt_unroll = np.array([(gt_meas.unroll()) for gt_meas in self.gt_data])

        gt_timestamps = gt_unroll[:, 5]

        # Only keep imu data that is within the ground truth time span
        x_data = x_data[(imu_timestamps > gt_timestamps[0]) * (imu_timestamps < gt_timestamps[-1])]
        imu_timestamps = np.array([imu_meas.timestamp for imu_meas in x_data])
        self.imu_data = x_data

        gt_pos = np.stack(gt_unroll[:, 0])
        gt_vel = np.stack(gt_unroll[:, 1])
        gt_att = np.stack(gt_unroll[:, 2])
        gt_ang_vel = np.stack(gt_unroll[:, 3])
        gt_acc = np.stack(gt_unroll[:, 4])

        # Interpolate Ground truth to match IMU time acquisitions
        gt_pos_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_pos)
        gt_vel_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_vel)
        gt_att_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_att, is_quaternion=True)
        gt_ang_vel_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_ang_vel)
        gt_acc_interp = interpolate_ts(gt_timestamps, imu_timestamps, gt_acc)
        
        gt_interp = (gt_pos_interp, gt_vel_interp, gt_att_interp, gt_ang_vel_interp, gt_acc_interp, imu_timestamps)
        
        # Re-make vector of interpolated GT measurements
        n_samples = len(gt_pos_interp)
        gt_interp = [GT().read_from_tuple(tuple([gt_interp[i][j] for i in range(6)])) for j in range(n_samples)]

        self.gt_data = gt_interp

    def plot_all_data(self, title="", from_numpy=False, show=False):
        """
        Plots the imu and ground truth data in two separate figures

        :param title: title of the plot
        :param from_numpy: format of the input data
        :param show: whether to show plot or not
        :return:
        """

        if from_numpy:
            fig = plt.figure()
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(np.stack(self.imu_data[:, 0]))
            ax.set_title("IMU: gyroscope")
            ax = fig.add_subplot(2, 1, 2)
            ax.plot(np.stack(self.imu_data[:, 1]))
            ax.set_title("IMU: accelerometer")
            fig.suptitle(title)

            fig = plt.figure()
            ax = fig.add_subplot(2, 2, 1)
            ax.plot(np.stack(self.gt_data[:, 0]))
            ax.set_title("GT: position")
            ax = fig.add_subplot(2, 2, 2)
            ax.plot(np.stack(self.gt_data[:, 1]))
            ax.set_title("GT: velocity")
            ax = fig.add_subplot(2, 2, 3)
            ax.plot(np.stack(self.gt_data[:, 2]))
            ax.set_title("GT: attitude")
            ax = fig.add_subplot(2, 2, 4)
            ax.plot(np.stack(self.gt_data[:, 3]))
            ax.set_title("GT: angular velocity")
            fig.suptitle(title)

        else:
            fig = plt.figure()
            ax = fig.add_subplot(2, 1, 1)
            ax.plot([imu.gyro for imu in self.imu_data])
            ax.set_title("IMU: gyroscope")
            ax = fig.add_subplot(2, 1, 2)
            ax.plot([imu.acc for imu in self.imu_data])
            ax.set_title("IMU: accelerometer")
            fig.suptitle(title)

            fig = plt.figure()
            ax = fig.add_subplot(2, 2, 1)
            ax.plot([gt.pos for gt in self.gt_data])
            ax.set_title("GT: position")
            ax = fig.add_subplot(2, 2, 2)
            ax.plot([gt.vel for gt in self.gt_data])
            ax.set_title("GT: velocity")
            ax = fig.add_subplot(2, 2, 3)
            ax.plot([gt.att for gt in self.gt_data])
            ax.set_title("GT: attitude")
            ax = fig.add_subplot(2, 2, 4)
            ax.plot([gt.ang_vel for gt in self.gt_data])
            ax.set_title("GT: angular velocity")
            fig.suptitle(title)

        if show:
            plt.show()
