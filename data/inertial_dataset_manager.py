import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import butter as butterworth_filter
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

from data.imu_dataset_generators import windowed_imu_integration_dataset, imu_img_dataset
from data.utils.data_utils import save_train_and_test_datasets, interpolate_ts, filter_with_coeffs, load_mat_data
from data.utils.blackbird_utils import BlackbirdDSManager
from utils import add_text_to_txt_file


SCALER_GYRO_FILE = "scaler_gyro.save"
SCALER_ACC_FILE = "scaler_acc.save"
SCALER_DIR_FILE = 'scaler_files_dir.txt'
DATASET_CONF_PARAMS_FILE = "generated_ds_params.txt"


class DatasetManager:
    def __init__(self, prepared_train_data_file, prepared_test_data_file, trained_model_dir, dataset_name):
        """

        :param prepared_train_data_file: Name of the preprocessed training dataset
        :param prepared_test_data_file: Name of the preprocessed testing dataset
        :param trained_model_dir: Local directory of the model currently being trained
        :param dataset_name: Name of the dataset to use
        """

        self.train_data_file = prepared_train_data_file
        self.test_data_file = prepared_test_data_file
        self.training_dir = trained_model_dir

        self.scaler_gyro_file = SCALER_GYRO_FILE
        self.scaler_acc_file = SCALER_ACC_FILE
        self.scaler_dir_file = SCALER_DIR_FILE
        self.dataset_conf_file = DATASET_CONF_PARAMS_FILE

        self.dataset_type = None

        if dataset_name == 'blackbird':
            self.dataset = BlackbirdDSManager(self)
        elif dataset_name == 'euroc':
            # TODO: implement!
            raise NotImplemented()
        else:
            raise NameError("Invalid dataset name")

    def get_dataset(self, dataset_type, *args, train, batch_size, validation_split, split_percentage=0.1, plot=False,
                    shuffle=True, normalize=True, full_batches=False, force_remake=False):
        """

        :param dataset_type: Type of dataset to be generated
        :param args: extra arguments for dataset generation
        :param train: whether dataset is for training or testing
        :param batch_size: batch size of training, validation and testing dataset (same batch size for the three)
        :param validation_split: whether a validation split should be generated
        :param split_percentage: the percentage of dataset to be split for validation/testing
        :param plot: whether to plot the dataset
        :param shuffle: whether to shuffle the dataset
        :param normalize: whether to normalize the dataset
        :param full_batches: whether to enforce same-sized batches in the dataset
        :param force_remake: whether to reinforce the reconstruction of the dataset, or try to load it from directory
        :return: the requested tensorflow dataset/datasets
        """

        self.dataset_type = dataset_type

        if not self.is_dataset_ready(args) or force_remake:

            raw_imu, raw_gt = self.dataset.get_raw_ds()
            add_text_to_txt_file(str(args), self.dataset.get_ds_directory(), self.dataset_conf_file, overwrite=True)
            processed_imu, processed_gt = self.pre_process_data(raw_imu, raw_gt, self.dataset.get_ds_directory())

            # Show plots of the training dataset
            if plot:
                self.plot_all_data(raw_imu, raw_gt, title="raw")
                self.plot_all_data(processed_imu, processed_gt, title="filtered", from_numpy=True, show=True)

            # Generate the training and testing datasets
            self.generate_dataset(processed_imu, processed_gt, args, split_percentage, shuffle=shuffle)

        if train:
            add_text_to_txt_file(self.dataset.get_ds_directory(), self.training_dir, self.scaler_dir_file)

        return self.generate_tf_ds(args,
                                   normalize=normalize,
                                   shuffle=shuffle,
                                   training=train,
                                   validation_split=validation_split,
                                   split_percentage=split_percentage,
                                   batch_size=batch_size,
                                   full_batches=full_batches)

    def generate_dataset(self, x_data, y_data, args, test_split, shuffle):
        """
        Generates training and testing datasets, and saves a copy of them

        :param x_data: 3D array of IMU measurements (n_samples x 2 <gyro, acc> x 3 <x, y, z>)
        :param y_data: list of 3D arrays with the ground truth measurements
        :param args: extra arguments for dataset generation
        :param test_split: the percentage of dataset to be split for testing
        :param shuffle: whether datasets should be shuffled
        """

        ds_dir = self.dataset.get_ds_directory()

        if self.dataset_type == "windowed_imu_integration":
            train_data_tensor, gt_tensor = windowed_imu_integration_dataset(x_data, y_data, args)
        else:
            train_data_tensor, gt_tensor = imu_img_dataset(x_data, y_data, args)

        storage_train_ds_file = "{0}{1}".format(ds_dir, self.train_data_file)
        storage_test_ds_file = "{0}{1}".format(ds_dir, self.test_data_file)
        save_train_and_test_datasets(storage_train_ds_file, storage_test_ds_file, train_data_tensor, gt_tensor,
                                     test_split, shuffle)

    def is_dataset_ready(self, args):
        """
        Checks if the generated dataset files are compatible with the requested dataset

        :param args: extra arguments for dataset generation
        :return: whether the dataset is available and compatible
        """

        try:
            file = open(self.dataset.get_ds_directory() + self.dataset_conf_file, "r")
            generated_ds_params = file.read()
            return generated_ds_params == str(args)
        except (NotADirectoryError, FileNotFoundError):
            return False

    def pre_process_data(self, raw_imu_data, gt_interp, euroc_dir):
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

        # TODO: get sampling frequency from somewhere

        # Design butterworth filter
        fs = 100.0  # Sample frequency (Hz)
        f0 = 10.0  # Frequency to be removed from signal (Hz)
        w0 = f0 / (fs / 2)  # Normalized Frequency
        [b_bw, a_bw] = butterworth_filter(10, w0, output='ba')

        filtered_imu_vec = np.stack([filter_with_coeffs(a_bw, b_bw, imu_unroll[:, i], fs) for i in range(imu_channels)],
                                    axis=1)
        filtered_gt_vec = np.stack([filter_with_coeffs(a_bw, b_bw, gt_unroll[:, i], fs) for i in range(gt_channels)],
                                   axis=1)

        scale_g = MinMaxScaler()
        scale_g.fit(np.stack(filtered_imu_vec[:, 0]))
        scale_a = MinMaxScaler()
        scale_a.fit(np.stack(filtered_imu_vec[:, 1]))

        joblib.dump(scale_g, euroc_dir + self.scaler_gyro_file)
        joblib.dump(scale_a, euroc_dir + self.scaler_acc_file)

        # Add back the timestamps to the data matrix and return
        filtered_imu_vec = np.append(filtered_imu_vec, np.expand_dims(imu_unroll[:, -1], axis=1), axis=1)
        filtered_gt_vec = np.append(filtered_gt_vec, np.expand_dims(gt_unroll[:, -1], axis=1), axis=1)

        return filtered_imu_vec, filtered_gt_vec

    def generate_tf_ds(self, args, normalize, shuffle, training, validation_split, split_percentage, batch_size,
                       full_batches):
        """
        Recovers the dataset from the files, and generates tensorflow-compatible datasets

        :param args: extra arguments for dataset generation
        :param training: whether the dataset is for training or testing
        :param validation_split: whether a validation split should be generated
        :param split_percentage: the percentage of dataset to be split for validation/testing
        :param shuffle: whether to shuffle the dataset
        :param normalize: whether to normalize the dataset
        :param batch_size: batch size of training, validation and testing dataset (same batch size for the three)
        :param full_batches: whether to enforce same-sized batches in the dataset
        :return:
        """

        seed = 8901

        if training:
            filename = self.dataset.get_ds_directory() + self.train_data_file
        else:
            filename = self.dataset.get_ds_directory() + self.test_data_file

        imu_tensor, gt_tensor = load_mat_data(filename)

        if normalize:
            file = open(self.training_dir + self.scaler_dir_file, "r")
            scaler_dir = file.read()

            scale_g = joblib.load(scaler_dir + self.scaler_gyro_file)
            scale_a = joblib.load(scaler_dir + self.scaler_acc_file)

            for i in range(args[0]):
                imu_tensor[:, i, 0:3, 0] = scale_g.transform(imu_tensor[:, i, 0:3, 0])
                imu_tensor[:, i, 3:6, 0] = scale_a.transform(imu_tensor[:, i, 3:6, 0])

        total_ds_len = len(gt_tensor)

        if validation_split:
            val_ds_len = int(np.ceil(total_ds_len * split_percentage))
        else:
            val_ds_len = 0

        main_ds_len = total_ds_len - val_ds_len

        if shuffle:
            val_ds_indexes = np.random.choice(range(total_ds_len), int(val_ds_len), replace=False)
        else:
            val_ds_indexes = range(total_ds_len - val_ds_len, total_ds_len)

        val_ds_imu_vec = imu_tensor[val_ds_indexes]
        val_ds_v_vec = gt_tensor[val_ds_indexes]

        imu_tensor = np.delete(imu_tensor, val_ds_indexes, axis=0)
        gt_tensor = np.delete(gt_tensor, val_ds_indexes, axis=0)

        main_ds = tf.data.Dataset.from_tensor_slices((imu_tensor, gt_tensor))
        val_ds = tf.data.Dataset.from_tensor_slices((val_ds_imu_vec, val_ds_v_vec))

        if shuffle:
            main_ds.shuffle(batch_size, seed=seed)

        main_ds = main_ds.batch(batch_size, drop_remainder=full_batches).repeat()
        val_ds.batch(batch_size, drop_remainder=full_batches)

        if validation_split:
            return main_ds, val_ds, (main_ds_len, val_ds_len)
        else:
            return main_ds, main_ds_len

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
