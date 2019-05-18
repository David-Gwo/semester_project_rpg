import numpy as np
import tensorflow as tf
from sklearn.externals import joblib

from data.imu_dataset_generators import StatePredictionDataset
from data.utils.data_utils import save_train_and_test_datasets, load_mat_data
from utils.directories import add_text_to_txt_file


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

        self.dataset_formatting = None

        if dataset_name == 'blackbird':
            from data.utils.blackbird_utils import BlackbirdDSManager
            self.dataset = BlackbirdDSManager()
        elif dataset_name == 'euroc':
            from data.utils.euroc_utils import EurocDSManager
            self.dataset = EurocDSManager()
        elif dataset_name == 'simulated':
            from data.utils.simulated_ds_utils import GenDSManager
            self.dataset = GenDSManager()
        else:
            raise NameError("Invalid dataset name")

        self.dataset_generator = StatePredictionDataset()

    def get_dataset(self, dataset_type, *args, train, batch_size, validation_split, split_percentage=0.1, plot=False,
                    shuffle=True, normalize=True, full_batches=False, repeat_ds=False, force_remake=False,
                    tensorflow_format=True):
        """
        Generates datasets for training or testing

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
        :param repeat_ds: whether to repeat indefinitely the main generated dataset
        :param force_remake: whether to reinforce the reconstruction of the dataset, or try to load it from directory
        :param tensorflow_format: whether to return the dataset in tensorflow dataset format or numpy array
        :return: the requested dataset/datasets
        """

        self.dataset_formatting = dataset_type

        if not self.is_dataset_ready(args) or force_remake:

            self.dataset.get_raw_ds()
            if plot:
                self.dataset.plot_all_data(title="raw")

            # TODO: export as json/yaml
            add_text_to_txt_file(str(args), self.dataset.get_ds_directory(), self.dataset_conf_file, overwrite=True)
            processed_imu, processed_gt = self.dataset.pre_process_data(self.scaler_gyro_file, self.scaler_acc_file)

            if plot:
                self.dataset.plot_all_data(title="filtered", from_numpy=True, show=True)

            # Generate the training and testing datasets
            self.save_dataset_to_files(processed_imu, processed_gt, args, split_percentage, shuffle=shuffle)

        if train:
            add_text_to_txt_file(self.dataset.get_ds_directory(), self.training_dir, self.scaler_dir_file)

        return self.generate_tf_ds(args,
                                   normalize=normalize,
                                   shuffle=shuffle,
                                   training=train,
                                   validation_split=validation_split,
                                   split_percentage=split_percentage,
                                   batch_size=batch_size,
                                   full_batches=full_batches,
                                   repeat_main_ds=repeat_ds,
                                   tensorflow_format=tensorflow_format)

    def save_dataset_to_files(self, x_data, y_data, args, test_split, shuffle):
        """
        Generates training and testing datasets, and saves a copy of them

        :param x_data: 3D array of IMU measurements (n_samples x 2 <gyro, acc> x 3 <x, y, z>)
        :param y_data: list of 3D arrays with the ground truth measurements
        :param args: extra arguments for dataset generation
        :param test_split: the percentage of dataset to be split for testing
        :param shuffle: whether datasets should be shuffled
        """

        ds_dir = self.dataset.get_ds_directory()

        self.dataset_generator.load_data(x_data, y_data)
        self.dataset_generator.generate_dataset(self.dataset_formatting, args)
        training_data, ground_truth_data = self.dataset_generator.get_dataset()

        storage_train_ds_file = "{0}{1}".format(ds_dir, self.train_data_file)
        storage_test_ds_file = "{0}{1}".format(ds_dir, self.test_data_file)
        save_train_and_test_datasets(storage_train_ds_file, storage_test_ds_file, training_data, ground_truth_data,
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

    def generate_tf_ds(self, args, normalize, shuffle, training, validation_split, split_percentage, batch_size,
                       full_batches, repeat_main_ds, tensorflow_format):
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
        :param repeat_main_ds: whether to repeat indefinitely the main generated dataset
        :param tensorflow_format: whether to return the dataset in tensorflow dataset format or numpy array
        :return: the requested dataset/datasets
        """

        seed = 8901

        if training:
            filename = self.dataset.get_ds_directory() + self.train_data_file
        else:
            filename = self.dataset.get_ds_directory() + self.test_data_file

        x_keys, y_keys = self.dataset_generator.get_dataset_keys(self.dataset_formatting)
        training_x, training_y = load_mat_data(filename, x_keys, y_keys)

        # TODO: find more elegant way to chose the tensor to normalize?
        if normalize:
            file = open(self.training_dir + self.scaler_dir_file, "r")
            scaler_dir = file.read()

            scale_g = joblib.load(scaler_dir + self.scaler_gyro_file)
            scale_a = joblib.load(scaler_dir + self.scaler_acc_file)

            imu_tensor = training_x["imu_input"]

            for i in range(args[0]):
                imu_tensor[:, i, 0:3, 0] = scale_g.transform(imu_tensor[:, i, 0:3, 0])
                imu_tensor[:, i, 3:6, 0] = scale_a.transform(imu_tensor[:, i, 3:6, 0])

            training_x["imu_input"] = imu_tensor

        # Compute main dataset and validation dataset lengths
        y_sample = list(training_y.values())[0]
        total_ds_len = len(y_sample)
        if validation_split:
            val_ds_len = int(np.ceil(total_ds_len * split_percentage))
        else:
            val_ds_len = 0
        main_ds_len = total_ds_len - val_ds_len

        # Get validation dataset indexes
        if shuffle:
            val_ds_indexes = np.random.choice(range(total_ds_len), int(val_ds_len), replace=False)
        else:
            val_ds_indexes = range(total_ds_len - val_ds_len, total_ds_len)

        # Split the training dataset into training and validation
        validation_x = {}
        validation_y = {}

        for x_key in x_keys:
            validation_x[x_key] = training_x[x_key][val_ds_indexes]
            training_x[x_key] = np.delete(training_x[x_key], val_ds_indexes, axis=0)
        for y_key in y_keys:
            validation_y[y_key] = training_y[y_key][val_ds_indexes]
            training_y[y_key] = np.delete(training_y[y_key], val_ds_indexes, axis=0)

        # If data is not to be transformed to tensorflow dataset, return
        if not tensorflow_format:
            # TODO: shuffle if requested

            if validation_split:
                return (training_x, training_y), (validation_x, validation_y), (main_ds_len, val_ds_len)
            else:
                return (training_x, training_y), main_ds_len

        # Otherwise generate the tensorflow datasets
        main_ds = tf.data.Dataset.from_tensor_slices((training_x, training_y))
        val_ds = tf.data.Dataset.from_tensor_slices((validation_x, validation_y))

        # Shuffle dataset if requested
        if shuffle:
            main_ds = main_ds.shuffle(batch_size, seed=seed)

        # Batch dataset
        main_ds = main_ds.batch(batch_size, drop_remainder=full_batches)
        val_ds = val_ds.batch(batch_size, drop_remainder=full_batches)

        # Repeat dataset if requested
        if repeat_main_ds:
            main_ds = main_ds.repeat()

        if validation_split:
            return main_ds, val_ds, (main_ds_len, val_ds_len)
        else:
            return main_ds, main_ds_len
