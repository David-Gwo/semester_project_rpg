import numpy as np
import collections
from utils.algebra import log_mapping, exp_mapping, quaternion_error, rotate_vec
from tensorflow.python.keras.utils import Progbar
from pyquaternion import Quaternion


class StatePredictionDataset:
    def __init__(self):

        self.imu_raw = None
        self.gt_raw = None
        self.x_ds = {}
        self.y_ds = {}
        self.accepted_datasets = ["windowed_imu_integration",
                                  "windowed_imu_speed_regression",
                                  "windowed_imu_integration_with_so3_rotation",
                                  "windowed_imu_preintegration"]

        self.dataset_keys = {
            "windowed_imu_integration": {
                "x_keys": ["state_input", "imu_input"],
                "y_keys": ["state_output"]
            },
            "windowed_imu_speed_regression": {
                "x_keys": ["state_input"],
                "y_keys": ["state_output"]
            },
            "windowed_imu_integration_with_so3_rotation": {
                "x_keys": ["state_input", "imu_input"],
                "y_keys": ["state_output"]
            },
            "windowed_imu_preintegration": {
                "x_keys": ["state_input", "imu_input"],
                "y_keys": ["pre_integrated_R", "pre_integrated_v", "pre_integrated_p", "state_output"]
            }
        }

        assert collections.Counter(self.accepted_datasets) == collections.Counter(list(self.dataset_keys.keys())), \
            "There is one or more key mismatch in the accepted dataset dictionaries"

    def load_data(self, imu, gt):
        """
        Loads the imu and ground truth data. They will be used to generate a dataset

        :param imu: vector of ordered IMU readings. Shape: <n, 7>, n = number of acquisitions, the first three columns
        correspond to the three gyro readings (x,y,z), the next three to the accelerometer readings, and the last one is
        the time difference between the previous and current acquisition. By convention raw_imu[0, 7] = 0
        :param gt: ground truth velocity data. Shape: <n, 17>, n = number of acquisitions, and each acquisition is a
        17-dimensional vector with the components: x,y,z position, x,y,z velocity, w,x,y,z attitude, x,y,z angular
        velocity, x,y,z acceleration and timestamp difference (same as `raw_imu`)
        """

        self.imu_raw = imu
        self.gt_raw = gt

    def generate_dataset(self, dataset, args):
        """
        Generates the chosen dataset

        :param dataset: version of dataset to generate (must be one of the accepted keys)
        :param args: extra arguments for dataset generation
        """

        assert dataset in self.accepted_datasets, "The dataset version must be among {0}".format(self.accepted_datasets)

        if dataset == "windowed_imu_integration":
            self.windowed_imu_for_state_prediction(args)
        elif dataset == "windowed_imu_integration_with_so3_rotation":
            self.windowed_with_so3_rotation(args)
        elif dataset == "windowed_imu_speed_regression":
            self.imu_speed_regression(args)
        elif dataset == "windowed_imu_preintegration":
            self.windowed_imu_preintegration_dataset(args)

    def set_outputs(self, keys, outputs):
        """
        Adds the outputs to the dictionary of outputs using the provided keys

        :param keys: names of the outputs
        :param outputs: outputs to be predicted
        """

        assert isinstance(keys, list) == isinstance(outputs, list), "The parameters must be lists of same length"
        assert len(keys) == len(outputs), "There must be as many keys as outputs"
        assert len(np.unique(keys)) == len(keys), "There must not be two outputs with the same key"

        for i in range(len(keys)):
            self.y_ds[keys[i]] = outputs[i]

    def set_inputs(self, keys, inputs):
        """
        Adds the inputs to the dictionary of inputs using the provided keys

        :param keys: names of the inputs
        :param inputs: inputs to be used for prediction
        """

        assert isinstance(keys, list) == isinstance(inputs, list), "The parameters must be lists of same length"
        assert len(keys) == len(inputs), "There must be as many keys as inputs"
        assert len(np.unique(keys)) == len(keys), "There must not be two inputs with the same key"

        for i in range(len(keys)):
            self.x_ds[keys[i]] = inputs[i]

    def get_dataset(self):
        """
        Gets the inputs and the outputs of the generated dataset

        :return: the inputs and outputs dictionaries of the dataset
        """

        assert self.x_ds, "The dataset has not been generated yet!"
        assert self.y_ds, "The dataset has not been generated yet!"

        return self.x_ds, self.y_ds

    def get_dataset_keys(self, dataset_type):
        """
        Gets the input and output keys of the generated dataset

        :return: the input and output keys of the generated dataset
        """
        return self.dataset_keys[dataset_type]["x_keys"], self.dataset_keys[dataset_type]["y_keys"]

    def imu_speed_regression(self, args):
        """
        :param args: extra arguments for dataset generation

        Generates a dataset of imu images to regress linear speed.
            Inputs 1: a window of imu samples of dimensions <imu_len x 7>, where 7 are the 6 dimensions of the IMU
            readings (3 gyro + 3 acc) plus the time differences between imu acquisitions, and the number of rows are the
            number of used imu samples.
            Output 1: the regressed scalar value of the speed at the end of the window of IMU samples
        """

        window_len = args[0]

        # Initialize y data. Will be the absolute ground truth value of the speed of the drone
        gt_v_tensor = np.linalg.norm(self.gt_raw[:, 3:6], axis=1)

        imu_img_tensor = self.window_imu_data(window_len)

        self.set_inputs(["state_input"], [imu_img_tensor])
        self.set_outputs(["state_output"], [gt_v_tensor])

    def windowed_imu_for_state_prediction(self, args):
        """
        :param args: extra arguments for dataset generation

        Generates a dataset that aims at performing IMU integration.
            Input 1: one initial 10-dimensional state consisting on initial position (x,y,z), velocity (x,y,z) and
            orientation (w,x,y,z)
            Input 2: a window of imu samples of dimensions <imu_len x 7>, where 7 are the 6 dimensions of the IMU
            readings (3 gyro + 3 acc) plus the time differences between imu acquisitions, and the number of rows are the
            number of used imu samples.
            Output 1: the final 10-dimensional state consisting on final position (x,y,z), velocity (x,y,z) and
            orientation (w,x,y,z)
        """

        window_len = args[0]

        n_samples = len(self.imu_raw) - window_len

        gt = reformat_data(self.gt_raw)

        # Keep only position, attitude, velocity information (remove angular velocity, acceleration and timestamp)
        gt = np.delete(gt, np.s_[10:17], axis=1)

        # Copy the first row window_len times at the beginning of the dataset
        gt_augmented = np.append(np.ones((window_len, 1))*np.expand_dims(gt[0, :], axis=0), gt[1:, :], axis=0)

        # Add the initial state of the window at the beginning of each training sequence
        initial_state_vec = gt_augmented[:n_samples, :]

        imu_window = self.window_imu_data(window_len)[:n_samples, :, :, :]

        self.set_inputs(["state_input", "imu_input"], [initial_state_vec, imu_window])
        self.set_outputs(["state_output"], [gt[1:-window_len + 1, :]])

    def windowed_with_so3_rotation(self, args):
        """
        :param args: extra arguments for dataset generation

        Generates a dataset that aims at performing IMU integration.
            Input 1: one initial 10-dimensional state consisting on initial position (x,y,z), velocity (x,y,z) and
            orientation (w,x,y,z)
            Input 2: a window of imu samples of dimensions <imu_len x 7>, where 7 are the 6 dimensions of the IMU
            readings (3 gyro + 3 acc) plus the time differences between imu acquisitions, and the number of rows are the
            number of used imu samples.
            Output 1: the final 9-dimensional state consisting on final position (x,y,z), velocity (x,y,z) and
            orientation (x,y,z), in so(3) representation
        """
        self.windowed_imu_for_state_prediction(args)

        self.y_ds["state_output"] = np.concatenate((self.y_ds["state_output"][:, :6],
                                                    log_mapping(self.y_ds["state_output"][:, 6:])), axis=1)

    def windowed_imu_preintegration_dataset(self, args):
        """
        :param args: extra arguments for dataset generation

        Generates a dataset that aims at performing IMU integration.
            Input 1: one initial 10-dimensional state consisting on initial position (x,y,z), velocity (x,y,z) and
            orientation (w,x,y,z)
            Input 2: a window of imu samples of dimensions <imu_len x 7>, where 7 are the 6 dimensions of the IMU
            readings (3 gyro + 3 acc) plus the time differences between imu acquisitions, and the number of rows are the
            number of used imu samples.
            Output 1: the final 9-dimensional state consisting on final position (x,y,z), velocity (x,y,z) and
            orientation (x,y,z), in so(3) representation
            Output 2: the pre-integrated rotation for each window element, with shape <n_samples, imu_len, 3> in so(3)
            Output 3: the pre-integrated velocity for each window element, with shape <n_samples, imu_len, 3> in R(3)
            Output 4: the pre-integrated position for each window element, with shape <n_samples, imu_len, 3> in R(3)

        """

        window_len = args[0]

        # TODO: get as a parameter of the dataset
        g_val = 9.81

        n_samples = len(self.imu_raw) - window_len

        self.windowed_imu_for_state_prediction(args)

        gt_augmented = self.x_ds["state_input"]
        gt_augmented = np.concatenate((gt_augmented, self.y_ds["state_output"][-window_len:, :]), axis=0)

        imu_window = self.x_ds["imu_input"]

        # Define the pre-integrated rotation, velocity and position vectors
        pre_int_rot = np.zeros((n_samples, window_len, 3))
        pre_int_v = np.zeros((n_samples, window_len, 3))
        pre_int_p = np.zeros((n_samples, window_len, 3))

        print("Generating pre-integration dataset. This may take a while...")
        prog_bar = Progbar(n_samples)

        for i in range(n_samples):
            pi = np.tile(gt_augmented[i, 0:3], [window_len, 1])
            vi = np.tile(gt_augmented[i, 3:6], [window_len, 1])
            qi = np.tile(gt_augmented[i, 6:], [window_len, 1])

            # imu_window[i, :, -1, 0] is a <1, window_len> vector containing all the dt between two consecutive samples
            # of the imu. We compute the cumulative sum to get the total time for every sample in the window since the
            # beginning of the window itself
            cum_dt_vec = np.cumsum(imu_window[i, :, -1, 0]) / 1000

            # We calculate the quaternion that rotates q(i) to q(i+t) for all t in [0, window_len], and map it to so(3)
            pre_int_rot[i, :, :] = log_mapping(
                np.array([q.elements for q in quaternion_error(qi, gt_augmented[i:i+window_len, 6:])]))

            g_contrib = np.expand_dims(cum_dt_vec * g_val, axis=1)*np.array([0, 0, 1])
            pre_int_v[i, :, :] = rotate_vec(gt_augmented[i:i+window_len, 3:6] - vi - g_contrib, qi)

            v_contrib = np.multiply(np.expand_dims(cum_dt_vec, axis=1), vi)
            g_contrib = 1/2 * np.expand_dims(cum_dt_vec ** 2 * g_val, axis=1)*np.array([0, 0, 1])
            pre_int_p[i, :, :] = rotate_vec(gt_augmented[i:i+window_len, 0:3] - pi - v_contrib - g_contrib, qi)

            prog_bar.update(i+2)

        self.set_outputs(
            ["pre_integrated_R", "pre_integrated_v", "pre_integrated_p"], [pre_int_rot, pre_int_v, pre_int_p])

    def window_imu_data(self, window_len):
        """
        # TODO: complete
        """

        raw_imu = reformat_data(self.imu_raw)

        window_channels = np.shape(raw_imu)[1]

        # Initialize x data. Will be sequence of IMU measurements of size (imu_len x window_channels)
        imu_img_tensor = np.zeros((len(raw_imu), window_len, window_channels, 1))

        for i in range(len(raw_imu) - window_len):
            imu_img = np.zeros((window_len, window_channels))

            # The first imu_x_len data vectors will not be full of data (not enough acquisitions to fill it up yet)
            if i < window_len:
                imu_img[window_len - i - 1:window_len, :] = raw_imu[0:i + 1, :]
            else:
                imu_img = raw_imu[i - window_len + 1:i + 1, :]

            imu_img_tensor[i, :, :, :] = np.expand_dims(imu_img, 2)

        return imu_img_tensor


def reformat_data(compact_data):
    """
    Expands the data (converts from IMU/GT object to a flat-dimensional vector. Additionally, computes the timestamp
    differences

    :param compact_data: data from IMU/GT, with the timestamps at the last component
    """

    data_channels = np.shape(compact_data)[1] - 1

    # Reshape to 2D array
    flattened_data = np.concatenate([np.stack(compact_data[:, i], axis=0) for i in range(data_channels)], axis=1)
    flattened_data = np.append(flattened_data, np.expand_dims(compact_data[:, data_channels], axis=1), axis=1)

    # TODO: get timestamp format (s/ms/us). blackbird is in us
    # Calculate difference between timestamps, and change units to ms
    flattened_data[1:, -1] = np.diff(flattened_data[:, -1]) / 1000
    flattened_data[0, -1] = 0

    return flattened_data
