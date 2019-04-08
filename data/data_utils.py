import numpy as np
import re
import os
import sys
import cv2
import errno
import requests
import scipy.io

from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras import datasets as k_ds
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.data import Dataset
############################################################################
# EXAMPLE CLASS TO FETCH FILENAMES (AND OPTIONALLY LABELS) FROM DIRECTORIES#
############################################################################


class DirectoryIterator(Iterator):
    """
    Class for managing data loading.of images and labels
    We assume that the folder structure is:
    root_folder/
           folder_1/
                    images/
                    labels.txt
           folder_2/
                    images/
                    labels.txt
           .
           .
           folder_n/
                    images/
                    labels.txt

    # Arguments
       directory: Path to the root directory to read data from.
       target_size: tuple of integers, dimensions to resize input images to.
       batch_size: The desired batch size
       shuffle: Whether to shuffle data or not
       seed : numpy seed to shuffle data
       follow_links: Bool, whether to follow symbolic links or not

    """
    def __init__(self, directory, target_size=(224, 224), batch_size=32, shuffle=True, seed=None, follow_links=False):
        self.directory = directory
        self.target_size = tuple(target_size)
        self.follow_links = follow_links
        self.image_shape = self.target_size + (3,)

        # First count how many experiments are out there
        self.samples = 0

        experiments = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                experiments.append(subdir)
        self.num_experiments = len(experiments)
        self.formats = {'png'}
        print("----------------------------------------------------")
        print("Loading the following formats {}".format(self.formats))
        print("----------------------------------------------------")

        # Associate each filename with a corresponding label
        self.filenames = []
        self.ground_truth = []

        for subdir in experiments:
            subpath = os.path.join(directory, subdir)
            try:
                self._decode_experiment_dir(subpath)
            except:
                raise ImportWarning("Image reading in {} failed".format(subpath))

        if self.samples == 0:
            raise IOError("Did not find any file in the dataset folder")

        # Conversion of list into array
        self.ground_truth = np.array(self.ground_truth, dtype = np.uint8)

        print('Found {} images belonging to {} experiments.'.format(self.samples, self.num_experiments))
        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links), key=lambda tpl: tpl[0])

    def _decode_experiment_dir(self, dir_subpath):
        labels_filename = os.path.join(dir_subpath, "labels.txt")

        # Try load labels
        try:
            ground_truth = np.loadtxt(labels_filename, usecols=0)
        except:
            raise IOError("Labels file was not found found")

        # Now fetch all images in the image subdir
        image_dir_path = os.path.join(dir_subpath, "images")
        for root, _, files in self._recursive_list(image_dir_path):
            sorted_files = sorted(files, key=lambda fname: int(re.search(r'\d+', fname).group()))
            for frame_number, fname in enumerate(sorted_files):
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    self.filenames.append(absolute_path)
                    self.ground_truth.append(ground_truth[frame_number])
                    self.samples += 1

    def next(self):
        """
        Public function to fetch next batch. Note that this function
        will only be used for EVALUATION and TESTING, but not for training.

        # Returns
            The next batch of images and labels.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(
                    self.index_generator)

        # Image transformation is not under thread lock, so it can be done in
        # parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=np.uint8)
        batch_y = np.zeros((current_batch_size,), dtype=np.uint8)

        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = load_img(os.path.join(fname))
            batch_x[i] = x

        batch_y = self.ground_truth[index_array]

        return batch_x, batch_y


def load_img(path, target_size=None):
    """
    Load an image. Ans reshapes it to target size

    # Arguments
        path: Path to image file.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_width, img_height)`.

    # Returns
        Image as numpy array.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_mnist_datasets(img_h, img_w, batch_s):

    fashion_mnist = k_ds.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Further break training data into train / validation sets
    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]

    # Reshape input data from (28, 28) to (28, 28, 1)
    w, h = img_w, img_h
    x_train = x_train.reshape(x_train.shape[0], w, h, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
    x_test = x_test.reshape(x_test.shape[0], w, h, 1)

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_valid = to_categorical(y_valid, 10)
    y_test = to_categorical(y_test, 10)

    train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(batch_s).batch(batch_s).repeat()
    validation_ds = Dataset.from_tensor_slices((x_valid, y_valid)).shuffle(batch_s).batch(batch_s).repeat()
    test_ds = Dataset.from_tensor_slices((x_test, y_test)).shuffle(batch_s).batch(batch_s)
    ds_lengths = (len(x_train), len(x_valid))

    return train_ds, validation_ds, ds_lengths


def safe_mkdir_recursive(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise


def get_file_from_url(file_name, link):
    with open(file_name, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s%s]" % ('=' * done, '>', ' ' * (50 - done - 1)))
                sys.stdout.flush()


def generate_imu_img_dataset(imu_vec, gt_vec, imu_len):
    """
    Generates a dataset of imu images to regress linear speed. An imu image is defined as a stack of matrices of
    dimensions <imu_len x 6>, where 6 are the 6 dimensions of the IMU readings (3 gyro + 3 acc), and the number of
    rows are the number of used imu samples to regress the speed corresponding to the last sample.

    :param imu_vec: vector of ordered IMU readings. Shape: <n, 2, 3>, n = number of acquisitions, imu_vec[:, 0, :]
    corresponds to the three gyro readings and imu_vec[:, 1, :] to the tree accelerometer readings
    :param gt_vec: ground truth velocity data. Shape: <n, 3>, n = number of acquisitions, and each acquisition is a
    3-dimensional vector with x,y,z velocities
    :param imu_len: number of columns of the imu_image
    :return: the constructed dataset following the above indications, in the format imu_img_tensor, gt_tensor
    """

    # Initialize x data. Will be sequence of IMU measurements of size (imu_len x 6)
    imu_img_tensor = np.zeros((len(imu_vec), imu_len, 6, 1))
    # Initialize y data. Will be the absolute ground truth value of the speed of the drone
    gt_v_tensor = np.zeros(len(imu_vec))

    for i in range(len(imu_vec) - imu_len):
        imu_img = np.zeros((imu_len, 6))

        # The first imu_x_len data vectors will not be full of data (not enough acquisitions to fill it up yet)
        if i < imu_len:
            imu_img[imu_len - i - 1:imu_len, :] = imu_vec[0:i+1, :, :].reshape(i+1, 6)
        else:
            imu_img = imu_vec[i:i + imu_len, :, :].reshape(imu_len, 6)

        # TODO: Should the elapsed time be included in the data?

        imu_img_tensor[i, :, :, :] = np.expand_dims(imu_img, 2)
        gt_v_tensor[i] = np.linalg.norm(gt_vec[i])

    return imu_img_tensor, gt_v_tensor


def generate_imu_speed_integration_dataset(imu_vec, gt_vec, imu_len):
    """
    Generates a dataset of combine imu and ground truth velocity data to do one-step speed prediction. The training data
    is defined as a stack of matrices of dimensions <imu_len x 7>, where the first 6 of the 7 columns are the IMU
    readings (3 gyro + 3 acc), and the 7th is the corresponding ground truth velocity. The number of rows correspond to
    the number of used imu samples to predict the speed for next timestamp.

    :param imu_vec: vector of ordered IMU readings. Shape: <n, 2, 3>, n = number of acquisitions, imu_vec[:, 0, :]
    corresponds to the three gyro readings and imu_vec[:, 1, :] to the tree accelerometer readings
    :param gt_vec: ground truth velocity data. Shape: <n, 3>, n = number of acquisitions, and each acquisition is a
    3-dimensional vector with x,y,z velocities
    :param imu_len: number of columns of the imu_image
    :return: the constructed dataset following the above indications, in the format imu_img_tensor, gt_tensor
    """

    # Initialize x data. Will be sequence of IMU measurements of size (imu_len x 6)
    imu_img_tensor = np.zeros((len(imu_vec) - 1, imu_len, 7, 1))
    # Initialize y data. Will be the absolute ground truth value of the speed of the drone
    gt_v_tensor = np.zeros(len(imu_vec) - 1)

    gt_vec = np.expand_dims(np.linalg.norm(gt_vec, axis=1), axis=1)
    imu_vec = np.append(imu_vec, np.zeros((imu_len - 1, 2, 3)), axis=0)

    for i in range(len(imu_vec) - imu_len - 1):
        imu_img = np.append(imu_vec[i:i + imu_len, :, :].reshape(imu_len, 6), gt_vec[i:i + imu_len], axis=1)

        imu_img_tensor[i] = np.expand_dims(imu_img, 2)

        gt_v_tensor[i] = gt_vec[i + imu_len]

    return imu_img_tensor, gt_v_tensor


def save_processed_dataset_files(train_ds_node, test_ds_node, x_data, y_data):
    """
    Saves a copy of the train & test datasets as a mat file in a specified file names

    :param train_ds_node: Training ds file name <dir/file.mat>
    :param test_ds_node: Test ds file name <dir/file.mat>
    :param x_data: x data (samples in first dimension)
    :param y_data: y data (samples in first dimension)
    """
    if os.path.exists(train_ds_node):
        os.remove(train_ds_node)
    os.mknod(train_ds_node)

    if os.path.exists(test_ds_node):
        os.remove(test_ds_node)
    os.mknod(test_ds_node)

    total_ds_len = int(len(y_data))
    test_ds_len = int(np.ceil(total_ds_len * 0.1))

    # Choose some entries to separate for the test set
    test_indexes = np.random.choice(total_ds_len, test_ds_len, replace=False)

    test_set_x = x_data[test_indexes]
    test_set_y = y_data[test_indexes]

    # Remove the test ds entries from train dataset
    train_set_x = np.delete(x_data, test_indexes, 0)
    train_set_y = np.delete(y_data, test_indexes)

    scipy.io.savemat(train_ds_node, mdict={'x': train_set_x, 'y': train_set_y}, oned_as='row')
    scipy.io.savemat(test_ds_node, mdict={'x': test_set_x, 'y': test_set_y}, oned_as='row')


def load_mat_data(directory):
    """
    Loads a dataset saved with the save_processed_dataset_files function

    :param directory: directory of the mat file <dir/file.mat>
    :return: the x and y data, in format [x, y]
    """

    mat_data = scipy.io.loadmat(directory)

    y_tensor = np.expand_dims(mat_data['y'][0], 1)
    x_tensor = mat_data['x']

    return x_tensor, y_tensor
