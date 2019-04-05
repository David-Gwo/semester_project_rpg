import numpy as np
import re
import os
import cv2
import errno

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


def safe_mkdir_recursive(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(dir):
                pass
            else:
                raise
