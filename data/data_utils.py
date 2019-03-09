import numpy as np
import re
import os
import cv2

from tensorflow.python.keras.preprocessing.image import Iterator

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
    def __init__(self, directory,
            target_size=(224,224),
            batch_size=32, shuffle=True, seed=None, follow_links=False):
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
