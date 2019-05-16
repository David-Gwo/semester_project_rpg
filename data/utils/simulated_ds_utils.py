import subprocess
import gflags
import sys
import csv
import os

import numpy as np
from pyquaternion import Quaternion

from utils.directories import safe_mkdir_recursive
from utils.algebra import correct_quaternion_flip
from data.config.blackbird_flags import FLAGS
from data.inertial_ABCs import IMU, GT, InertialDataset
from data.utils.data_utils import get_file_from_url


class GenIMU(IMU):
    def __init__(self):
        super(GenIMU, self).__init__()
        self.gyro_indx = [14, 15, 16]
        self.acc_indx = [19, 20, 21]

    def read(self, data):
        data = np.array(data)
        self.timestamp = data[0].astype(np.float) / 1000
        self.gyro = data[self.gyro_indx].astype(np.float)
        self.acc = data[self.acc_indx].astype(np.float)


class GenGT(GT):
    def __init__(self):
        super(GenGT, self).__init__()

    def read(self, data):
        data = np.array(data)
        data = data.astype(np.float)
        self.timestamp = data[0]
        self.pos = data[1:4]
        self.att = data[4:8]


class BlackbirdDSManager(InertialDataset):
    def __init__(self, *args):
        super(BlackbirdDSManager, self).__init__()

        self.sampling_freq = 100

        self.ds_flags = FLAGS

        # Accepted Blackbird parameters
        self.valid_yaw_types = ["yawConstant", "yawForward"]
        self.valid_trajectory_names = \
            ["3dFigure8", "ampersand", "bentDice", "clover", "dice", "figure8", "halfMoon",
             "mouse", "oval", "patrick", "picasso", "sid", "sphinx", "star", "thrice", "tiltedThrice", "winter"]
        self.valid_max_speeds = [0.5, 1, 2, 3, 4, 5, 6, 7]

        # Inner pipeline variables
        self.gt_file_name = "poses.csv"
        self.data_file_name = "data.bag"
        self.csv_imu_file_name = "data/_slash_blackbird_slash_imu.csv"
        self.bag2csv_script = "./data/utils/convert_bag_to_csv.sh"

        self.blackbird_local_dir = './data/dataset/blackbird_dataset/'
        self.blackbird_url = 'http://blackbird-dataset.mit.edu/BlackbirdDatasetData'
        self.rosbag_topics = '/blackbird/imu'

        try:
            _ = FLAGS(args)  # parse flags
        except gflags.FlagsError:
            print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
            sys.exit(1)

        self.ds_version = self.get_dataset_version()
        self.ds_local_dir = "{0}{1}/".format(self.blackbird_local_dir, self.ds_version)

    @staticmethod
    def encode_max_speed(max_speed):
        if max_speed == 0.5:
            return "maxSpeed0p5"
        elif max_speed == 1:
            return "maxSpeed1p0"
        elif max_speed == 2:
            return "maxSpeed2p0"
        elif max_speed == 3:
            return "maxSpeed3p0"
        elif max_speed == 4:
            return "maxSpeed4p0"
        elif max_speed == 5:
            return "maxSpeed5p0"
        elif max_speed == 6:
            return "maxSpeed6p0"
        else:
            return "maxSpeed7p0"

    def convert_to_csv(self, file_name):
        print("\nTransforming bag file to csv...")
        subprocess.call("./{0} {1} {2}".format(self.bag2csv_script, file_name, self.rosbag_topics), shell=True)

    def get_dataset_version(self):

        yaw_type = self.ds_flags.yaw_type
        trajectory_name = self.ds_flags.trajectory_name
        max_speed = self.ds_flags.max_speed

        assert yaw_type in self.valid_yaw_types
        assert trajectory_name in self.valid_trajectory_names
        assert max_speed in self.valid_max_speeds

        max_speed = self.encode_max_speed(max_speed)

        dataset_version = "{0}/{1}/{2}".format(trajectory_name, yaw_type, max_speed)

        return dataset_version

    def download_blackbird_data(self):

        safe_mkdir_recursive(self.ds_local_dir)
        pose_file_dir = "{0}{1}".format(self.ds_local_dir, self.gt_file_name)
        data_file_dir = "{0}{1}".format(self.ds_local_dir, self.data_file_name)

        max_speed = self.encode_max_speed(self.ds_flags.max_speed)

        # root url of github repo
        root = "{0}/{1}/".format(self.blackbird_url, self.ds_version)
        data_file = "{0}_{1}.bag".format(self.ds_flags.trajectory_name, max_speed)
        poses_file = "{0}_{1}_poses.csv".format(self.ds_flags.trajectory_name, max_speed)

        url = "{0}{1}".format(root, poses_file)

        if not os.path.exists(pose_file_dir):
            get_file_from_url(pose_file_dir, url)
        else:
            print("Ground truth data file already available")

        url = "{0}/{1}".format(root, data_file)
        if not os.path.exists(data_file_dir):
            get_file_from_url(data_file_dir, url)
            self.convert_to_csv(data_file_dir)

        else:
            print("Sensor file already available")

    def read_blackbird_data(self):
        data_file_dir = "{0}{1}".format(self.ds_local_dir, self.csv_imu_file_name)
        gt_file_dir = "{0}{1}".format(self.ds_local_dir, self.gt_file_name)

        raw_imu_data = []
        ground_truth_data = []

        with open(data_file_dir, 'rt') as csv_file:
            header_line = 1
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                if header_line:
                    header_line = 0
                    continue

                imu = BBIMU()
                imu.read(row)
                raw_imu_data.append(imu)

        with open(gt_file_dir, 'rt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header_line = 1

            gt_old = BBGT()

            for row in csv_reader:

                gt = BBGT()
                gt.read(row)
                ground_truth_data.append(gt)

                if header_line:
                    header_line = 0
                    gt_old = gt
                    continue

                gt.integrate(gt_old)

                gt_old = gt

        self.imu_data = raw_imu_data
        self.gt_data = ground_truth_data

    def get_raw_ds(self):

        self.download_blackbird_data()
        self.read_blackbird_data()
        self.interpolate_ground_truth()

        # Cut away last 5% samples (noisy measurements)
        self.imu_data = self.imu_data[0:int(np.ceil(0.95 * len(self.imu_data)))]
        self.gt_data = self.gt_data[0:int(np.ceil(0.95 * len(self.gt_data)))]

        return self.imu_data, self.gt_data

    def pre_process_data(self, gyro_scale_file, acc_scale_file, filter_freq):
        super(BlackbirdDSManager, self).pre_process_data(gyro_scale_file, acc_scale_file, filter_freq)

        corrected_quaternion = correct_quaternion_flip(np.stack(self.gt_data[:, 2]))
        for i in range(len(self.gt_data)):
            self.gt_data[i, 2] = corrected_quaternion[i, :]

        return self.imu_data, self.gt_data
