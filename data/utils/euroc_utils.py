import csv
import yaml
import numpy as np
import gflags
import sys

from data.inertial_ABCs import IMU, GT, InertialDataset
from data.config.euroc_flags import FLAGS


class EurocIMU(IMU):
    def __init__(self):
        super(EurocIMU, self).__init__()

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


class EurocDSManager(InertialDataset):
    def __init__(self, *args):
        super(EurocDSManager, self).__init__()

        self.sampling_freq = 200

        self.imu_data_file = 'mav0/imu0/data.csv'
        self.sensor_yaml_file = 'mav0/imu0/sensor.yaml'
        self.gt_data_file = 'mav0/state_groundtruth_estimate0/data.csv'

        self.ds_flags = FLAGS

        try:
            _ = FLAGS(args)  # parse flags
        except gflags.FlagsError:
            print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
            sys.exit(1)

        self.euroc_local_dir = './data/dataset/EuRoC_dataset/'

        self.ds_local_dir = "{0}{1}/".format(self.euroc_local_dir, self.ds_flags.dataset_version)

    def read_euroc_data(self):

        imu_file = "{0}{1}".format(self.ds_local_dir, self.imu_data_file)
        imu_yaml_file = "{0}{1}".format(self.ds_local_dir, self.sensor_yaml_file)
        ground_truth_file = "{0}{1}".format(self.ds_local_dir, self.gt_data_file)

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

                    imu = EurocIMU()
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

                    gt = EurocGT()
                    gt.read(row)
                    ground_truth_data.append(gt)

        except IOError:
            print("Dataset file not found")

        except yaml.YAMLError as exc:
            print(exc)

        self.imu_data = raw_imu_data
        self.gt_data = ground_truth_data

    def get_raw_ds(self):

        self.read_euroc_data()
        self.interpolate_ground_truth()

        # Cut away last 1% samples (noisy measurements)
        self.imu_data = self.imu_data[0:int(np.ceil(0.2 * len(self.imu_data)))]
        self.gt_data = self.gt_data[0:int(np.ceil(0.2 * len(self.gt_data)))]

        return self.imu_data, self.gt_data

    def pre_process_data(self, gyro_scale_file, acc_scale_file):
        self.basic_preprocessing(gyro_scale_file, acc_scale_file, 10)

        # self.imu_data, self.gt_data = expand_dataset_region(self.imu_data, self.gt_data)

        return self.imu_data, self.gt_data


def expand_dataset_region(filt_imu_vec, filt_gt_v_interp):
    # Add more flat region so avoid model from learning average value
    flat_region = filt_imu_vec[6000:7000, :]
    flat_region_v = filt_gt_v_interp[6000:7000, :]
    flat_region = np.repeat(np.concatenate((flat_region, flat_region[::-1, :])), [4], axis=0)
    flat_region_v = np.repeat(np.concatenate((flat_region_v, flat_region_v[::-1])), [4], axis=0)
    filt_imu_vec = np.concatenate((filt_imu_vec[0:6000, :], flat_region, filt_imu_vec[7000:, :]))
    filt_gt_v_interp = np.concatenate((filt_gt_v_interp[0:6000, :], flat_region_v, filt_gt_v_interp[7000:, :]))

    return filt_imu_vec, filt_gt_v_interp
