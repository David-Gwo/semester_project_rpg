import csv
import yaml
import numpy as np


class IMU:
    def __init__(self):
        self.timestamp = 0.0
        self.gyro = [0.0, 0.0, 0.0]
        self.acc = [0.0, 0.0, 0.0]

    def read(self, data):
        data = np.array(data)
        data = data.astype(np.float)
        self.timestamp = data[0]
        self.gyro = data[1:4]
        self.acc = data[4:7]


class GT:
    def __init__(self):
        self.timestamp = 0.0
        self.pos = [0.0, 0.0, 0.0]
        self.att = [0.0, 0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.ang_vel = [0.0, 0.0, 0.0]
        self.acc = [0.0, 0.0, 0.0]

    def read(self, data):
        data = np.array(data)
        data = data.astype(np.float)
        self.timestamp = data[0]
        self.pos = data[1:4]
        self.att = data[4:8]
        self.vel = data[8:11]
        self.ang_vel = data[11:14]
        self.acc = data[14:17]


def read_euroc_dataset(euroc_dir):
    imu_file = euroc_dir + 'imu0/data.csv'
    imu_yaml_file = euroc_dir + 'imu0/sensor.yaml'
    ground_truth_file = euroc_dir + 'state_groundtruth_estimate0/data.csv'

    imu_yaml_data = dict()
    raw_imu_data = []
    ground_truth_data = []

    try:
        # Read IMU data
        with open(imu_file, 'rb') as csv_file:
            header_line = 1
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                if header_line:
                    header_line = 0
                    continue

                imu = IMU()
                imu.read(row)
                raw_imu_data.append(imu)

        # Read IMU sensor yaml file
        with open(imu_yaml_file, 'r') as stream:
            imu_yaml_data = yaml.load(stream)

        # Read ground truth data
        with open(ground_truth_file, 'rb') as csv_file:
            header_line = 1
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                if header_line:
                    header_line = 0
                    continue

                gt = GT()
                gt.read(row)
                ground_truth_data.append(gt)

    except IOError:
        print "Dataset file not found"

    except yaml.YAMLError as exc:
        print exc

    return [imu_yaml_data, raw_imu_data, ground_truth_data]
