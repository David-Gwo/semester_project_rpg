import subprocess
import gflags
import sys
import csv
import os

import numpy as np
import quaternion as q

from data.euroc_utils import IMU, GT, interpolate_ground_truth, pre_process_data, add_scaler_ref_to_training_dir, \
    generate_tf_imu_train_ds
from data.data_utils import safe_mkdir_recursive, get_file_from_url
from data.data_utils import generate_imu_speed_integration_dataset, save_processed_dataset_files
from data.blackbird_flags import FLAGS


class BBIMU(IMU):
    def __init__(self):
        super(BBIMU, self).__init__()
        self.gyro_indx = [14, 15, 16]
        self.acc_indx = [19, 20, 21]

    def read(self, data):
        data = np.array(data)
        self.timestamp = data[0].astype(np.float) / 1000
        self.gyro = data[self.gyro_indx].astype(np.float)
        self.acc = data[self.acc_indx].astype(np.float)


class BBGT(GT):
    def __init__(self):
        super(BBGT, self).__init__()

    def read(self, data):
        data = np.array(data)
        data = data.astype(np.float)
        self.timestamp = data[0]
        self.pos = data[1:4]
        self.att = data[4:8]

    def integrate(self, gt_old):

        dt = (self.timestamp - gt_old.timestamp) * 10e-6
        self.vel = (self.pos - gt_old.pos) / dt
        att_q = q.quaternion(self.att[0], self.att[1], self.att[2], self.att[3])
        att = q.as_euler_angles(att_q)
        old_att_q = q.quaternion(gt_old.att[0], gt_old.att[1], gt_old.att[2], gt_old.att[3])
        old_att = q.as_euler_angles(old_att_q)
        self.ang_vel = (att - old_att) / dt


class BlackbirdDSManager:
    def __init__(self, *args):

        self.ds_flags = FLAGS

        # Accepted Blackbird parameters
        self.valid_yaw_types = ["yawConstant", "yawForward"]
        self.valid_trajectory_names = \
            ["3dFigure8", "ampersand", "bentDice", "cameraCalibration", "clover", "dice", "figure8", "halfMoon",
             "mouse", "oval", "patrick", "picasso", "sid", "sphinx", "star", "thrice", "tiltedThrice", "winter"]
        self.valid_max_speeds = [0.5, 1, 2, 3, 4, 5, 6, 7]

        # Inner pipeline variables
        self.gt_file_name = "poses.csv"
        self.data_file_name = "data.bag"
        self.csv_imu_file_name = "data/_slash_blackbird_slash_imu.csv"
        self.bag2csv_script = "./data/convert_bag_to_csv.sh"

        try:
            _ = FLAGS(args)  # parse flags
        except gflags.FlagsError:
            print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
            sys.exit(1)

        self.ds_version = self.get_dataset_version()
        self.ds_local_dir = "{0}{1}/".format(self.ds_flags.blackbird_local_dir, self.ds_version)

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
        print("Transforming bag file to csv...")
        subprocess.call("./{0} {1} {2}".format(self.bag2csv_script, file_name, self.ds_flags.blackbird_topics), shell=True)
        print("Done")

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

        save_dir = "{0}{1}/".format(self.ds_flags.blackbird_local_dir, self.ds_version)
        safe_mkdir_recursive(save_dir)
        pose_file_dir = "{0}{1}".format(save_dir, self.gt_file_name)
        data_file_dir = "{0}{1}".format(save_dir, self.data_file_name)

        max_speed = self.encode_max_speed(self.ds_flags.max_speed)

        # root url of github repo
        root = "{0}/{1}/".format(self.ds_flags.blackbird_url, self.ds_version)
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

        return save_dir

    def read_blackbird_data(self, save_dir):
        data_file_dir = "{0}{1}".format(save_dir, self.csv_imu_file_name)
        gt_file_dir = "{0}{1}".format(save_dir, self.gt_file_name)

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

        return [raw_imu_data, ground_truth_data]


def generate_speed_integration_dataset(imu_len, raw_imu, gt_v, ds_dir, train_file_name, test_file_name):

    imu_img_tensor, gt_v_tensor = generate_imu_speed_integration_dataset(raw_imu, gt_v, imu_len)
    euroc_training_ds = "{0}{1}".format(ds_dir, train_file_name)
    euroc_testing_ds = "{0}{1}".format(ds_dir, test_file_name)
    save_processed_dataset_files(euroc_training_ds, euroc_testing_ds, imu_img_tensor, gt_v_tensor)


def load_blackbird_dataset(batch_size, imu_w_len, train_file_name, test_file_name, processed_ds_available,
                           trained_model_dir):

    bbds = BlackbirdDSManager()

    if not processed_ds_available:
        save_dir = bbds.download_blackbird_data()
        raw_imu_data, ground_truth_data = bbds.read_blackbird_data(save_dir)
        raw_imu_data, gt_v_interp = interpolate_ground_truth(raw_imu_data, ground_truth_data)

        # Cut away last few samples (outlier)
        raw_imu_data = raw_imu_data[0:int(np.ceil(0.95*len(raw_imu_data)))]
        gt_v_interp = gt_v_interp[0:int(np.ceil(0.95*len(gt_v_interp)))]

        processed_imu, processed_v = pre_process_data(raw_imu_data, gt_v_interp, save_dir)
        generate_speed_integration_dataset(imu_w_len, processed_imu, processed_v, bbds.ds_local_dir, train_file_name,
                                           test_file_name)

    add_scaler_ref_to_training_dir(bbds.ds_local_dir, trained_model_dir)
    return generate_tf_imu_train_ds(bbds.ds_local_dir, train_file_name, batch_size, trained_model_dir)
