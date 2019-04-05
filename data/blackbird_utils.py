import subprocess
import requests
import gflags
import sys
import csv
import os

import numpy as np
import quaternion as q

from data.euroc_utils import IMU, GT, interpolate_ground_truth, pre_process_data
from data.data_utils import safe_mkdir_recursive
from data.blackbird_flags import FLAGS


VALID_YAW_TYPES = ["yawConstant", "yawForward"]
VALID_TRAJECTORY_NAMES = ["3dFigure8", "ampersand", "bentDice", "cameraCalibration", "clover", "dice", "figure8",
                          "halfMoon", "mouse", "oval", "patrick", "picasso", "sid", "sphinx", "star", "thrice",
                          "tiltedThrice", "winter"]
VALID_MAX_SPEEDS = [0.5, 1, 2, 3, 4, 5, 6, 7]

GT_FILE_NAME = "poses.csv"
DATA_FILE_NAME = "data.bag"
CSV_IMU_FILE_NAME = "data/_slash_blackbird_slash_imu.csv"
BAG2CSV_SCRIPT = "convert_bag_to_csv.sh"
GYRO_INDX = [14, 15, 16]
ACC_INDX = [19, 20, 21]

ds_flags = {}


class BBIMU(IMU):
    def __init__(self):
        super(BBIMU, self).__init__()

    def read(self, data):
        data = np.array(data)
        self.timestamp = data[0].astype(np.float) / 1000
        self.gyro = data[GYRO_INDX].astype(np.float)
        self.acc = data[ACC_INDX].astype(np.float)


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


def save_url_to_file(file_name, link):
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


def convert_to_csv(file_name):
    print("Transforming bag file to csv...")
    subprocess.call("./{0} {1} {2}".format(BAG2CSV_SCRIPT, file_name, ds_flags["blackbird_topics"]), shell=True)
    print("Done")


def download_blackbird_data(trajectory_name, yaw_type, max_speed):

    assert yaw_type in VALID_YAW_TYPES
    assert trajectory_name in VALID_TRAJECTORY_NAMES
    assert max_speed in VALID_MAX_SPEEDS

    max_speed = encode_max_speed(max_speed)

    dataset_version = "{0}/{1}/{2}".format(trajectory_name, yaw_type, max_speed)
    save_dir = "{0}/{1}".format(ds_flags["blackbird_local_dir"], dataset_version)
    safe_mkdir_recursive(save_dir)
    pose_file_dir = "{0}/{1}".format(save_dir, GT_FILE_NAME)
    data_file_dir = "{0}/{1}".format(save_dir, DATA_FILE_NAME)

    # root url of github repo
    root = "{0}/{1}/".format(ds_flags["blackbird_url"], dataset_version)
    data_file = "{0}_{1}.bag".format(trajectory_name, max_speed)
    poses_file = "{0}_{1}_poses.csv".format(trajectory_name, max_speed)

    url = "{0}/{1}".format(root, poses_file)

    if not os.path.exists(pose_file_dir):
        save_url_to_file(pose_file_dir, url)
    else:
        print("Ground truth data file already available")

    url = "{0}/{1}".format(root, data_file)
    if not os.path.exists(data_file_dir):
        save_url_to_file(data_file_dir, url)
        convert_to_csv(data_file_dir)

    else:
        print("Sensor file already available")

    return dataset_version, save_dir


def read_blackbird_data(save_dir):
    data_file_dir = "{0}/{1}".format(save_dir, CSV_IMU_FILE_NAME)
    gt_file_dir = "{0}/{1}".format(save_dir, GT_FILE_NAME)

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


def main(argv):
    try:
        _ = FLAGS(argv)
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)

    for key in FLAGS.__flags.keys():
        ds_flags[key] = getattr(FLAGS, key)

    # TODO: pass as argument:
    trajectory_name = "bentDice"
    yaw_type = "yawForward"
    max_speed = 0.5

    dataset_version, save_dir = download_blackbird_data(trajectory_name, yaw_type, max_speed)
    raw_imu_data, ground_truth_data = read_blackbird_data(save_dir)
    raw_imu_data, gt_v_interp = interpolate_ground_truth(raw_imu_data, ground_truth_data)
    processed_imu, processed_v = pre_process_data(raw_imu_data, gt_v_interp, save_dir + '/')

    print("Hello world")


if __name__ == "__main__":
    main(sys.argv)
