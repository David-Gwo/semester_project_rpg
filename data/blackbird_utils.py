import subprocess
import requests
import gflags
import sys
import os

from data.data_utils import safe_mkdir_recursive
from data.blackbird_flags import FLAGS


VALID_YAW_TYPES = ["yawConstant", "yawForward"]
VALID_TRAJECTORY_NAMES = ["3dFigure8", "ampersand", "bentDice", "cameraCalibration", "clover", "dice", "figure8",
                          "halfMoon", "mouse", "oval", "patrick", "picasso", "sid", "sphinx", "star", "thrice",
                          "tiltedThrice", "winter"]
VALID_MAX_SPEEDS = [0.5, 1, 2, 3, 4, 5, 6, 7]

GT_FILE_NAME = "poses.csv"
DATA_FILE_NAME = "data.bag"
BAG2CSV_SCRIPT = "convert_bag_to_csv.sh"

ds_flags = {}


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
    rc = subprocess.call("./{0} {1} {2}".format(BAG2CSV_SCRIPT, file_name, ds_flags["blackbird_topics"]), shell=True)


def download_blackbird_data(trajectory_name, yaw_type, max_speed):

    assert yaw_type in VALID_YAW_TYPES
    assert trajectory_name in VALID_TRAJECTORY_NAMES
    assert max_speed in VALID_MAX_SPEEDS

    max_speed = encode_max_speed(max_speed)

    dataset_version = "{0}/{1}/{2}".format(trajectory_name, yaw_type, max_speed)
    root = "{0}/{1}/".format(ds_flags["blackbird_url"], dataset_version)

    data_file = "{0}_{1}.bag".format(trajectory_name, max_speed)
    poses_file = "{0}_{1}_poses.csv".format(trajectory_name, max_speed)

    save_dir = "{0}/{1}".format(ds_flags["blackbird_local_dir"], dataset_version)
    safe_mkdir_recursive(save_dir)

    pose_file_dir = "{0}/{1}".format(save_dir, GT_FILE_NAME)
    data_file_dir = "{0}/{1}".format(save_dir, DATA_FILE_NAME)

    url = "{0}/{1}".format(root, poses_file)

    if not os.path.exists(pose_file_dir):
        save_url_to_file(pose_file_dir, url)
    else:
        print("Ground truth data file already available")

    url = "{0}/{1}".format(root, data_file)
    if not os.path.exists(data_file_dir):
        save_url_to_file(data_file_dir, url)

    else:
        print("Sensor file already available")
    convert_to_csv(data_file_dir)


def read_blackbird_data(trajectory_name, yaw_type, max_speed):
    dataset_version = "{0}/{1}/{2}".format(trajectory_name, yaw_type, encode_max_speed(max_speed))
    save_dir = "{0}/{1}".format(ds_flags["blackbird_local_dir"], dataset_version)
    data_file_dir = "{0}/{1}".format(save_dir, DATA_FILE_NAME)


def main(argv):
    try:
        _ = FLAGS(argv)
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)

    for key in FLAGS.__flags.keys():
        ds_flags[key] = getattr(FLAGS, key)

    download_blackbird_data("bentDice", "yawForward", 0.5)
    read_blackbird_data("bentDice", "yawForward", 0.5)


if __name__ == "__main__":
    main(sys.argv)
