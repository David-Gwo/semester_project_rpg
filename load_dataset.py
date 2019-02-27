import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from euroc_utils import read_euroc_dataset, GT


def interpolate_ground_truth(raw_imu_data, ground_truth_data):
    raw_imu_data = np.array(raw_imu_data)

    imu_timestamps = np.array([imu_meas.timestamp for imu_meas in raw_imu_data])
    gt_timestsamps = np.array([gt_meas.timestamp for gt_meas in ground_truth_data])
    gt_velocity = np.array([gt_meas.vel for gt_meas in ground_truth_data])

    # Only keep imu data that is within the ground truth time span
    raw_imu_data = raw_imu_data[(imu_timestamps > gt_timestsamps[0]) * (imu_timestamps < gt_timestsamps[-1])]

    imu_timestamps = np.array([imu_meas.timestamp for imu_meas in raw_imu_data])

    # Interpolate Ground truth velocities to match IMU time acquisitions
    v_x_interp = interp1d(gt_timestsamps, gt_velocity[:, 0])
    v_y_interp = interp1d(gt_timestsamps, gt_velocity[:, 1])
    v_z_interp = interp1d(gt_timestsamps, gt_velocity[:, 2])

    # Initialize array of interpolated Ground Truth velocities
    v_interp = [GT() for _ in range(len(imu_timestamps))]

    # Fill in array
    for i, imu_timestamp in enumerate(imu_timestamps):
        v_interp[i] = [v_x_interp(imu_timestamp), v_y_interp(imu_timestamp), v_z_interp(imu_timestamp)]

    return [raw_imu_data, v_interp]


def load_euroc_dataset():
    euroc_dir = 'EuRoC_dataset/mav0/'

    [imu_yaml_data, raw_imu_data, ground_truth_data] = read_euroc_dataset(euroc_dir)

    [raw_imu_data, gt_v_interp] = interpolate_ground_truth(raw_imu_data, ground_truth_data)

    plt.show()


if __name__ == "__main__":
    load_euroc_dataset()


