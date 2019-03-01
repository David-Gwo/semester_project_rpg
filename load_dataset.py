import tensorflow as tf
import numpy as np

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
        v_interp[i] = np.array([v_x_interp(imu_timestamp), v_y_interp(imu_timestamp), v_z_interp(imu_timestamp)])

    return [raw_imu_data, v_interp]


def generate_cnn_train_data(image_size, raw_imu, gt_v):

    assert image_size % 10 == 0

    # Initialize data tensors
    imu_img_tensor = np.zeros((len(raw_imu), 10, int(image_size/10), 6))
    gt_v_tensor = np.zeros((len(raw_imu),3))

    for i, _ in enumerate(raw_imu[0:len(raw_imu)-image_size]):
        imu_img = np.zeros((image_size, 6))
        if i < image_size:
            imu_img[image_size-i-1:image_size, :] = \
                np.array([list(imu_s.gyro) + list(imu_s.acc) for imu_s in raw_imu[0:i+1]])
        else:
            imu_img = np.array([list(imu_s.gyro) + list(imu_s.acc) for imu_s in raw_imu[i:i+image_size]])

        # TODO: Should the elapsed time be included in the data?

        imu_img_tensor[i, :, :, :] = np.reshape(imu_img, (10, int(image_size/10), 6))
        gt_v_tensor[i, :] = gt_v[i]

    return [imu_img_tensor, gt_v_tensor]


def load_euroc_dataset():
    euroc_dir = 'EuRoC_dataset/mav0/'
    image_size = 200
    batch_size = 64

    [imu_yaml_data, raw_imu_data, ground_truth_data] = read_euroc_dataset(euroc_dir)

    [raw_imu_data, gt_v_interp] = interpolate_ground_truth(raw_imu_data, ground_truth_data)

    [x_tensor, y_tensor] = generate_cnn_train_data(image_size, raw_imu_data, gt_v_interp)

    data_set = tf.data.Dataset.from_tensor_slices(x_tensor)
    data_set = data_set.shuffle().cache().batch(batch_size=batch_size)
    iterator = data_set.make_one_shot_iterator()

    input_layer = next(iterator)

    # out = net(input_layer)

    # train...


if __name__ == "__main__":
    load_euroc_dataset()
