import numpy as np
import os
import re
import matplotlib.pyplot as plt
from data.euroc_manager import plot_prediction


def plot_regression_predictions(test_ds, pred_y, manual_pred=None, epoch=None, i=0):

    y = [np.squeeze(y_ds) for (_, y_ds) in test_ds]
    y_flat = np.array([item for sublist in y for item in sublist])

    fig = plot_prediction(y_flat, pred_y, manual_pred)

    if epoch is not None:
        if i != 0:
            fig.savefig('figures/fig_{0}_{1}.png'.format(epoch, i))
        else:
            fig.savefig('figures/fig_{0}'.format(epoch))
        plt.close(fig)

    else:
        plt.show()


def get_checkpoint_file_list(checkpoint_dir, name):
    regex = name + r"_[0-9]"
    files = [f for f in os.listdir(checkpoint_dir) if re.match(regex, f)]
    files.sort(key=str.lower)
    return files


def imu_integration(data, window_len):

    # TODO: get a better comparison

    # Get features of training set. Discard final labels
    x = [np.squeeze(x_ds) for (x_ds, y_ds) in data]
    x_flat = np.array([item for sublist in x for item in sublist])

    samples = len(x_flat)
    output_dim = np.shape(x_flat)[1] - window_len

    out = np.zeros((samples, output_dim))

    imu, t_diff, x_0 = x_flat[:, :window_len, :6], x_flat[:, :window_len, 6:], x_flat[:, window_len:, 0]

    # Convert time diff to seconds
    t_diff = np.squeeze(np.stack(t_diff/1000))

    acc_correction = np.zeros((window_len, 3))
    acc_correction[:, -1] = 9.81

    for sample in range(samples):

        dt = np.tile(t_diff[sample, :], (window_len, 1))

        acc = imu[sample, :, 3:] + acc_correction

        dv = np.sum(dt.dot(acc), axis=0)
        dx = np.sum(0.5*dt.dot(dt.dot(acc)), axis=0)

        out[sample, 0:3] = x_0[sample, 0:3] + dx
        out[sample, 3:6] = x_0[sample, 3:6] + dv
        out[sample, 6:] = x_0[sample, 6:]

    return out
