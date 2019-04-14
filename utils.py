import numpy as np
import os
import re
import matplotlib.pyplot as plt
from data.euroc_manager import plot_prediction


def plot_regression_predictions(test_ds, pred_y, epoch=None, i=0):

    y = [np.squeeze(y_ds) for (_, y_ds) in test_ds]
    y_flat = np.array([item for sublist in y for item in sublist])

    fig = plot_prediction(y_flat, pred_y)

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
