import numpy as np
import os
import re
from tensorflow.python.keras.utils.generic_utils import Progbar
import matplotlib.pyplot as plt


def compute_loss(sess, learner_object, generator, steps,
                      verbose=0):
    """Generates predictions for the input samples from a data generator.
    The generator should return the same kind of data as accepted by
    `predict_on_batch`.
    # Arguments
        sess: current tensorflow session
        learner_object: object with all basic inference funtionalities
        generator: Generator yielding batches of input samples and ground truth
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        verbose: verbosity mode, 0 or 1.
    # Returns
        Scalar loss value and accuracy on the testing set.
    # Raises
        ValueError: In case the generator yields
            data in an invalid format.
    """
    steps_done = 0
    all_losses = []
    all_accuracies = []
    inputs = {}
    outputs = {}

    if verbose == 1:
        progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(generator)

        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, gt_lab = generator_output
            elif len(generator_output) == 3:
                x, gt_lab, _ = generator_output
            else:
                raise ValueError('output of generator should be '
                                 'a tuple `(x, y, sample_weight)` '
                                 'or `(x, y)`. Found: ' +
                                 str(generator_output))
        else:
            raise ValueError('Output not valid for current evaluation')

        inputs['images'] = x
        inputs['labels'] = gt_lab
        results = learner_object.inference(inputs, sess)

        all_losses.append(results['loss'])
        all_accuracies.append(results['accuracy'])
        steps_done += 1

        progbar.update(steps_done)

    outputs['loss'] = float(np.mean(all_losses))
    outputs['accuracy'] = float(np.mean(all_accuracies))
    return outputs


def plot_regression_predictions(test_ds, pred_y, epoch=None, i=0):

    y = [np.squeeze(y_ds) for (_, y_ds) in test_ds]
    y_flat = [item for sublist in y for item in sublist]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.squeeze(y_flat))
    ax.plot(np.squeeze(pred_y), 'r')

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
