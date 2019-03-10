import tensorflow as tf
import numpy as np
import logging

from cnn_model import cnn_model_fn

import tensorflow.python.estimator.estimator as estimator
import tensorflow.python.estimator.training as estimator_training
from tensorflow.contrib.training.python.training.hparam import HParams

logging.getLogger().setLevel(logging.INFO)

# Load training and eval data
((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required

params = HParams(learning_rate=0.001)
run_config = estimator.run_config.RunConfig(
    model_dir='/home/guillem/Documents/NN/mnist_example',
    save_summary_steps=100,
)

estimator = estimator.Estimator(
    model_fn=cnn_model_fn,
    params=params,
    config=run_config,
)


def train_input_fn():  # returns x, y

    dataset = tf.data.Dataset()

    return dataset


def eval_input_fn():  # returns x, y
    pass


train_spec = estimator_training.TrainSpec(input_fn=train_input_fn, max_steps=1000)
eval_spec = estimator_training.EvalSpec(input_fn=eval_input_fn)

estimator_training.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)
