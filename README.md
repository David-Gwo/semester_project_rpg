# rpg\_imu\_prior\_learning
### Improve Inertial Odometry using Deep Learning

This repository is set-up for processing of Inertial datasets with ground truth pose estimates, with the objective of training deep models for Inertial Odometry using eager tensorflow with the keras backend. The code is set up to run with tensorflow (tf) 2.0.0a and python >= 3.5 (see [setup script](./setup.py)).

The pipeline followed by this repo is mainly optimized for CPU training (as the Inertial Odometry problem deals with low-dimensional data and models can be trained within minutes or a few hours), but it can be run on GPU architectures as well. In summary, this repository:
1. Uses the simplified callback to Tensorboard to log and debug the training processwith tf 2.0 and the keras backend
2. Builds a basic file structure that follows some basic good practices
3. Ensures the process of model saving, stopping and resuming training using the keras checkpoint callback
4. Enables easy processing of inertial datasets for supervised training, provided they have some format of ground truth data

We have used this repository to work with the [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and [BlackBird](https://github.com/mit-fast/Blackbird-Dataset) datasets, but other similar ones can be easily added. 

## Table of Contents

* [Installation](#installation)
* [Project structure](#repository-structure)


## Installation

To use properly this repository, we recommend setting up a python>=3.5 virtual environment, especially since at the time this readme is being written (May 2019), tensorflow 2.0 is still in alpha version. If you want to work with a native python3.x interpreter, you can skip to . If you already have virtualenvwrapper installed, skip to [virtualenv creation](#make-a-new-virtualenv)

### Clone the repository
```
git clone https://github.com/uzh-rpg/rpg_imu_prior_learning/
```

### Install virtualenvwrapper for python 3
```
# Install virtualenvwrapper with pip
sudo apt-get install python3-pip
sudo pip3 install virtualenvwrapper

# Create a folder for saving the virtual environments 
mkdir ~/.virtualenvs
export WORKON_HOME=~/.virtualenvs
```
Add the virtualenvwrapper commands to your .bashrc (`vi ~/.bashrc`)
```
VIRTUALENVWRAPPER_PYTHON='/usr/bin/python3' # This line needs to be before the next
source /usr/local/bin/virtualenvwrapper.sh
```
Reopen the shell terminal to continue to next section.

### Make a new virtualenv 
To create a new virtual environment, run:
```
mkvirtualenv tensorflow_2.0_venv # Use any name you prefer for the venv
```
This will activate the virtual environment by default. Then, install the python dependencies:
```
cd rpg_imu_prior_learning
python setup.py install
```

## Repository structure:
### Project tree
.
  * [catkin_ws](./catkin_ws)
     * [src](./catkin_ws/src)
       *  [bag2csv](./catkin_ws/src/bag2csv)
 * [common_flags.py](./common_flags.py)
 * [data](./data)
   * [config](./data/config)
     * [blackbird_flags.py](./data/config/blackbird_flags.py)
     * [euroc_flags.py](./data/config/euroc_flags.py)
   * [imu_dataset_generators.py](./data/imu_dataset_generators.py)
   * [inertial_ABCs.py](./data/inertial_ABCs.py)
   * [inertial_dataset_manager.py](./data/inertial_dataset_manager.py)
   * [utils](./data/utils)
     * [blackbird_utils.py](./data/utils/blackbird_utils.py)
     * [convert_bag_to_csv.sh](./data/utils/convert_bag_to_csv.sh)
     * [data_utils.py](./data/utils/data_utils.py)
     * [euroc_utils.py](./data/utils/euroc_utils.py)
 * [models](./models)
   * [base_learner.py](./models/base_learner.py)
   * [customized_tf_funcs](./models/customized_tf_funcs)
     * [custom_callbacks.py](./models/customized_tf_funcs/custom_callbacks.py)
     * [custom_layers.py](./models/customized_tf_funcs/custom_layers.py)
     * [custom_losses.py](./models/customized_tf_funcs/custom_losses.py)
   * [nets.py](./models/nets.py)
   * [test_experiments.py](./models/test_experiments.py)
 * [setup.py](./setup.py)
 * [test.py](./test.py)
 * [train.py](./train.py)
 * [utils](./utils)
     * [algebra.py](./utils/algebra.py)
     * [directories.py](./utils/directories.py)
     * [models.py](./utils/models.py)
     * [visualization.py](./utils/visualization.py)

## What should I do after cloning the repo?

The code in this repository is written for a simple 5 class classification problem. There is already a small [sample dataset](./data/sample_data) to
test if everything is working properly. In case you are dealing with a supervised learning problem (classification / regression),
it is a good idea to imitate the folder structure of the sample dataset to build your own dataset.
Indeed, minimal changes to the code will be required to build your model.

### Getting started

To start training the sample classification problem with the provided dataset, use the following command:

```
python3 train.py --train_dir=./data/sample_data/training --val_dir=./data/sample_data/testing --summary_freq=20 --checkpoint_dir=./tests/test_0 --save_freq=5
```

After a few epochs, try to stop it. Later, you can continue training from your last checkpoint with this command:

```
python3 train.py --train_dir=./data/sample_data/training --summary_freq=20 --checkpoint_dir=./tests/test_0 --resume_train=True --val_dir=./data/sample_data/testing
```

To check over your training process, you can use [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard). To do it, use the following:

```
tensorboard --logdir=./tests/test_0
```

```
python test.py --test_dir=./data/sample_data/testing --ckpt_file=./tests/test_0/model-#
```
