# rpg\_imu\_prior\_learning
### Improve Inertial Odometry using Deep Learning

This repository is set-up for processing of Inertial datasets with ground truth pose estimates, with the objective of training deep models for Inertial Odometry using eager tensorflow with the keras backend. The code is set up to run with tensorflow (tf) 2.0.0a and python >= 3.5 (see [setup script](./setup.py)).

The pipeline followed by this repo is mainly thought for CPU training, but it can be run on GPU architectures as well, as far as tensorflow-gpu 2.0a is used. Additionally, this repository:
1. Uses the simplified callback to Tensorboard to log and debug the training processwith tf 2.0 and the keras backend
2. Builds a basic file structure that follows some basic good practices
3. Ensures the process of model saving, stopping and resuming training using the keras checkpoint callback
4. Enables easy processing of inertial datasets for supervised training with ground truth data.

We have used this repository to work with the [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and [BlackBird](https://github.com/mit-fast/Blackbird-Dataset) datasets, but other similar ones can be easily added. Also, it includes the [rpg_vi_simulation](./catkin_ws/src/rpg_vi_simulation) ROS package for generating synthetic IMU data from ground truth position and attitude measurements. For more information about the latter, please refer to [its GitHub repo](https://github.com/tmguillem/rpg_vi_simulation)

## Table of Contents

* [Installation](#installation)
* [Project structure](#repository-structure)
* [Working with new and old datasets](#working-with-datasets)
* [Training supervision](#training-supervision)
* [Testing a model](#testing-a-model)


## Installation

### Clone the repository
```
git clone https://github.com/uzh-rpg/rpg_imu_prior_learning/
```

### Download all the submodules
```
git submodule update --init --recursive
```

To use properly this repository, we recommend setting up a python>=3.5 virtual environment, especially since at the time this readme is being written (May 2019), tensorflow 2.0 is still in alpha version. If you want to work with a native python3.x interpreter, you can skip to [dependencies installation](#install-dependencies). If you already have virtualenvwrapper installed, skip to [virtualenv creation](#make-a-new-virtualenv)


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
This will activate the virtual environment by default. 

### Install dependencies
Finally, install the python dependencies:
```
cd rpg_imu_prior_learning
python setup.py install
```

## Repository structure:
### Project tree
 * [__catkin_ws/__](./catkin_ws): *catkin workspace for [ROS](wiki.ros.org) packages*
   * [__src/bag2csv/__](./catkin_ws/src/bag2csv): *Ros package for transforming ROS bag to .csv files. Used for blackbird dataset*
 * [__common_flags.py__](./common_flags.py): *Configuration flags to run the main [training](./train.py) and [testing](./test.py) scripts. More information [here](#common_flags.py-structure)*
 * [__data/__](./data): *This folder contains everything related to the datasets.*
   * [__config/blackbird_flags.py__](./data/config/blackbird_flags.py): *Configuration flags for the blackbird dataset manager*
   * [__config/euroc_flags.py__](./data/config/euroc_flags.py): *Configuration flags for the EuRoC dataset manager*
   * [__dataset/__](): *all the datasets should be placed in this directory (gitignored by default)*
   * [__imu_dataset_generators.py__](./data/imu_dataset_generators.py): *Contains the scripts that generate different datasets for different inference tasks given the pre-processed data*
   * [__inertial_ABCs.py__](./data/inertial_ABCs.py): *Each inertial dataset manager implements these classes. They are called by the [InertialDatasetManager](./data/inertial_dataset_manager.py) for processing all the datasets.*
   * [__inertial_dataset_manager.py__](./data/inertial_dataset_manager.py): *Provides the datasets to the main [training](./train.py) and [testing](./test.py) scripts by means of the [ABC's](./data/inertial_dataset_manager.py)*
   * [__utils/__](./data/utils)
     * [__blackbird_utils.py__](./data/utils/blackbird_utils.py): *[ABC](./data/inertial_dataset_manager.py) implementations for the blackbird dataset and other utilities*
     * [__convert_bag_to_csv.sh__](./data/utils/convert_bag_to_csv.sh): *Details [here](#pre-configured-datasets)* 
     * [__data_utils.py__](./data/utils/data_utils.py): *Any other interesting utilities for dataset processing (e.g. interpolation)*
     * [__euroc_utils.py__](./data/utils/euroc_utils.py): *[ABC](./data/inertial_dataset_manager.py) implementations for the EuRoC dataset and other utilities*
 * [__figures/__](): Figures generated by the test experiments will be stored here (gitignored by default)
 * [__models/__](./models)
   * [__base_learner.py__](./models/base_learner.py): *All the core functions for model generation, training and testing*
   * [__customized_tf_funcs/__](./models/customized_tf_funcs)
     * [__custom_callbacks.py__](./models/customized_tf_funcs/custom_callbacks.py): *Customized keras callbacks*
     * [__custom_layers.py__](./models/customized_tf_funcs/custom_layers.py): *Customized keras Layers*
     * [__custom_losses.py__](./models/customized_tf_funcs/custom_losses.py): *Customized keras losses*
   * [__nets.py__](./models/nets.py): *Deep keras models*
   * [__test_experiments.py__](./models/test_experiments.py): *Script that implements several testing experiments (e.g. iterative prediction)*
 * [__results/__](): All the trained models will be kept in this directory (gitignored by default)
 * [__setup.py__](./setup.py): *Python setup script to install dependencies*
 * [__test.py__](./test.py): *Runnable test script*
 * [__train.py__](./train.py) * Runnable training script*
 * [__utils/__](./utils): *Any utilities for the training and testing scripts*
     * [__algebra.py__](./utils/algebra.py): *Algebra functions (e.g. quaternion/Lie algebra)*
     * [__directories.py__](./utils/directories.py): *Directory utilities*
     * [__models.py__](./utils/models.py): *Model utilities*
     * [__visualization.py__](./utils/visualization.py): *Visualization utilities*
     
### Common_flags.py structure
This script contains all the configurable flags to run the training and testing scripts. They can be changed from the command line as in the following example:

```
python train.py --model_name=my_model_name --max_epochs=40 --train_ds=blackbird
python test.py --model_name=my_model_name --test_ds=blackbird --test_model_number=4
```

This is the complete list of editable flags, and their purpose
 * __model_name__: Name for the deep model, both for training (model to be trained) and for testing (model to be tested). Each model is automatically appended an id value to avoid overlaps (e.g. my_model_0, my_model_5)
 * __batch_size__: Batch size in training and evaluation
 * __learning_rate__: Learning rate for adam optimizer (as configured by default)
 * __beta1__: Momentum term of adam optimizer
 * __max_epochs__: Maximum number of training epochs before stopping training
 * __patience__: Patience epochs before interrupting training (i.e. if validation loss does not improve for given value, interrupt training)
 * __checkpoint_dir__: Directory name to save checkpoints and logs.
 * __window_length__: The number of used IMU samples for the imu integration task
 * __train_ds__: Which dataset to use for training (e.g. euroc/blackbird)
 * __plot_ds__: Whether to plot the dataset during its generation
 * __resume_train__: Whether to restore a trained model for training continue training. Will use the name specified in *model_name*. 
 * __resume_train_model_number__: In case training must be resumed, which Id of the model to use
 * __summary_freq__: Frequency of logging in Tensorboard (in epochs)
 * __save_freq__: Frequency of saving the current training model (in epochs)
 * __test_ds__: Which dataset to use for testing (e.g. euroc/blackbird)
 * __test_model_number__: Which Id of the model to test")
 * __force_ds_remake__: (Mostly for debugging) The pipeline saves a local copy of the dataset with all the pre-processing operations performed on it for efficiency. In normal conditions, it is also able to tell when such pre-processing must be performed anew, but if this flag is set to True, the dataset will be re-processed regardless.

## Working with datasets

### Pre-configured datasets: 
#### [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) dataset

The EuRoC dataset, which is smaller (only 11 recorded flights) and more format friendly than the blackbird. [This link](http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip) will download one instance of the 11 flights. The repository is prepared to accept the following data structure for the EuRoC dataset:

    └── /data/
        └── dataset/
            └── EuRoC_dataset/
                ├── dataset_0/
                ├── dataset_1/
                └── ...

Where each `dataset_i` represents one of these 11 flights. Inside each of these folders, the downloaded zip must be manually decompressed (and left as it comes out), such that:

    └── dataset_i/
        └── mav_0/
            ├── imu_0/
            ├── state_groundtruth_estimate0/
            └── (the rest of the folders can be deleted, as they contain images, which are not used)

The model can only be trained on one dataset at a time, so the number of the EuRoC dataset must be specified in the [EuRoC flags file](./data/config/euroc_flags.py), by the `dataset_version` flag (e.g. dataset_version=dataset_3, to use the flight stored in folder EuRoC_dataset/dataset_3)

#### [BlackBird](https://github.com/mit-fast/Blackbird-Dataset) dataset

The blackbird dataset is somewhat less user friendly since it has many more recorded flights, and the data is only provided in rosbag format. Interested readers are referenced to the [dataset main page]((https://github.com/mit-fast/Blackbird-Dataset). To cope with this, the pipeline to work with this dataset has been automatized. 

The instance of the blackbird dataset to be used must be specified in the [blackbird flags file](./data/config/blackbird_flags.py). Such specification implies setting up three parameters:
  * `trajectory_name`: e.g. 'bentDice'
  * `yaw_type`: either 'yawForward' or 'yawConstant' 
  * `max_speed`: e.g. 2.0
  
With these three values, a request will be made to extract the flight information (provided that the combination exists) to download the inertial and ground truth data. The inertial data, however, is compressed inside a rosbag with many other topics. A [shell script](./data/utils/convert_bag_to_csv.sh) has been created, which is run automatically by the pipeline, which calls [this ROS python script](./catkin_ws/src/bag2csv), that will automatically extract the IMU topic into a .csv file. This script **assumes the native default python interpreter is 2.7 and has ROS installed.** If that's not the case the bash script should be changed, maybe to use a python 2.7 venv.

Once the .csv files have been extracted from the rosbag, the pipeline will use them to generate the dataset. Although all these processes are automatic, for the user's interest, this is the compact folder structure tree generated:

    └── .data/dataset/blackbird_dataset/
        └── bentDice/yawForward/maxSpeed2p0/
            ├── data/_slash_blackbird_slash_imu.csv
            ├── data.bag
            └── poses.csv
      
 #### Adding a new dataset
 
Adding a new dataset so it's fully compatible with the pipeline is simple. The following steps should be completed to do it:
  * Create a new python script in `./data/utils/<my_dataset>_utils.py`. See the [blackbird_utils.py](./data/utils/blackbird_utils.py) as a reference. 
  * Implement in this new script the [three ABC's](./data/inertial_ABCs.py) with the specified abstract methods. 
    * The classes `GT` and `IMU` are used to read the ground truth and IMU data respectively. For each one, the method `read()` must be implemented
    * `InertialDataset` is the third abstract class to be implented, which provides and processes each dataset according to its specific needs. In fact, two methods from this class must be completed:
      * `get_raw_ds()`: which returns the IMU and ground truth data using the GT and IMU based classes. This method might get as complicated as the user wants. For instance, for the blackbird dataset, it performs the http request to download the data, decompresses the rosbags into csv files and constructs the data. For EuRoC, it assumes that the files are already downloaded. 
      * `pre_process_data()`: which does any kind of pre-processing needed. There is one basic pre-processing function in the `super` class called `basic_pre_processing()` which performs a low-pass filtering that can be used if needed.
      * Assign the values to the class variables `ds_local_dir` and `sampling_freq`, which contain the location of the dataset within the repository (e.g. `./data/dataset/EuRoC/`), and the sampling frequency of the IMU used (e.g. 200 [Hz])
    * Add the new implementation of `InertialDataset` to the [DatasetManager](./data/inertial_dataset_manager.py) class (see lines 39-44
        
If all these classes are methods are completed properly, the dataset should be fully embedded in the pipeline with any further effort.

## Training Supervision

As commented previously, this repo uses the simplified keras callback function for Tensorboard to log training summaries. These can be visualized using the command line:

```
tensorboard -logdir=/path_to_repo/results/my_model_name_Id/
```

Or equivalently if the command is not exported to your current shell by:
```
python3 /usr/local/lib/python3.x/dist-packages/tensorboard/main.py -logdir=/path_to_repo/results/my_model_name_Id/
```

## Testing a model

To test models on different datasets, there is a separate script which can be run: [test.py](./test.py). In it, the user should define the experiments that he wants to run. The experiment definition is structured in a nested dictionary structure. As an example:
```
"plot_predictions": {
    "ds_testing_non_tensorflow": ["predict"],
    "ds_testing_non_tensorflow_unnormalized": ["ground_truth"],
    "options": {
        "output": "show",
        "append_save_name": "my_first_experiment",
        "plot_data": {
            "state_output": {
                "type": "10-dof-state",
                "dynamic_plot": True,
                "sparsing_factor": 2,
            },
            "pre_integrated_p": {
                "type": "pre_integration"
            },
            "pre_integrated_v": {
                "type": "pre_integration"
            },
            "pre_integrated_R": {
                "type": "pre_integration"
            }
        }
    }
}
```

The outtermost dictionary should only have a key, which defines the type of experiment that is going to be run. In this case `plot_predictions` is an experiment that consists on taking the most recent version of a model, and plotting its predictions in comparison with other values. Inside this dictionary, the user should specify which data is going to be compared in the experiment, and how to process it. In this case, we want to compare the model predictions on the testing dataset and the ground truth. Additionally, we want the ground truth to display the actual unnormalized data, as the model might require some pre-procecssing to generate the predictions. As such, the first two elements in the experiment dictionary are going to be:

```
# The `non_tensorflow` flag isrequired for the experiments. It means that the dataset 
# will be un numpy format, rather than a tensorflow dataset.
# The order of the flags is not important
"ds_testing_non_tensorflow": ["predict"],
"ds_testing_non_tensorflow_unnormalized": ["ground_truth"],
```        
Finally, an `options` dictionary should be added, to include other information about the experiment. 
  - The key `output` can be either `show` or `save`, to display the generated images or store them in the temporal `./figures/` directory.
  - The names of the generated figures are automatic, but users can add some specific name with the key `append_save_name` 
  - The key `plot_data` specifies how to plot each output of the model. For now, two types are supported: `10-dof-state` and `pre-integration`, and for the first sort, a `dynamic_plot` can be added with the simulation of the trajectory. The `sparsing_factor` refers to the simulation speed increase factor.
