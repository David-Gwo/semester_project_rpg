import os
import sys
import math

from tensorflow.python.keras import callbacks

from utils.directories import get_checkpoint_file_list, safe_mkdir_recursive
from data.inertial_dataset_manager import DatasetManager
from models.nets import *
from models.customized_tf_funcs.custom_callbacks import CustomModelCheckpoint
from models.customized_tf_funcs.custom_losses import *
from experiments.test_experiments import ExperimentManager

#############################################################################
# IMPORT HERE A LIBRARY TO PRODUCE ALL THE FILENAMES (and optionally labels)#
# OF YOUR DATASET. HAVE A LOOK AT `DirectoryIterator' FOR AN EXAMPLE        #
#############################################################################
sys.path.append("../")


class Learner(object):
    def __init__(self, config):
        self.config = config
        self.trainable_model = None
        self.model_name = None
        self.model_version_number = None
        self.last_epoch_number = 0
        self.trained_model_dir = ""
        self.experiment_manager = None

        self.valid_model_types = [
            "speed_regression_net", "integration_net", "integration_so3_net", "preintegration_net"]

        if self.config.model_type not in self.valid_model_types:
            raise ValueError("This type of the model is not one of the valid ones: %s" % self.valid_model_types)

    def build_and_compile_model(self, is_testing=False):

        trainable_model = None
        loss_connections = {}
        loss_weight = {}

        if self.config.model_type == "speed_regression_net":
            trainable_model = vel_cnn(self.config.window_length)
            loss_connections = {"state_output": l1_loss}
            loss_weight = {"state_output": 1.0}
        elif self.config.model_type == "integration_net":
            trainable_model = imu_integration_net(self.config.window_length, 10)
            loss_connections = {"state_output": state_loss}
            loss_weight = {"state_output": 1.0}
        elif self.config.model_type == "integration_so3_net":
            trainable_model = imu_integration_net(self.config.window_length, 9)
            loss_connections = {"state_output": 'mse'}
            loss_weight = {"state_output": 1.0}
        elif self.config.model_type == "preintegration_net":
            trainable_model = cnn_rnn_pre_int_net(self.config.window_length, 2)
            loss_connections = {"pre_integrated_R": 'mse',
                                "pre_integrated_v": 'mse',
                                "pre_integrated_p": 'mse',
                                "state_output": state_loss}
            loss_weight = {"pre_integrated_R": 1.0,
                           "pre_integrated_v": 1.0,
                           "pre_integrated_p": 1.0,
                           "state_output": 0.2}

        print(trainable_model.summary())

        if not is_testing:
            trainable_model.compile(optimizer=tf.keras.optimizers.Adam(self.config.learning_rate, self.config.beta1),
                                    loss=loss_connections,
                                    loss_weight=loss_weight)
        else:
            trainable_model.compile(optimizer=tf.keras.optimizers.Adam(self.config.learning_rate, self.config.beta1),
                                    loss=loss_connections)

        self.trainable_model = trainable_model

    def get_dataset(self, train, val_split, shuffle, random_split=True, const_batch_size=False, normalize=True,
                    repeat_ds=False, tensorflow_format=True):

        force_remake = self.config.force_ds_remake

        dataset_name = self.config.dataset

        dataset_manager = DatasetManager(prepared_train_data_file='imu_dataset_train.mat',
                                         prepared_test_data_file='imu_dataset_test.mat',
                                         trained_model_dir=self.trained_model_dir,
                                         dataset_name=dataset_name)

        return dataset_manager.get_dataset(self.config.dataset_type,
                                           self.config.window_length,
                                           batch_size=self.config.batch_size,
                                           validation_split=val_split,
                                           random_split=random_split,
                                           train=train,
                                           plot=self.config.plot_ds,
                                           shuffle=shuffle,
                                           full_batches=const_batch_size,
                                           normalize=normalize,
                                           repeat_ds=repeat_ds,
                                           force_remake=force_remake,
                                           tensorflow_format=tensorflow_format)

    def train(self):
        self.build_and_compile_model()

        # Identify last version of trained model
        files = get_checkpoint_file_list(self.config.checkpoint_dir, self.config.model_name)

        if not files:
            model_number = self.config.model_name + "_0"
        else:
            # Resume training vs new training decision
            if self.config.resume_train:
                print("Resume training from previous checkpoint")
                try:
                    self.recover_model_from_checkpoint()
                    model_number = self.model_version_number
                except FileNotFoundError:
                    print("Model not found. Creating new model")
                    model_number = self.model_version_number
                    safe_mkdir_recursive(self.config.checkpoint_dir + model_number)
            else:
                model_number = self.config.model_name + '_' + str(int(files[-1].split('_')[-1]) + 1)
                os.mkdir(self.config.checkpoint_dir + model_number)

        self.trained_model_dir = self.config.checkpoint_dir + model_number + '/'

        # Get training and validation datasets from saved files
        dataset = self.get_dataset(train=True, val_split=True, random_split=False, shuffle=True, repeat_ds=True,
                                   normalize=True)
        train_ds, validation_ds, ds_lengths = dataset

        train_steps_per_epoch = int(math.ceil(ds_lengths[0]/self.config.batch_size))
        val_steps_per_epoch = int(math.ceil((ds_lengths[1]/self.config.batch_size)))

        def lr_scheduler(epoch, lr):
            decay_rate = 0.5
            if epoch % self.config.lr_scheduler == 0 and epoch:
                return lr * decay_rate
            return lr

        keras_callbacks = [
            callbacks.EarlyStopping(patience=self.config.patience, monitor='val_loss'),
            callbacks.TensorBoard(
                write_images=True,
                log_dir=self.config.checkpoint_dir + model_number + "/keras",
                histogram_freq=self.config.summary_freq),
            callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
            CustomModelCheckpoint(
                filepath=os.path.join(
                    self.config.checkpoint_dir + model_number, self.config.model_name + "_{epoch:02d}.h5"),
                save_weights_only=True,
                verbose=1,
                period=self.config.save_freq,
                extra_epoch_number=self.last_epoch_number + 1),
        ]

        # Train!
        self.trainable_model.fit(
            train_ds,
            verbose=2,
            epochs=self.config.max_epochs,
            steps_per_epoch=train_steps_per_epoch,
            validation_steps=val_steps_per_epoch,
            validation_data=validation_ds,
            callbacks=keras_callbacks)

    def recover_model_from_checkpoint(self, model_used_pos=-1):
        """
        Loads the weights of the default model from the checkpoint files
        """

        model_number = self.config.model_number

        # Directory from where to load the saved weights of the model
        self.model_version_number = self.config.model_name + "_" + str(model_number)
        recovered_model_dir = self.config.checkpoint_dir + self.model_version_number

        files = get_checkpoint_file_list(recovered_model_dir, self.config.model_name)
        if not files:
            raise FileNotFoundError()

        model_version_used = files[model_used_pos]

        tf.print("Loading weights from ", recovered_model_dir + '/' + model_version_used)
        self.trainable_model.load_weights(recovered_model_dir + '/' + model_version_used)

        # Get last epoch of training of the model
        self.last_epoch_number = int(model_version_used.split(self.config.model_name)[1].split('.')[0].split('_')[1])

        if model_version_used == files[-1]:
            return -1
        else:
            return model_used_pos + 1

    def test(self, experiments):
        self.build_and_compile_model(is_testing=True)
        self.experiment_manager = ExperimentManager(window_len=self.config.window_length,
                                                    final_epoch=self.last_epoch_number,
                                                    model_loader_func=self.experiment_model_request,
                                                    dataset_loader_func=self.experiment_dataset_request)

        self.recover_model_from_checkpoint()
        self.trained_model_dir = self.config.checkpoint_dir + self.model_version_number + '/'

        for experiment in experiments.keys():
            self.experiment_manager.run_experiment(experiment, experiments[experiment])

    def experiment_model_request(self, requested_model_num=None):

        model_pos = -1

        if requested_model_num is None:
            self.recover_model_from_checkpoint(model_used_pos=model_pos)
            return self.trainable_model
        else:
            new_model_num = self.recover_model_from_checkpoint(model_used_pos=requested_model_num)
            return self.trainable_model, new_model_num

    def experiment_dataset_request(self, dataset_tags):
        train = False
        val_split = False
        const_batch_size = False
        shuffle = False
        normalize = True
        repeat_ds = False
        random_split = False
        tensorflow_format = True

        if 'training' in dataset_tags:
            train = True
        if 'unnormalized' in dataset_tags:
            normalize = False
        if 'non_tensorflow' in dataset_tags:
            tensorflow_format = False

        dataset = self.get_dataset(train=train,
                                   val_split=val_split,
                                   const_batch_size=const_batch_size,
                                   shuffle=shuffle,
                                   random_split=random_split,
                                   normalize=normalize,
                                   repeat_ds=repeat_ds,
                                   tensorflow_format=tensorflow_format)

        if val_split:
            training, validation, _ = dataset
            return training, validation
        else:
            training, _ = dataset
            return training
