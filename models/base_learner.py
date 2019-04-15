import os
import sys
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.optimizers import Adam
from .nets import imu_integration_net as prediction_network
from utils import get_checkpoint_file_list, imu_integration
from data.utils.data_utils import get_mnist_datasets, safe_mkdir_recursive, DirectoryIterator, \
    plot_regression_predictions
from data.euroc_manager import load_euroc_dataset, generate_tf_imu_test_ds
from data.blackbird_manager import load_blackbird_dataset, BlackbirdDSManager
from models.custom_callback_fx import CustomModelCheckpoint

#############################################################################
# IMPORT HERE A LIBRARY TO PRODUCE ALL THE FILENAMES (and optionally labels)#
# OF YOUR DATASET. HAVE A LOOK AT `DirectoryIterator' FOR AN EXAMPLE        #
#############################################################################
sys.path.append("../")


class Learner(object):
    def __init__(self, config):
        self.config = config
        self.regressor_model = None
        self.model_name = None
        self.model_version_number = None
        self.last_epoch_number = 0
        self.trained_model_dir = ""
        pass

    def preprocess_image(self, image):
        ##############################
        # DO YOUR PREPROCESSING HERE #
        ##############################
        """ Preprocess an input image.
        Args:
            image: A uint8 tensor
        Returns:
            image: A preprocessed float32 tensor.
        """
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, [self.config.img_height, self.config.img_width])
        image = tf.cast(image, dtype=tf.float32)
        image = tf.divide(image, 255.0)
        return image

    def read_image_from_dir(self, image_dir):
        image = tf.io.read_file(image_dir)
        return self.preprocess_image(image)

    def read_from_disk(self, inputs_queue):
        """Consumes the inputs queue.
        Args:
            inputs_queue: A scalar string tensor.
        Returns:
            Two tensors: the decoded images, and the labels.
        """
        pnt_seq = tf.strings.to_number(inputs_queue[1], out_type=tf.dtypes.int32)
        file_content = inputs_queue[0].map(self.read_image_from_dir)
        image_seq = tf.image.decode_png(file_content, channels=3)

        return image_seq, pnt_seq

    @staticmethod
    def get_filenames_list(directory):
        """ This function should return all the filenames of the
            files you want to train on.
            In case of classification, it should also return labels.

            Args:
                directory: dataset directory
            Returns:
                List of filenames, [List of associated labels]
        """
        iterator = DirectoryIterator(directory, shuffle=False)
        return iterator.filenames, iterator.ground_truth

    @staticmethod
    def l2_loss(y_true, y_pred):
        return tf.abs(tf.math.subtract(tf.cast(y_true, tf.float32), y_pred))

    def custom_backprop(self, training_ds, validation_ds, ds_lengths, epoch):

        optimizer = tf.keras.optimizers.Adam(self.config.learning_rate, self.config.beta1)

        for i, (x, y) in enumerate(training_ds):

            if i % 100 == 0:
                self.last_epoch_number = epoch
                self.evaluate_model(validation_ds, ds_lengths[1], save_figures=True)

            with tf.GradientTape() as tape:
                # Forward pass
                logit = self.regressor_model(tf.cast(x, tf.float32))

                # External loss calculation
                loss = self.l2_loss(y, logit)

                # Manual loss combination:
                loss += sum(self.regressor_model.losses)

            if i % 10 == 0:
                tf.print("Batch {0} of {1}".format(i, ds_lengths[0]))
                tf.print("Training loss of batch {0}/{2} is: {1}".format(i, loss, ds_lengths[0]))

            # Get gradients
            gradient = tape.gradient(loss, self.regressor_model.trainable_weights)

            # Update weights of layer
            optimizer.apply_gradients(zip(gradient, self.regressor_model.trainable_weights))

    def build_and_compile_model(self):
        # model = prediction_network(input_shape=(self.config.img_height, self.config.img_width, 1),
        #                            l2_reg_scale=self.config.l2_reg_scale,
        #                            output_dim=self.config.output_dim)

        model = prediction_network(self.config.window_length, 10)

        print(model.summary())
        with tf.name_scope("compile_model"):
            model.compile(optimizer=Adam(self.config.learning_rate, self.config.beta1),
                          loss=self.l2_loss,
                          metrics=['mse'])
        self.regressor_model = model

    def get_dataset(self):

        dataset_name = self.config.train_ds

        if dataset_name == 'mnist':
            return get_mnist_datasets(self.config.img_height, self.config.img_width, self.config.batch_size)
        if dataset_name == 'euroc':
            return load_euroc_dataset(self.config.train_dir, self.config.batch_size, self.config.window_length,
                                      self.config.prepared_train_data_file, self.config.prepared_test_data_file,
                                      self.config.prepared_file_available, self.trained_model_dir)
        if dataset_name == 'blackbird':
            return load_blackbird_dataset(self.config.batch_size, self.config.window_length,
                                          self.config.prepared_train_data_file, self.config.prepared_test_data_file,
                                          self.config.prepared_file_available, self.trained_model_dir)

    def collect_summaries(self):
        """Collects all summaries to be shown in the tensorboard"""

        #######################################################
        # ADD HERE THE VARIABLES YOU WANT TO SEE IN THE BOARD #
        #######################################################
        tf_summary.scalar("train_loss", self.total_loss, collections=["step_sum"])
        tf_summary.scalar("accuracy", self.accuracy, collections=["step_sum"])
        tf_summary.histogram("logits_distribution", self.logits, collections=["step_sum"])
        tf_summary.histogram("predicted_out_distributions", tf.argmax(self.logits, 1), collections=["step_sum"])
        tf_summary.histogram("ground_truth_distribution", self.labels, collections=["step_sum"])

        ###################################################
        # LEAVE UNCHANGED (gradients and tensors summary) #
        ###################################################
        for var in tf.compat.v1.trainable_variables():
            tf_summary.histogram(var.op.name + "/values", var, collections=["step_sum"])
        for grad, var in self.grads_and_vars:
            tf_summary.histogram(var.op.name + "/gradients", grad, collections=["step_sum"])
        self.step_sum = tf.summary.merge(tf.compat.v1.get_collection('step_sum'))

        ####################
        # VALIDATION ERROR #
        ####################
        self.validation_loss = tf.compat.v1.placeholder(tf.float32, [])
        self.validation_accuracy = tf.compat.v1.placeholder(tf.float32, [])
        tf_summary.scalar("Validation_Loss", self.validation_loss, collections=["validation_summary"])
        tf_summary.scalar("Validation_Accuracy", self.validation_accuracy, collections=["validation_summary"])
        self.val_sum = tf_summary.merge(tf.compat.v1.get_collection('validation_summary'))

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to {}/model-{}".format(checkpoint_dir, step))
        if step == 'best':
            self.saver.save(sess, os.path.join(checkpoint_dir, model_name + '.best'))
        else:
            self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

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

        self.trained_model_dir = self.config.checkpoint_dir + model_number

        # Get training and validation datasets
        train_ds, validation_ds, ds_lengths = self.get_dataset()

        val_ds_splits = np.diff(np.linspace(0, ds_lengths[1], 2)/self.config.batch_size).astype(np.int)
        val_ds = {}

        for i, split in enumerate(val_ds_splits):
            val_ds[i] = validation_ds.take(split)
            validation_ds = validation_ds.skip(split)

        train_steps_per_epoch = int(math.ceil(ds_lengths[0]/self.config.batch_size))
        val_steps_per_epoch = int(math.ceil(val_ds_splits[0]))

        keras_callbacks = [
            callbacks.EarlyStopping(patience=20, monitor='val_loss'),
            callbacks.TensorBoard(log_dir=self.config.checkpoint_dir + model_number + "/keras", histogram_freq=5),
            CustomModelCheckpoint(
                filepath=os.path.join(
                    self.config.checkpoint_dir + model_number, self.config.model_name + "_{epoch:02d}.h5"),
                save_weights_only=True, verbose=1, period=self.config.save_freq,
                extra_epoch_number=self.last_epoch_number + 1),
        ]

        # for epoch in range(self.config.max_epochs):
        #     self.custom_backprop(val_ds[0], val_ds[0], (val_ds_splits[0], val_ds_splits[0]), epoch)

        # Train!
        history = self.regressor_model.fit(
            train_ds,
            verbose=2,
            epochs=self.config.max_epochs,
            steps_per_epoch=train_steps_per_epoch,
            validation_data=val_ds[0],
            validation_steps=val_steps_per_epoch,
            callbacks=keras_callbacks)

        # Evaluate model in validation set and entire training set
        self.evaluate_model(val_ds[0], val_ds_splits[0])
        self.evaluate_model(train_ds, ds_lengths[0]/self.config.batch_size)

    def recover_model_from_checkpoint(self, mode="train", model_used_pos=-1):
        """
        Loads the weights of the default model from the checkpoint files
        """

        if mode == "train":
            model_number = self.config.resume_train_model_number
        else:
            model_number = self.config.test_model_number

        # Directory from where to load the saved weights of the model
        self.model_version_number = self.config.model_name + "_" + str(model_number)
        recovered_model_dir = self.config.checkpoint_dir + self.model_version_number

        files = get_checkpoint_file_list(recovered_model_dir, self.config.model_name)
        if not files:
            raise FileNotFoundError()

        model_version_used = files[model_used_pos]

        tf.print("Loading weights from ", recovered_model_dir + '/' + model_version_used)
        self.regressor_model.load_weights(recovered_model_dir + '/' + model_version_used)

        # Get last epoch of training of the model
        self.last_epoch_number = int(model_version_used.split(self.config.model_name)[1].split('.')[0].split('_')[1])

        if model_version_used == files[-1]:
            return -1
        else:
            return model_used_pos+1

    def evaluate_model(self, testing_ds=None, steps=None, save_figures=False, fig_n=0, compare_manual=False):

        dataset = self.config.prepared_test_data_file
        if self.config.generate_training_progression:
            dataset = self.config.prepared_train_data_file

        dataset = self.config.prepared_train_data_file
        if testing_ds is None:
            # TODO: make more elegant
            if self.config.test_ds == 'blackbird':
                bb_manager = BlackbirdDSManager()
                test_dir = bb_manager.ds_local_dir
            else:
                test_dir = self.config.test_dir

            test_ds, steps = generate_tf_imu_test_ds(test_dir,
                                                     dataset,
                                                     self.config.batch_size,
                                                     self.config.checkpoint_dir + self.model_version_number,
                                                     self.config.window_length)
            steps = steps / self.config.batch_size
        else:
            test_ds = testing_ds.take(steps)

        predictions = self.regressor_model.predict(test_ds, verbose=1, steps=steps)

        if compare_manual:
            # TODO: idem
            if self.config.test_ds == 'blackbird':
                bb_manager = BlackbirdDSManager()
                test_dir = bb_manager.ds_local_dir
            else:
                test_dir = self.config.test_dir
            test_ds, steps = generate_tf_imu_test_ds(test_dir,
                                                     dataset,
                                                     self.config.batch_size,
                                                     self.config.checkpoint_dir + self.model_version_number,
                                                     self.config.window_length,
                                                     normalize=False)
            manual_predictions = imu_integration(test_ds, self.config.window_length)
        else:
            manual_predictions = None

        if save_figures:
            plot_regression_predictions(test_ds, predictions, self.last_epoch_number, fig_n)
        else:
            plot_regression_predictions(test_ds, predictions, manual_pred=manual_predictions)

    def epoch_end_callback(self, sess, sv, epoch_num):
        # Evaluate val accuracy
        val_loss = 0
        val_accuracy = 0
        for i in range(self.val_steps_per_epoch):
            loss, accuracy = sess.run([self.total_loss, self.accuracy],
                             feed_dict={self.is_training: False})
            val_loss+= loss
            val_accuracy += accuracy
        val_loss = val_loss / self.val_steps_per_epoch
        val_accuracy = val_accuracy / self.val_steps_per_epoch
        # Log to Tensorflow board
        val_sum = sess.run(self.val_sum, feed_dict ={
            self.validation_loss: val_loss,
            self.validation_accuracy: val_accuracy})
        sv.summary_writer.add_summary(val_sum, epoch_num)
        print("Epoch [{}] Validation Loss: {} Validation Accuracy: {}".format(
            epoch_num, val_loss, val_accuracy))
        # Model Saving
        if val_loss < self.min_val_loss:
            self.save(sess, self.config.checkpoint_dir, 'best')
            self.min_val_loss = val_loss
        if epoch_num % self.config.save_freq == 0:
            self.save(sess, self.config.checkpoint_dir, epoch_num)
