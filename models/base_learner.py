import os
import sys
import time
from itertools import count
import math
import random
import tensorflow as tf
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.keras.optimizers import Adam

###########################################################
# IMPORT YOUR FAVORITE NETWORK HERE (in place of resnet8) #
###########################################################

from .nets import mnist_cnn as prediction_network
from data import DirectoryIterator
from data.data_utils import get_mnist_datasets

#############################################################################
# IMPORT HERE A LIBRARY TO PRODUCE ALL THE FILENAMES (and optionally labels)#
# OF YOUR DATASET. HAVE A LOOK AT `DirectoryIterator' FOR AN EXAMPLE        #
#############################################################################
sys.path.append("../")


class Learner(object):
    def __init__(self):
        self.config = None
        self.classifier_model = None
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

    def generate_batches(self, data_dir):
        seed = random.randint(0, 2**31 - 1)

        # Load the list of training files into queues
        file_list, pnt_list = self.get_filenames_list(data_dir)

        # Create a dataset from all the image paths, read and process them
        path_ds = tf.data.Dataset.from_tensor_slices(file_list)
        image_ds = path_ds.map(self.read_image_from_dir, num_parallel_calls=self.config.num_threads)

        # Create a dataset from all the image labels
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(pnt_list, tf.int64))

        # Combine the image and label datasets into a zipped dataset
        inputs_queue = tf.data.Dataset.zip((image_ds, label_ds))
        inputs_queue = inputs_queue.shuffle(tf.shape(file_list, out_type=tf.int64)[0], seed=seed).repeat()
        inputs_queue = inputs_queue.batch(batch_size=self.config.batch_size, drop_remainder=False)
        inputs_queue = inputs_queue.prefetch(buffer_size=self.config.batch_size)

        # # Form training batches
        # image_batch, pnt_batch = tf.train.batch([image_seq,
        #      pnt_seq],
        #      batch_size=self.config.batch_size,
        #      num_threads=self.config.num_threads,
        #      capacity=self.config.capacity_queue,
        #      allow_smaller_final_batch=True)
        # return [image_batch, pnt_batch], len(file_list)
        return inputs_queue, len(file_list)

    def build_and_compile_model(self):
        # model = prediction_network(input_shape=(self.config.img_height, self.config.img_width, 1),
        #                            l2_reg_scale=self.config.l2_reg_scale,
        #                            output_dim=self.config.output_dim)

        model = prediction_network()

        print(model.summary())
        with tf.name_scope("compile_model"):
            model.compile(optimizer=Adam(self.config.learning_rate, self.config.beta1),
                          loss=CategoricalCrossentropy(),
                          metrics=['accuracy'])
        return model

    def get_datasets(self):
        with tf.name_scope("data_loading"):

            # In case of classification, this should be unchanged. Otherwise, adapt to load your inputs
            train_ds, n_samples_train = self.generate_batches(self.config.train_dir)
            val_ds, n_samples_val = self.generate_batches(self.config.val_dir)

        return train_ds, val_ds, (n_samples_train, n_samples_val)

    def collect_summaries(self):
        """Collects all summaries to be shown in the tensorboard"""

        #######################################################
        # ADD HERE THE VARIABLES YOU WANT TO SEE IN THE BOARD #
        #######################################################
        tf.compat.v1.summary.scalar("train_loss", self.total_loss, collections=["step_sum"])
        tf.compat.v1.summary.scalar("accuracy", self.accuracy, collections=["step_sum"])
        tf.compat.v1.summary.histogram("logits_distribution", self.logits, collections=["step_sum"])
        tf.compat.v1.summary.histogram("predicted_out_distributions", tf.argmax(self.logits, 1), collections=["step_sum"])
        tf.compat.v1.summary.histogram("ground_truth_distribution", self.labels, collections=["step_sum"])

        ###################################################
        # LEAVE UNCHANGED (gradients and tensors summary) #
        ###################################################
        for var in tf.compat.v1.trainable_variables():
            tf.compat.v1.summary.histogram(var.op.name + "/values", var, collections=["step_sum"])
        for grad, var in self.grads_and_vars:
            tf.compat.v1.summary.histogram(var.op.name + "/gradients", grad, collections=["step_sum"])
        self.step_sum = tf.compat.v1.summary.merge(tf.compat.v1.get_collection('step_sum'))

        ####################
        # VALIDATION ERROR #
        ####################
        self.validation_loss = tf.compat.v1.placeholder(tf.float32, [])
        self.validation_accuracy = tf.compat.v1.placeholder(tf.float32, [])
        tf.compat.v1.summary.scalar("Validation_Loss", self.validation_loss, collections=["validation_summary"])
        tf.compat.v1.summary.scalar("Validation_Accuracy", self.validation_accuracy, collections=["validation_summary"])
        self.val_sum = tf.compat.v1.summary.merge(tf.compat.v1.get_collection('validation_summary'))

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to {}/model-{}".format(checkpoint_dir, step))
        if step == 'best':
            self.saver.save(sess, os.path.join(checkpoint_dir, model_name + '.best'))
        else:
            self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def train(self, config):
        """High level train function.
        Args:
            config: Configuration dictionary
        Returns:
            None
        TODO: Add progbar from keras
        """

        self.config = config
        self.classifier_model = self.build_and_compile_model()
        # train_ds, validation_ds, ds_lengths = self.get_datasets()
        train_ds, validation_ds, test_ds, ds_lengths = get_mnist_datasets(
            self.config.img_height, self.config.img_width, self.config.batch_size)

        train_steps_per_epoch = int(math.ceil(ds_lengths[0]/self.config.batch_size))
        val_steps_per_epoch = int(math.ceil(ds_lengths[1]/self.config.batch_size))

        keras_callbacks = [
            # Interrupt training if `val_loss` stops improving for over 2 epochs
            callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            # Write TensorBoard logs to `./logs` directory
            callbacks.TensorBoard(log_dir=config.checkpoint_dir + "/keras"),
            # Model checkpoint and saver
            callbacks.ModelCheckpoint(filepath=os.path.join(config.checkpoint_dir, "cnn_net{epoch:02d}.h5"),
                                      save_weights_only=True, verbose=1)
        ]

        self.classifier_model.fit(
            train_ds,
            verbose=1,
            epochs=self.config.max_epochs,
            steps_per_epoch=train_steps_per_epoch,
            validation_data=validation_ds,
            validation_steps=val_steps_per_epoch,
            callbacks=keras_callbacks)

        # TODO: implement resume training?

        # self.build_train_graph()
        # self.collect_summaries()
        # self.min_val_loss = math.inf  # Initialize to max value
        # with tf.name_scope("parameter_count"):
        #     parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.compat.v1.trainable_variables()])
        # self.saver = tf.compat.v1.train.Saver([var for var in tf.compat.v1.trainable_variables()] + [self.global_step], max_to_keep=5)
        # sv = tf.train.Supervisor(logdir=config.checkpoint_dir, save_summaries_secs=0, saver=None)
        # with sv.managed_session() as sess:
        #     print("Number of trainable params: {}".format(sess.run(parameter_count)))
        #     if config.resume_train:
        #         print("Resume training from previous checkpoint")
        #         checkpoint = tf.train.latest_checkpoint(config.checkpoint_dir)
        #         self.saver.restore(sess, checkpoint)
        #
        #     progbar = Progbar(target=self.train_steps_per_epoch)
        #     for step in count(start=1):
        #         if sv.should_stop():
        #             break
        #         start_time = time.time()
        #         fetches = { "train" : self.train_op,
        #                       "global_step" : self.global_step,
        #                       "incr_global_step": self.incr_global_step
        #                      }
        #         if step % config.summary_freq == 0:
        #             #########################################################
        #             # ADD HERE THE TENSORS YOU WANT TO EVALUATE (maybe loss)#
        #             #########################################################
        #             fetches["loss"] = self.total_loss
        #             fetches["accuracy"] = self.accuracy
        #             fetches["summary"] = self.step_sum
        #
        #         # Runs the series of operations
        #         ######################################################
        #         # REMOVE THE LEARNING PHASE IF NOT USING KERAS MODELS#
        #         ######################################################
        #         results = sess.run(fetches,
        #                            feed_dict={ self.is_training : True })
        #         progbar.update(step % self.train_steps_per_epoch)
        #
        #         gs = results["global_step"]
        #         if step % config.summary_freq == 0:
        #             sv.summary_writer.add_summary(results["summary"], gs)
        #             train_epoch = math.ceil( gs /self.train_steps_per_epoch)
        #             train_step = gs - (train_epoch - 1) * self.train_steps_per_epoch
        #             print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f "
        #                   " accuracy: %.3f" \
        #                % (train_epoch, train_step, self.train_steps_per_epoch, \
        #                         time.time() - start_time, results["loss"], \
        #                   results["accuracy"]))
        #
        #         if step % self.train_steps_per_epoch == 0:
        #             # This differ from the last when resuming training
        #             train_epoch = int(gs / self.train_steps_per_epoch)
        #             progbar = Progbar(target=self.train_steps_per_epoch)
        #             self.epoch_end_callback(sess, sv, train_epoch)
        #             if (train_epoch == self.config.max_epochs):
        #                 print("-------------------------------")
        #                 print("Training completed successfully")
        #                 print("-------------------------------")
        #                 break

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

    def build_test_graph(self):
        """This graph will be used for testing. In particular, it will
           compute the loss on a testing set, or some other utilities.
           Here, data will be passed though placeholders and not via
           input queues.
        """
        ##################################################################
        # UNCHANGED FOR CLASSIFICATION. ADAPT THE INPUT TO OTHER PROBLEMS#
        ##################################################################
        image_height, image_width = self.config.test_img_height, \
                                    self.config.test_img_width
        input_uint8 = tf.placeholder(tf.uint8, [None, image_height,
                                    image_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)

        gt_labels = tf.placeholder(tf.uint8, [None], name='gt_labels')
        input_labels = tf.cast(gt_labels, tf.int32)

        ################################################
        # DONT CHANGE NAMESCOPE (NECESSARY FOR LOADING)#
        ################################################
        with tf.name_scope("CNN_prediction"):
            logits = prediction_network(input_mc,
                    l2_reg_scale=self.config.l2_reg_scale, is_training=False,
                    output_dim=self.config.output_dim)

        ###########################################
        # ADAPT TO YOUR LOSSES OR TESTING METRICS #
        ###########################################

        with tf.name_scope("compute_loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=input_labels, logits= logits)
            loss = tf.reduce_mean(loss)

        with tf.name_scope("accuracy"):
            pred_out = tf.cast(tf.argmax(logits, 1), tf.int32)
            correct_prediction = tf.equal(input_labels, pred_out)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ################################################################
        # PUT HERE THE PLACEHOLDERS YOU NEED TO USE, AND OPERATIONS YOU#
        # WANT TO EVALUATE                                             #
        ################################################################
        self.inputs = input_uint8
        self.gt_labels = gt_labels
        self.total_loss = loss
        self.predictions = pred_out
        self.accuracy = accuracy

    def setup_inference(self, config):
        """Sets up the inference graph.
        Args:
            config: config dictionary.
        """
        self.config = config
        self.build_test_graph()

    def inference(self, inputs, sess):
        """Outputs a dictionary with the results of the required operations.
        Args:
            inputs: Dictionary with variable to be feed to placeholders
            sess: current session
        Returns:
            results: dictionary with output of testing operations.
        """
        ################################################################
        # CHANGE INPUTS TO THE PLACEHOLDER YOU NEED, AND OUTPUTS TO THE#
        # RESULTS OF YOUR OPERATIONS                                   #
        ################################################################
        results = {}
        results['loss'], results['accuracy'] = sess.run([self.total_loss,
                self.accuracy], feed_dict= {self.inputs: inputs['images'],
                                            self.gt_labels: inputs['labels']})
        return results
