import gflags
FLAGS = gflags.FLAGS


###################################
# DEFINE HERE FLAGS FOR YOUR MODEL#
###################################


# Train parameters
gflags.DEFINE_integer('img_width', 28, 'Target Image Width')
gflags.DEFINE_integer('img_height', 28, 'Target Image Height')
gflags.DEFINE_integer('batch_size', 10, 'Batch size in training and evaluation')
gflags.DEFINE_float("l2_reg_scale", 0.00000001, "Scale for regularization losses")
gflags.DEFINE_integer('output_dim', 1, "Number of outputs")

gflags.DEFINE_float("learning_rate", 0.00001, "Learning rate for adam optimizer")  # 0.0000001
gflags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
gflags.DEFINE_integer("max_epochs", 40, "Maximum number of training epochs")

gflags.DEFINE_string("model_name", "cnn_vel_net", "Name for the deep model")
gflags.DEFINE_string('checkpoint_dir', "./results/", "Directory name to save checkpoints and logs.")


###############################################################
# MAKE SURE TO CONFIG THIS PARAMETER SUCH THAT YOUR GPU USAGE #
# (that you can see with `$ nvidia-smi`) IS AT LEAST 60%      #
###############################################################
gflags.DEFINE_integer('num_threads', 8, 'Number of threads reading and ' 
                      '(optionally) preprocessing input files into queues')
gflags.DEFINE_integer('capacity_queue', 100, 'Capacity of input queue. A high '
                      'number speeds up computation but requires more RAM')

# Reading parameters
gflags.DEFINE_string('train_dir', './data/EuRoC_dataset_1/', 'Directory of the training dataset')
gflags.DEFINE_boolean('processed_train_ds', False, 'Whether there is a processed dataset file available to load from')
gflags.DEFINE_string('euroc_data_filename_train', 'imu_dataset_train.mat', 'Preprocessed EuRoC dataset training file')

# Log parameters
gflags.DEFINE_bool('resume_train', False, 'Whether to restore a trained model for training')
gflags.DEFINE_integer("resume_train_model_number", 1, "Which model number to resume training")
gflags.DEFINE_integer("summary_freq", 20, "Logging every log_freq iterations")
gflags.DEFINE_integer("save_freq", 5, "Save the latest model every save_freq epochs")

# Testing parameters
gflags.DEFINE_string('test_dir', './data/EuRoC_dataset_0/', 'Directory of the training dataset')
gflags.DEFINE_string('euroc_data_filename_test', 'imu_dataset_test.mat', 'Preprocessed EuRoC dataset testing file')
gflags.DEFINE_integer("test_model_number", 1, "Which model number to test")
gflags.DEFINE_string("ckpt_file", None, "Checkpoint file")
