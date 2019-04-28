import gflags
FLAGS = gflags.FLAGS


###################################
# DEFINE HERE FLAGS FOR YOUR MODEL#
###################################


# Train parameters
gflags.DEFINE_integer('batch_size', 20, 'Batch size in training and evaluation')
gflags.DEFINE_float("l2_reg_scale", 0.00000001, "Scale for regularization losses")

gflags.DEFINE_float("learning_rate", 0.00001, "Learning rate for adam optimizer")  # 0.0000001
gflags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
gflags.DEFINE_integer("max_epochs", 50, "Maximum number of training epochs")

gflags.DEFINE_string("model_name", "imu_int_50_depth", "Name for the deep model")
gflags.DEFINE_string('checkpoint_dir', "./results/", "Directory name to save checkpoints and logs.")
gflags.DEFINE_integer('window_length', 50, 'The number of past samples used to predict next velocity value')
gflags.DEFINE_integer('output_size', 9, 'The output size at the end of the deep model')

###############################################################
# MAKE SURE TO CONFIG THIS PARAMETER SUCH THAT YOUR GPU USAGE #
# (that you can see with `$ nvidia-smi`) IS AT LEAST 60%      #
###############################################################
gflags.DEFINE_integer('num_threads', 8, 'Number of threads reading and ' 
                      '(optionally) pre-processing input files into queues')
gflags.DEFINE_integer('capacity_queue', 100, 'Capacity of input queue. A high '
                      'number speeds up computation but requires more RAM')

# Reading parameters
gflags.DEFINE_string('train_ds', 'blackbird', 'Which dataset to use for training')
gflags.DEFINE_boolean('force_ds_remake', False, 'Whether to force re-processing of the dataset file')
gflags.DEFINE_string('prepared_train_data_file', 'imu_dataset_train.mat', 'Pre-processed dataset training file')
gflags.DEFINE_bool('plot_ds', False, 'Whether to plot the dataset during its generation')

# Log parameters
gflags.DEFINE_bool('resume_train', True, 'Whether to restore a trained model for training')
gflags.DEFINE_integer("resume_train_model_number", 3, "Which model number to resume training")
gflags.DEFINE_integer("summary_freq", 20, "Logging every log_freq iterations")
gflags.DEFINE_integer("save_freq", 5, "Save the latest model every save_freq epochs")

# Testing parameters
gflags.DEFINE_string('test_ds', 'blackbird', 'Which dataset to use for testing')
gflags.DEFINE_string('prepared_test_data_file', 'imu_dataset_test.mat', 'Preprocessed dataset testing file')
gflags.DEFINE_integer("test_model_number", 3, "Which model number to test")
