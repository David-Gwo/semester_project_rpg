import gflags
FLAGS = gflags.FLAGS


###################################
# DEFINE HERE FLAGS FOR YOUR MODEL#
###################################

# Main parameters
gflags.DEFINE_string("model_name", "imu_int_lstm", "Name for the deep model")

# Train parameters
gflags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
gflags.DEFINE_float("learning_rate", 0.00001, "Learning rate for adam optimizer")  # 0.0000001
gflags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
gflags.DEFINE_integer("max_epochs", 3, "Maximum number of training epochs")
gflags.DEFINE_integer("patience", 10, "Patience epochs before interrupting training")

gflags.DEFINE_string('checkpoint_dir', "./results/", "Directory name to save checkpoints and logs.")
gflags.DEFINE_integer('window_length', 50, 'The number of past samples used to predict next velocity value')

gflags.DEFINE_string('train_ds', 'blackbird', 'Which dataset to use for training')
gflags.DEFINE_bool('plot_ds', False, 'Whether to plot the dataset during its generation')
gflags.DEFINE_bool('resume_train', False, 'Whether to restore a trained model for training')
gflags.DEFINE_integer("resume_train_model_number", 7, "Which model number to resume training")

# Log parameters
gflags.DEFINE_integer("summary_freq", 4, "Logging every log_freq iterations")
gflags.DEFINE_integer("save_freq", 1, "Save the latest model every save_freq epochs")

# Testing parameters
gflags.DEFINE_string('test_ds', 'blackbird', 'Which dataset to use for testing')
gflags.DEFINE_integer("test_model_number", 15, "Which model number to test")

# Debugging parameters
gflags.DEFINE_boolean('force_ds_remake', False, 'Whether to force re-processing of the dataset file')
