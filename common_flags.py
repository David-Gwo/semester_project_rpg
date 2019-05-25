import gflags
FLAGS = gflags.FLAGS


# Main parameters
gflags.DEFINE_string("model_name", "cnn_vel_net", "Name for the deep model")
gflags.DEFINE_integer("model_number", 0, "Which model to train or test")
gflags.DEFINE_string("model_type", "speed_regression_net", "Type of the deep model")
gflags.DEFINE_string('dataset', 'euroc', 'Which dataset to use for training and testing')
gflags.DEFINE_integer('window_length', 200, 'The number of past samples used to predict next velocity value')
gflags.DEFINE_string('dataset_type', "windowed_imu_speed_regression", 'Dataset structure to be built')

# Train parameters
gflags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
gflags.DEFINE_float("learning_rate", 0.00005, "Learning rate for adam optimizer")
gflags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
gflags.DEFINE_integer("max_epochs", 50, "Maximum number of training epochs")
gflags.DEFINE_integer("patience", 10, "Patience epochs before interrupting training")
gflags.DEFINE_integer("lr_scheduler", 10, "Reduce lr to half after this number of epochs")
gflags.DEFINE_string('checkpoint_dir', "./results/", "Directory name to save checkpoints and logs.")
gflags.DEFINE_bool('resume_train', False, 'Whether to restore a trained model for training')

# Log parameters
gflags.DEFINE_integer("summary_freq", 4, "Logging every log_freq iterations")
gflags.DEFINE_integer("save_freq", 3, "Save the latest model every save_freq epochs")

# Debugging parameters
gflags.DEFINE_boolean('force_ds_remake', False, 'Whether to force re-processing of the dataset file')
gflags.DEFINE_bool('plot_ds', False, 'Whether to plot the dataset during its generation')
