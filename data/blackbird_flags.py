import gflags
FLAGS = gflags.FLAGS

# Train parameters
gflags.DEFINE_string('blackbird_local_dir', './data/blackbird_dataset/', 'Local general directory of blackbird dataset')
gflags.DEFINE_string('blackbird_url', 'http://blackbird-dataset.mit.edu/BlackbirdDatasetData', 'URL of blackbird dataset')
gflags.DEFINE_string('blackbird_topics', '/blackbird/imu', 'List of topics to extract, separated by spaces')

# Dataset specifications
gflags.DEFINE_string('trajectory_name', 'bentDice', 'The name of the trajectory to use from the dataset')
gflags.DEFINE_string('yaw_type', 'yawForward', 'The yaw type to use from the dataset')
gflags.DEFINE_string('max_speed', 2.0, 'The maximum speed of the drone in the dataset')
