import gflags
FLAGS = gflags.FLAGS

# Dataset selection parameters
gflags.DEFINE_string('trajectory_name', 'bentDice', 'The name of the trajectory to use from the dataset')
gflags.DEFINE_string('yaw_type', 'yawForward', 'The yaw type to use from the dataset')
gflags.DEFINE_float('max_speed', 2.0, 'The maximum speed of the drone in the dataset')
