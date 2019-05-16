import gflags
FLAGS = gflags.FLAGS

# Dataset selection parameters
gflags.DEFINE_string('dataset_version', 'dataset_1', 'Which version to use of the EuRoC dataset')
