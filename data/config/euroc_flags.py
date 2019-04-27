import gflags
FLAGS = gflags.FLAGS

# Train parameters
gflags.DEFINE_string('euroc_local_dir', './data/dataset/EuRoC_dataset/', 'Local general directory of the EuRoC dataset')

# Dataset selection parameters
gflags.DEFINE_string('dataset_version', 'dataset_1', 'Which version to use of the EuRoC dataset')
