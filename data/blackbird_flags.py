import gflags
FLAGS = gflags.FLAGS


###################################
# DEFINE HERE FLAGS FOR YOUR MODEL#
###################################


# Train parameters
gflags.DEFINE_string('blackbird_local_dir', 'blackbird_dataset', 'Local general directory of blackbird dataset')
gflags.DEFINE_string('blackbird_url', 'http://blackbird-dataset.mit.edu/BlackbirdDatasetData', 'URL of blackbird dataset')
gflags.DEFINE_string('blackbird_topics', '/blackbird/imu /blackbird/pose_ref', 'Topics to extract from the data')