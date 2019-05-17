import gflags
FLAGS = gflags.FLAGS

# Dataset selection parameters
gflags.DEFINE_string('dataset_version', 'euroc', 'Which version of the generated dataset to use')
gflags.DEFINE_string('flight', 'MH01', 'Which flight version of the dataset to use')
gflags.DEFINE_integer('number', 0, 'Which number of the generated version to use')
