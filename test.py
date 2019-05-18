import sys
import gflags
from models.base_learner import Learner
import tensorflow as tf
import numpy as np
import pprint
import random

from common_flags import FLAGS
from experiments.experiment_default_confs import experiment_builder


def _main():

    seed = 8964
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    print_flags_dict = {}
    for key in FLAGS.__flags.keys():
        print_flags_dict[key] = getattr(FLAGS, key)

    pp.pprint(print_flags_dict)
    learner = Learner(FLAGS)

    learner.test(experiment_builder(FLAGS.model_type))


def main(argv):
    # Utility main to load flags
    try:
        _ = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
