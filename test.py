import sys
import gflags
from models.base_learner import Learner

from common_flags import FLAGS


def _main():

    learner = Learner(FLAGS)
    learner.build_and_compile_model()
    if FLAGS.generate_training_progression:
        model_pos = 0
        while model_pos != -1:
            model_pos = learner.recover_model_from_checkpoint(mode="test", model_used_pos=model_pos)
            learner.evaluate_model(save_figures=True)
    else:
        learner.recover_model_from_checkpoint(mode="test")
        learner.evaluate_model(compare_manual=True)
        learner.iterate_model_output()


def main(argv):
    # Utility main to load flags
    try:
        argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
