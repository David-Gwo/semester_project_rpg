import os
import sys
import gflags
import math
import tensorflow as tf
import json

from models.base_learner import Learner
#######################################################
# IMPORT HERE A SCRIPT TO TEST YOUR NETWORK.          #
# HAVE A LOOK AT `compute_loss' FOR AN EXAMPLE        #
#######################################################
from utils import compute_loss
#############################################################################
# IMPORT HERE A LIBRARY TO PRODUCE ALL THE FILENAMES (and optionally labels)#
# OF YOUR DATASET. HAVE A LOOK AT `DirectoryIterator' FOR AN EXAMPLE        #
#############################################################################
from data import DirectoryIterator

from common_flags import FLAGS

def _main():
    learner = Learner()
    learner.setup_inference(FLAGS)

    saver = tf.train.Saver([var for var in tf.trainable_variables()])

    test_generator = DirectoryIterator(FLAGS.test_dir,
                          shuffle=False,
                          target_size=(FLAGS.img_height, FLAGS.img_width),
                          batch_size = FLAGS.batch_size)

    steps = int(math.ceil(test_generator.samples / FLAGS.batch_size))

    with tf.Session() as sess:
        try:
            saver.restore(sess, FLAGS.ckpt_file)
            print("--------------------------------------------------")
            print("Restored checkpoint file {}".format(FLAGS.ckpt_file))
            print("--------------------------------------------------")
        except:
            print("--------------------------------------------------")
            print("Impossible to find weight path. Returning untrained model")
            print("--------------------------------------------------")

        results = compute_loss(sess, learner, test_generator, steps, verbose=1)


    ####################################################
    # LOG YOUR TESTING METRICS (stdout and json file)  #
    ####################################################

    print("\nAverage Loss: {:.3f}".format(results['loss']))
    print("Average Accuracy: {:.3f}".format(results['accuracy']))

    # Write results in output directory
    out_filename = FLAGS.ckpt_file.split('/')[:-1]
    out_filename = os.path.join(*out_filename, 'test_results.json')

    with open(out_filename, "w") as f:
        json.dump(results, f)
        print("Written evaluation file {}".format(out_filename))

def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _main()

if __name__ == "__main__":
    main(sys.argv)
