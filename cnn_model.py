import tensorflow as tf


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    with tf.name_scope("input"):
        x = features["x"]
        y = labels

    # Input Layer
    with tf.name_scope("reshape_layer"):
        input_layer = tf.reshape(x, [-1, 28, 28, 1])

    # Convolutional Layer #1
    with tf.name_scope("convolutional_1"):
        conv1 = tf.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)(input_layer)

    # Pooling Layer #1
    with tf.name_scope("pooling_1"):
        pool1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    # Convolutional Layer #2 and Pooling Layer #2
    with tf.name_scope("convolutional_2"):
        conv2 = tf.layers.Conv2D(filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)(pool1)
    with tf.name_scope("pooling_2"):
        pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)

    # Dense Layer and Dropout
    with tf.name_scope("dense_1"):
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.Dense(units=1024, activation=tf.nn.relu)(pool2_flat)

    with tf.name_scope("dropout"):
        dropout = tf.layers.Dropout(rate=0.4)(dense)

    # Logits Layer
    with tf.name_scope("logits"):
        logits = tf.layers.Dense(units=10)(dropout)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
