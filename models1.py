import tensorflow as tf


class Model1(object):

    @staticmethod
    def evaluate(path_results, path_tfrecords, num_examples, tf_step_result):
        # this function is to make validation and evaluate accuracy
        # use extra as the data for validation
        batch_size = 128
        num_batches = num_examples / batch_size
        with tf.Graph().as_default():
            image_batch, length_batch, digits_batch = Utils.build_batch(
                path_tfrecords, num_examples, batch_size, False) # shuffled = False
            length, digits = Models.cnn_inference(image_batch, 0.0) # drop_rate = 0.0
            length_max = tf.argmax(length, axis=1)
            digits_max = tf.argmax(digits, axis=2)
            labels = digits_batch
            predictions = digits_max
            label_strings = tf.reduce_join(tf.as_string(labels), axis=1)
            prediction_strings = tf.reduce_join(tf.as_string(predictions), axis=1)
            accuracy, updated_op = tf.metric.accuracy(labels=label_strings, predictions=prediction_strings)
            tf.summary.image('image', image_batch)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram(
                'variables', tf.concat([tf.reshape(variable, [-1]) for variable in tf,trainable_variables()], axis=0))
            summary=tf.summary.merge_all()
            with tf.Session as sess:
                summary_writer = tf.summary.FileWriter(path_results, sess.graph)
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                coordinator = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
                for i in range(1, num_batches):
                    sess.run(updated_op)
                accuracy_result, summary_result = sess.run([accuracy, summary])
                summary_writer.add_summary(summary_result, global_step=tf_step_result)
        coordinator.request_stop()
        coordinator.join(threads)
        return accuracy_result
    
    @staticmethod
    def cnn_loss(length, digits, length_batch, digits_batch):
        # function as a minimizer calculator
        cross_entropys = []
        # length cross entropy
        cross_entropys.append(
            tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length, logits=length_batch)))
        for i in range(5):
            cross_entropys.append(
                tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits[:, i], logits=digits_batch[:, i, :])))
        return sum(cross_entropys)
    
    @staticmethod
    def cnn_inference(image, drop_rate):
        # feedforward
        # https://arxiv.org/pdf/1312.6082.pdf Section 5.1
        with tf.variable_scope('hidden1'):
            conv2d = tf.layers.conv2d(image, filters=48, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv2d)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden1 = dropout
        with tf.variable_scope('hidden2'):
            conv2d = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv2d)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden2 = dropout
        with tf.variable_scope('hidden3'):
            conv2d = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv2d)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden3 = dropout
        with tf.variable_scope('hidden4'):
            conv2d = tf.layers.conv2d(hidden3, filters=160, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv2d)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden4 = dropout
        with tf.variable_scope('hidden5'):
            conv2d = tf.layers.conv2d(hidden4, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv2d)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden5 = dropout
        with tf.variable_scope('hidden6'):
            conv2d = tf.layers.conv2d(hidden5, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv2d)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden6 = dropout
        with tf.variable_scope('hidden7'):
            conv2d = tf.layers.conv2d(hidden6, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv2d)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden7 = dropout
        with tf.variable_scope('hidden8'):
            conv2d = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv2d)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden8 = dropout
        flatten = tf.reshape(hidden8, [-1, 4 * 4 * 192])  # 3072 = 4 * 4 * 192
        with tf.variable_scope('hidden9'):
            dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.sigmoid)
            hidden9 = dense
        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(hidden9, units=3072, activation=tf.nn.relu)
            hidden10 = dense
        with tf.variable_scope('hidden11'):
            dense = tf.layers.dense(hidden10, units=3072, activation=tf.nn.relu)
            hidden11 = dense
        with tf.variable_scope('digit_length'):
            dense = tf.layers.dense(hidden11, units=7)
            digit_length = dense
        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(hidden11, units=11)
            digit1 = dense
        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(hidden11, units=11)
            digit2 = dense
        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(hidden11, units=11)
            digit3 = dense
        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(hidden11, units=11)
            digit4 = dense
        with tf.variable_scope('digit5'):
            dense = tf.layers.dense(hidden11, units=11)
            digit5 = dense
        return digit_length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)