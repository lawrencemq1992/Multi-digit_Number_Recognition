import tensorflow as tf


class Utils(object):
    
    @staticmethod
    def build_batch(path_tfrecords, num_example, batch_size, shuffled):
        filenames = tf.train.string_input_producer([path_tfrecords], num_epochs=None)
        tfreader = tf.TFRecordReader()
        _, serialized_example = tfreader.read(filenames)
        single_example = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'length': tf.FixedLenFeature([], tf.int64),
                'digits': tf.FixedLenFeature([5], tf.int64)
            })
        image = tf.decode_raw(single_example['image'], tf.uint8)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.multiply(tf.subtract(image, 0.5), 2)
        image = tf.reshape(image, [64, 64, 3])
        image = tf.random_crop(image, [54, 54, 3])
        length = tf.cast(single_example['length'], tf.int32)
        digits = tf.cast(single_example['digits'], tf.int32)
        num_queue_examples = int(0.4 * num_example)
        if shuffled:
            image_batch, length_batch, digits_batch = tf.train.shuffle_batch(
                [image, length, digits],
                batch_size=batch_size,
                num_threads=2,
                capacity=num_queue_examples + 3 * batch_size,
                min_after_dequeue=num_queue_examples)
        else:
            image_batch, length_batch, digits_batch = tf.train.batch(
                [image, length, digits],
                batch_size=batch_size,
                num_threads=2,
                capacity=num_queue_examples + 3 * batch_size)
        return image_batch, length_batch, digits_batch