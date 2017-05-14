import imghdr
import os
import tensorflow as tf


def is_image_valid(filepath):
    return imghdr.what(filepath) is not None


def get_image_paths(image_dir):
    image_paths = []
    for root, directories, filenames in os.walk(image_dir):
        image_paths += [os.path.join(root, filename) for filename in filenames]
    image_paths = [filepath for filepath in image_paths if is_image_valid(filepath)]

    return image_paths


def inputs(image_dir, batch_size, min_queue_examples, input_height, input_width):
    def read_images(image_paths):
        filename_queue = tf.train.string_input_producer(image_paths)
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        image = tf.image.decode_image(value)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image.set_shape([None, None, 3])

        return image

    image_paths = get_image_paths(image_dir)
    images = read_images(image_paths)
    images = tf.image.crop_to_bounding_box(images, 30, 0, 178, 178)
    # images = tf.image.random_flip_left_right(images)
    images = tf.image.resize_images(images, [input_height, input_width])

    total_image_count = len(image_paths)
    input_batch = tf.train.shuffle_batch([images],
                                         batch_size=batch_size,
                                         num_threads=16,
                                         capacity=min_queue_examples + 3 * batch_size,
                                         min_after_dequeue=min_queue_examples)

    return input_batch, total_image_count


if __name__ == '__main__':
    pass
