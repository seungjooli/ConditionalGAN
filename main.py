import os
import numpy as np
import tensorflow as tf
import cv2
import download
import input
import model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('download_data', False, 'whether to download, extract image data')
flags.DEFINE_string('download_dir', './downloads/', 'directory path to download data')
flags.DEFINE_string('train_dir', './images/train/', 'directory path to training set')
flags.DEFINE_string('test_dir', './images/test/', 'directory path to test set')

flags.DEFINE_integer('input_height', 256, 'resized image height, model input')
flags.DEFINE_integer('input_width', 256, 'resized image width, model input')

flags.DEFINE_string('mode', 'train', 'train or test')
flags.DEFINE_boolean('load_ckpt', True, 'whether to try restoring model from checkpoint')
flags.DEFINE_string('ckpt_dir', './checkpoints/', 'directory path to checkpoint files')

flags.DEFINE_integer('epoch', 10, 'total number of epoch to train')
flags.DEFINE_integer('batch_size', 4, 'size of batch')
flags.DEFINE_integer('min_queue_examples', 1000, 'minimum number of elements in batch queue')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('l1_weight', 100, 'weight on L1 term for generator')
flags.DEFINE_float('beta1', 0.5, 'adam optimizer beta1 parameter')
flags.DEFINE_string('log_dir', './logs/', 'directory path to write summary')


def main(argv):
    m = model.Model(FLAGS.log_dir, FLAGS.ckpt_dir, FLAGS.load_ckpt, FLAGS.input_height, FLAGS.input_width)
    if FLAGS.mode == 'train':
        train(m)
    elif FLAGS.mode == 'test':
        test(m)
    else:
        print('Unexpected mode: {}  Choose \'train\' or \'test\''.format(FLAGS.mode))
    m.close()


def train(m):
    if FLAGS.download_data:
        google_drive_file_id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
        download_path = os.path.join(FLAGS.download_dir, 'img_align_celeba.zip')
        download.maybe_download_from_google_drive(google_drive_file_id, download_path)
        download.maybe_extract(download_path, FLAGS.train_dir, FLAGS.test_dir)

    training_inputs, count = input.inputs(FLAGS.train_dir, FLAGS.batch_size, FLAGS.min_queue_examples,
                                          FLAGS.input_height, FLAGS.input_width)
    steps_per_epoch = int(count / FLAGS.batch_size)

    test_inputs, _ = input.inputs(FLAGS.test_dir, FLAGS.batch_size, 0, FLAGS.input_height, FLAGS.input_width)

    m.train(training_inputs, test_inputs,
            FLAGS.epoch, steps_per_epoch, FLAGS.learning_rate, FLAGS.l1_weight, FLAGS.beta1, FLAGS.load_ckpt)


def test(m):
    class DrawingState:
        def __init__(self):
            self.x_prev = 0
            self.y_prev = 0
            self.drawing = False
            self.update = True

    def interactive_drawing(event, x, y, flags, param):
        image = param[0]
        state = param[1]
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.x_prev, state.y_prev = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if state.drawing:
                cv2.line(image, (state.x_prev, state.y_prev), (x, y), (1, 1, 1), 1)
                state.x_prev = x
                state.y_prev = y
                state.update = True
        elif event == cv2.EVENT_LBUTTONUP:
            state.drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            image.fill(0)
            state.update = True

    cv2.namedWindow('Canvas')
    image_input = np.zeros((FLAGS.input_height, FLAGS.input_width, 3), np.float32)
    state = DrawingState()
    cv2.setMouseCallback('Canvas', interactive_drawing, [image_input, state])
    while cv2.getWindowProperty('Canvas', 0) >= 0:
        if state.update:
            reshaped_image_input = np.array([image_input])
            image_output = m.test(reshaped_image_input)
            concatenated = np.concatenate((image_input, image_output[0]), axis=1)
            color_converted = cv2.cvtColor(concatenated, cv2.COLOR_RGB2BGR)
            cv2.imshow('Canvas', color_converted)
            state.update = False

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # esc
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()
