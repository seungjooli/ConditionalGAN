import os
import numpy as np
import tensorflow as tf
import cv2


class Model:
    def __init__(self, log_dir, ckpt_dir, load_ckpt, image_height, image_width, image_channel=3):
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.image_channel = image_channel
        self.sess = tf.Session()
        self.trained_step = 0

        self.inputs = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel])
        self.targets = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel])
        self.is_training = tf.placeholder(tf.bool)
        self.g = self.create_generator(self.inputs, self.image_channel, self.is_training)
        self.d_fake = self.create_discriminator(self.inputs, self.g, self.is_training)
        self.d_real = self.create_discriminator(self.inputs, self.targets, self.is_training, reuse=True)

        self.sess.run(tf.global_variables_initializer())
        if load_ckpt:
            self.trained_step = self.load()

    def close(self):
        self.sess.close()

    @staticmethod
    def lrelu(input_, leak=0.2):
        with tf.name_scope('lrelu'):
            return tf.maximum(input_, leak * input_)

    @staticmethod
    def batch_norm(input_, is_training):
        with tf.name_scope('batchnorm'):
            return tf.contrib.layers.batch_norm(input_, is_training=is_training)

    @staticmethod
    def conv(input_, output_channels, filter_size=4, stride=2, stddev=3e-2):
        with tf.variable_scope('conv'):
            in_channels = input_.get_shape()[-1]
            filter_ = tf.get_variable(
                name='filter',
                shape=[filter_size, filter_size, in_channels, output_channels],
                initializer=tf.truncated_normal_initializer(stddev=stddev),
            )
            conv = tf.nn.conv2d(input_, filter_, [1, stride, stride, 1], padding='SAME')
            return conv

    @staticmethod
    def deconv(input_, out_height, out_width, out_channels, filter_size=4, stride=2, stddev=3e-2):
        with tf.variable_scope("deconv"):
            in_channels = input_.get_shape().as_list()[-1]
            filter_ = tf.get_variable(
                name='filter',
                shape=[filter_size, filter_size, out_channels, in_channels],
                initializer=tf.truncated_normal_initializer(stddev=stddev),
            )

            batch_dynamic = tf.shape(input_)[0]
            output_shape = tf.stack([batch_dynamic, out_height, out_width, out_channels])
            conv = tf.nn.conv2d_transpose(input_, filter_, output_shape, [1, stride, stride, 1], padding="SAME")
            conv = tf.reshape(conv, [-1, out_height, out_width, out_channels])
            return conv

    @staticmethod
    def detect_edges(images):
        def blur(image):
            return cv2.GaussianBlur(image, (5, 5), 0)

        def canny_otsu(image):
            scale_factor = 255
            scaled_image = np.uint8(image * scale_factor)

            otsu_threshold = cv2.threshold(
                cv2.cvtColor(scaled_image, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
            lower_threshold = max(0, int(otsu_threshold * 0.5))
            upper_threshold = min(255, int(otsu_threshold))
            edges = cv2.Canny(scaled_image, lower_threshold, upper_threshold)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            return np.float32(edges) * (1 / scale_factor)

        blurred = [blur(image) for image in images]
        canny_applied = [canny_otsu(image) for image in blurred]

        return canny_applied

    @staticmethod
    def create_generator(input_, generator_output_channels, is_training):
        class Encoder:
            def __init__(self, name, out_channels, is_training, use_batch_norm=True):
                self.name = name
                self.out_channels = out_channels
                self.is_training = is_training
                self.use_batch_norm = use_batch_norm
                self.in_height = None
                self.in_width = None
                self.output = None

            def encode(self, input_):
                with tf.variable_scope(self.name):
                    output = Model.conv(input_, self.out_channels)
                    if self.use_batch_norm:
                        output = Model.batch_norm(output, self.is_training)
                    output = Model.lrelu(output)

                    input_shape = input_.get_shape().as_list()
                    self.in_height = input_shape[1]
                    self.in_width = input_shape[2]
                    self.output = output
                    return output

        class Decoder:
            def __init__(self, name, out_channels, is_training, use_batch_norm=True, dropout=None):
                self.name = name
                self.out_channels = out_channels
                self.is_training = is_training
                self.use_batch_norm = use_batch_norm
                self.dropout = dropout
                self.output = None

            def decode(self, input_, out_height, out_width, skip_input=None):
                with tf.variable_scope(self.name):
                    if skip_input is None:
                        merged_input = input_
                    else:
                        merged_input = tf.concat([input_, skip_input], axis=3)

                    output = Model.deconv(merged_input, out_height, out_width, self.out_channels)
                    if self.use_batch_norm:
                        output = Model.batch_norm(output, self.is_training)
                    output = tf.nn.relu(output)
                    if self.dropout:
                        output = tf.nn.dropout(output, keep_prob=1 - self.dropout)

                    self.output = output
                    return output

        with tf.variable_scope('generator'):
            ngf = 64

            encoders = [
                Encoder('encoder_0', ngf * 1, is_training, use_batch_norm=False),
                Encoder('encoder_1', ngf * 2, is_training),
                Encoder('encoder_2', ngf * 4, is_training),
                Encoder('encoder_3', ngf * 8, is_training),
                Encoder('encoder_4', ngf * 8, is_training),
                Encoder('encoder_5', ngf * 8, is_training),
                Encoder('encoder_6', ngf * 8, is_training),
                Encoder('encoder_7', ngf * 8, is_training),
            ]

            for i, encoder in enumerate(encoders):
                if i == 0:
                    encoder_input = input_
                else:
                    encoder_input = encoders[i - 1].output
                encoders[i].encode(encoder_input)

            decoders = [
                Decoder('decoder_0', ngf * 8, is_training, dropout=0.5),
                Decoder('decoder_1', ngf * 8, is_training, dropout=0.5),
                Decoder('decoder_2', ngf * 8, is_training, dropout=0.5),
                Decoder('decoder_3', ngf * 8, is_training),
                Decoder('decoder_4', ngf * 4, is_training),
                Decoder('decoder_5', ngf * 2, is_training),
                Decoder('decoder_6', ngf * 1, is_training),
                Decoder('decoder_7', generator_output_channels, is_training),
            ]

            for i, decoder in enumerate(decoders):
                if i == 0:
                    decoder_input = encoders[-1].output
                    decoder_skip_input = None
                else:
                    decoder_input = decoders[i - 1].output
                    decoder_skip_input = encoders[-i - 1].output

                decoders[i].decode(decoder_input, encoders[-i - 1].in_height, encoders[-i - 1].in_width, decoder_skip_input)

            return tf.nn.tanh(decoders[-1].output)

    @staticmethod
    def create_discriminator(input_, target, is_training, reuse=False):
        class Layer:
            def __init__(self, name, output_channels, stride, is_training, use_batch_norm=True, use_activation=True):
                self.name = name
                self.output_channels = output_channels
                self.stride = stride
                self.is_training = is_training
                self.use_batch_norm = use_batch_norm
                self.use_activation = use_activation
                self.output = None

            def conv(self, input_):
                with tf.variable_scope(self.name):
                    output = Model.conv(input_, self.output_channels, stride=self.stride)
                    if self.use_batch_norm:
                        output = Model.batch_norm(output, self.is_training)
                    if self.use_activation:
                        output = Model.lrelu(output)

                    self.output = output
                    return output

        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            ndf = 64

            layers = [
                Layer('layer_0', ndf * 1, 2, is_training, use_batch_norm=False),
                Layer('layer_1', ndf * 2, 2, is_training),
                Layer('layer_2', ndf * 4, 2, is_training),
                Layer('layer_3', ndf * 8, 1, is_training),
                Layer('layer_4', 1, 1, is_training, use_batch_norm=False, use_activation=False),
            ]

            for i, layer in enumerate(layers):
                if i == 0:
                    layer_input = tf.concat([input_, target], axis=3)
                else:
                    layer_input = layers[i - 1].output
                layers[i].conv(layer_input)

            return tf.nn.sigmoid(layers[-1].output)

    def train(self, training_image, test_image, total_epoch, steps_per_epoch, learning_rate, l1_weight, beta1, load_ckpt):
        epsilon = 1e-8

        loss_g_gan = tf.reduce_mean(-tf.log(self.d_fake + epsilon))
        loss_g_l1 = l1_weight * tf.reduce_mean(tf.abs(self.targets - self.g))
        loss_g = loss_g_gan + loss_g_l1

        loss_d_real = tf.reduce_mean(-tf.log(self.d_real + epsilon))
        loss_d_fake = tf.reduce_mean(-tf.log(tf.ones_like(self.d_fake) - self.d_fake + epsilon))
        loss_d = loss_d_real + loss_d_fake

        vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        # update batch_norm moving_mean, moving_variance
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_g = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss_g, var_list=vars_g)
            train_d = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss_d, var_list=vars_d)

        tf.summary.image('training_truth', self.targets, 4)
        tf.summary.image('training_input', self.inputs, 4)
        tf.summary.image('training_output', self.g, 4)

        tf.summary.histogram('D_real', self.d_real)
        tf.summary.histogram('D_fake', self.d_fake)

        tf.summary.scalar('G_loss', loss_g)
        tf.summary.scalar('G_loss_gan', loss_g_gan)
        tf.summary.scalar('G_loss_l1', loss_g_l1)
        tf.summary.scalar('D_loss', loss_d)
        tf.summary.scalar('D_loss_real', loss_d_real)
        tf.summary.scalar('D_loss_fake', loss_d_fake)

        for var in vars_g:
            tf.summary.histogram(var.name, var)
        for var in vars_d:
            tf.summary.histogram(var.name, var)

        training_summary = tf.summary.merge_all()

        test_summary_truth = tf.summary.image('test_truth', self.targets, 4)
        test_summary_input = tf.summary.image('test_input', self.inputs, 4)
        test_summary_output = tf.summary.image('test_output', self.g, 4)
        test_summary = tf.summary.merge([test_summary_input, test_summary_output, test_summary_truth])

        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # FIXME
        self.sess.run(tf.global_variables_initializer())
        if load_ckpt:
            self.trained_step = self.load()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        print('Training start')
        for epoch in range(total_epoch):
            for step in range(steps_per_epoch):
                image_value = self.sess.run(training_image)
                edges = self.detect_edges(image_value)

                feed_dict = {self.inputs: edges, self.targets: image_value, self.is_training: True}
                self.sess.run(train_d, feed_dict=feed_dict)
                self.sess.run(train_g, feed_dict=feed_dict)
                self.sess.run(train_g, feed_dict=feed_dict)

                self.trained_step += 1
                if self.trained_step % 100 == 0:
                    print('step: {}'.format(self.trained_step))

                    training_summary_value = self.sess.run(training_summary, feed_dict=feed_dict)
                    writer.add_summary(training_summary_value, self.trained_step)

                    image_value = self.sess.run(test_image)
                    edges = self.detect_edges(image_value)

                    feed_dict = {self.inputs: edges, self.targets: image_value, self.is_training: False}
                    test_summary_value = self.sess.run(test_summary, feed_dict=feed_dict)
                    writer.add_summary(test_summary_value, self.trained_step)

                    if self.trained_step % 1000 == 0:
                        self.save()

        coord.join(threads)

    def test(self, inputs):
        output = self.sess.run(self.g, feed_dict={self.inputs: inputs, self.is_training: False})
        return output

    def save(self):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(self.sess, self.ckpt_dir, global_step=self.trained_step)

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt.model_checkpoint_path)

            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            trained_step = int(os.path.splitext(ckpt_name)[0][1:])

            return trained_step
        else:
            return 0
