import math
import numpy as np
import tensorflow as tf

def get_trilinear_filter(filter_size):
    factor = (filter_size + 1) // 2

    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:filter_size, :filter_size, :filter_size]

    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor) * \
             (1 - abs(og[2] - center) / factor)

    return filter

def trilinear_weights(shape, dtype=tf.float32):

    weights = np.zeros(shape)
    trilinear_weights = get_trilinear_filter(shape[0]);

    for i in range(shape[3]):
        weights[:, :, :, i, i] = trilinear_weights;

    init_weights = tf.constant_initializer(value=weights, dtype=dtype)

    return init_weights

def batch_norm(input):
    # Using instance normalization
    return tf.keras.layers.BatchNormalization()(input, training = True)

def deconv3d(input, output_ch, output_shape,
             k_d=4, k_h=4, k_w=4,
             s_d=2, s_h=2, s_w=2,
             stddev=0.02,
             name="deconv3d"):

    with tf.variable_scope(name):

        # filter : [depth, height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_d, k_h, k_w, output_ch, input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))

        deconv = tf.nn.conv3d_transpose(input, w, output_shape=output_shape, strides=[1, s_d, s_h, s_w, 1])
        biases = tf.get_variable('biases', [output_ch], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        # Add this variable to summary writer
        # tf.summary.histogram(name, w);
        # tf.summary.histogram(name, biases);

        return deconv

def deconv3d_trilinear(input, output_ch, output_shape,
             k_d=4, k_h=4, k_w=4,
             s_d=2, s_h=2, s_w=2,
             stddev=0.02,
             name="deconv3d"):

    with tf.variable_scope(name):

        # filter : [depth, height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_d, k_h, k_w, output_ch, input.get_shape()[-1]],
                            initializer=trilinear_weights([k_d, k_h, k_w, output_ch, input.get_shape()[-1]]),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))

        deconv = tf.nn.conv3d_transpose(input, w, output_shape=output_shape, strides=[1, s_d, s_h, s_w, 1])
        biases = tf.get_variable('biases', [output_ch], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        # Add this variable to summary writer
        # tf.summary.histogram(name, w);
        # tf.summary.histogram(name, biases);

        return deconv

def conv3d_xavier(input, output_ch,
           k_d=3, k_h=3, k_w=3,
            s_d=1, s_h=1, s_w=1,
             name="conv3d", with_w=False, dilation_rate=None):

    with tf.variable_scope(name):
        # filter : [depth, height, width, in_channels, out_channels]
        w = tf.get_variable('w', [k_d, k_h, k_w, input.get_shape()[-1], output_ch],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))

        if dilation_rate is not None:
            dilation_rate = dilation_rate

        conv = tf.nn.conv3d(input, w, strides=[1, s_d, s_h, s_w, 1], padding='SAME')
        # conv = tf.nn.convolution(input,w,strides=[1, s_d, s_h, s_w, 1], padding='SAME');
        biases = tf.get_variable('biases', [output_ch], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        # Add this variable to summary writer
        # tf.summary.histogram(name, w);
        # tf.summary.histogram(name, biases);

        if with_w:
            return conv, w, biases
        else:
            return conv
