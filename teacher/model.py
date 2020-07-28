import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import vgg

class Model(object):

    def __init__(self, input_shape=[64, 64], batch_size=32, is_training=True,
                 rot_bins = np.arange(-np.pi, np.pi, np.pi/8)):
        self.input_image = tf.placeholder(
                           dtype=tf.float32, 
                           shape=[None, 
                           input_shape[0], 
                           input_shape[1], 
                           1], name='input')
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.rot_bins = rot_bins
        self.build_model(self.input_image, rot_bins, is_training)
        return

    def batch_norm_layer(self, x, eps=0.01):
        dimension = x.get_shape().as_list()[-1]
        mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
        beta = tf.get_variable('bta',
                               dimension,
                               tf.float32,
                               initializer= \
               tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gma',
                                dimension,
                                tf.float32,
                                initializer= \
                tf.constant_initializer(1.0, tf.float32))
        x = tf.nn.batch_normalization(x, mean,
                                variance, beta,
                                gamma, eps)
        return x

    def conv_layer(self, x, out_channels, 
                   kernel_size=3, dilation=1, 
                   linear=False, name=None):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(
                x, out_channels, 
                kernel_size=kernel_size, 
                dilation_rate=dilation,
                padding='same')
            if linear:
                return x
            x = self.batch_norm_layer(x)
            x = tf.nn.relu(x)
            return x

    def fc_layer(self, x, out_size, 
                 linear=False, name=None):
        with tf.variable_scope(name):
            x = tf.contrib.layers.flatten(x)
            x = tf.contrib.layers.fully_connected(
                 x, out_size, activation_fn=None)
            if linear:
                return x
            x = tf.nn.relu(x)
            return x

    def max_pool(self, x, name):
        with tf.variable_scope(name):
            x = tf.contrib.layers.max_pool2d(
                x, kernel_size=2, padding='SAME')
            return x
    
    def prob_to_rot(self, prob, rot_bins, name):
        rot_sin = np.sin(rot_bins).reshape([1, -1]) * prob
        rot_cos = np.cos(rot_bins).reshape([1, -1]) * prob
        rot_pred = tf.atan2(tf.reduce_sum(rot_sin, axis=-1),
                                tf.reduce_sum(rot_cos, axis=-1),
                                name=name)
        return rot_pred

    def build_model(self, inputs, rot_bins, is_training):
        with tf.variable_scope('backbone'):
            x = self.conv_layer(inputs, 32, name='conv1_1')
            x = self.conv_layer(x, 32, name='conv1_2')
            x = self.max_pool(x, 'pool1')

            x = self.conv_layer(x, 64, name='conv2_1')
            x = self.conv_layer(x, 64, name='conv2_2')
            x = self.max_pool(x, 'pool2')

            x = self.conv_layer(x, 128, name='conv3_1')
            x = self.conv_layer(x, 128, name='conv3_2')
            x = self.max_pool(x, 'pool3')

            x = self.conv_layer(x, 256, name='conv4_1')
            x = self.conv_layer(x, 256, name='conv4_2')
            x = self.conv_layer(x, 256, name='conv4_3')
            x = self.max_pool(x, 'pool4')

            x = self.conv_layer(x, 512, name='conv5_1')
            x = self.conv_layer(x, 512, name='conv5_2')
            x = self.conv_layer(x, 512, name='conv5_3')
            x = self.max_pool(x, 'pool5')

            x_c = self.fc_layer(x, 512, name='fc7_1')
            x_c = self.fc_layer(x_c, 256, name='fc7_2')
            x_c = self.fc_layer(x_c, 2, linear=True, name='fc7_3')

            x_f = self.fc_layer(x, 512, name='fc8_1')
            x_f = self.fc_layer(x_f, 256, name='fc8_2')
            x_f = self.fc_layer(x_f, 2, linear=True, name='fc8_3')

            x = self.fc_layer(x, 512, name='fc6_1')
            x = self.fc_layer(x, 256, name='fc6_2')
            x = self.fc_layer(x, len(rot_bins), linear=True, name='fc6_3')

        with tf.name_scope('output'): 
            prob = tf.nn.softmax(x)
            rot_pred = self.prob_to_rot(prob, rot_bins, 'rotation')
            self.rotation_prob = prob
            self.rotation = rot_pred
            
            self.class_prob = tf.nn.softmax(x_c)
            self.full_prob = tf.nn.softmax(x_f)
        
