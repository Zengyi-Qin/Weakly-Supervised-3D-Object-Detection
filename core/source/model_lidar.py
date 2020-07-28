import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import vgg

class ModelLiDAR(object):

    def __init__(self):
        return

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
            x = self.group_norm_layer(x)
            x = tf.nn.relu(x)
            return x

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

    def group_norm_layer(self, x, G=4, eps=1e-5):
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        gamma = tf.Variable(tf.constant(1.0, shape=[C]), 
                    dtype=tf.float32, name='gamma')
        beta = tf.Variable(tf.constant(0.0, shape=[C]), 
                    dtype=tf.float32, name='beta')
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        output = tf.transpose(output, [0, 2, 3, 1])
        return output

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

    def max_pool(self, x, stride, name):
        with tf.variable_scope(name):
            x = tf.nn.max_pool(x,
                    [1, 2, 2, 1],
                    padding='SAME',
                    strides=[1, stride[0], stride[1], 1],
                    name=name)
            return x

    def prob_to_rot(self, prob, rot_bins, name):
        rot_sin = np.sin(rot_bins).reshape([1, -1]) * prob
        rot_cos = np.cos(rot_bins).reshape([1, -1]) * prob
        rot_pred = tf.atan2(tf.reduce_sum(rot_sin, axis=-1),
                                tf.reduce_sum(rot_cos, axis=-1),
                                name=name)
        return rot_pred
    
    def build(self, sphere_map_cam,
              bbox, box_indices, bottom_centers, 
              roi_feature_size=[2, 2],
              roi_point_cloud_size=[32, 32],
              rot_bins = np.arange(-np.pi, np.pi, np.pi/8)):
      
        with tf.variable_scope('backbone_lidar', 
                 reuse=tf.AUTO_REUSE):
            x = tf.reshape(sphere_map_cam, 
                          [1, 64, 512, 5])
            x = self.conv_layer(x, 32, name='conv1_1')
            x = self.conv_layer(x, 32, name='conv1_2')
            x = self.max_pool(x, [1, 2], 'pool1')

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')
            x = self.max_pool(x, [1, 2], 'pool2')

            x = self.conv_layer(x, 64, name='conv3_1')
            x = self.conv_layer(x, 64, name='conv3_2')
            x = self.max_pool(x, [1, 2], 'pool3')

            x = self.conv_layer(x, 128, name='conv4_1')
            x = self.conv_layer(x, 128, name='conv4_2')
            x = self.max_pool(x, [2, 2], 'pool4')

            x = self.conv_layer(x, 128, name='conv5_1')
            x = self.conv_layer(x, 128, name='conv5_2')
            backbone_out = self.max_pool(x, [2, 2], 'pool5')
       
        with tf.variable_scope('class_rot_lidar',
                 reuse=tf.AUTO_REUSE):

            xyz_map_cam, _ = tf.split(
                                 tf.reshape(
                                     sphere_map_cam,
                                     [1, 64, 512, 5]), 
                                 [3, 2], axis=3)
            xyz_crops = tf.image.crop_and_resize(
                              image=xyz_map_cam,
                              boxes=bbox,
                              box_ind=box_indices,
                              crop_size= \
                              roi_point_cloud_size)
            centers = tf.reshape(bottom_centers, 
                              [-1, 1, 1, 3])
            xyz_crops = xyz_crops - centers
            x = self.conv_layer(xyz_crops, 32, 3, name='conv1_1')
            x = self.conv_layer(x, 32, 3, name='conv1_2')
            conv1_out = x
            x = self.max_pool(x, [2, 2], 'pool1')

            x = self.conv_layer(x, 32, 3, name='conv2_1')
            x = self.conv_layer(x, 32, 3, name='conv2_2')
            conv2_out = x
            x = self.max_pool(x, [2, 2], 'pool2')

            x = self.conv_layer(x, 64, 3, name='conv3_1')
            x = self.conv_layer(x, 64, 3, name='conv3_2')
            conv3_out = x
            x = self.max_pool(x, [2, 2], 'pool3')

            x = self.conv_layer(x, 128, 3, name='conv4_1')
            x = self.conv_layer(x, 128, 3, name='conv4_2')
            conv4_out = x
            conv4_out_flat = tf.contrib.layers.flatten(conv4_out)
            
            roi_feat = tf.image.crop_and_resize(
                              image=backbone_out,
                              boxes=bbox,
                              box_ind=box_indices,
                              crop_size=roi_feature_size)
            roi_feat_flat = tf.contrib.layers.flatten(roi_feat)

            feat_fuse = tf.concat([conv4_out_flat,
                                  roi_feat_flat],
                                  axis=1)

            rot_fc1 = self.fc_layer(
                              feat_fuse, 256,
                              name='roi_rot_fc1')
            rot_fc2 = self.fc_layer(
                              rot_fc1, 256,
                              name='roi_rot_fc2')
            rot_vect = tf.nn.softmax(self.fc_layer(
                              rot_fc2, len(rot_bins),
                              name='rot_vect',
                              linear=True))
            rotation = self.prob_to_rot(
                              rot_vect,
                              rot_bins,
                              name='rotation')
           
            class_fc1 = self.fc_layer(feat_fuse, 256,
                                    name='class_fc1')
            class_fc2 = self.fc_layer(class_fc1, 256,
                                    name='class_fc2')
            class_prob = tf.nn.softmax(self.fc_layer(
                                    class_fc2, 2,
                                    name='class_prob',
                                    linear=True))
                 
        with tf.variable_scope('mask_lidar',
                 reuse=tf.AUTO_REUSE):
            pheight, pwidth = roi_point_cloud_size
            reduce_conv4 = self.conv_layer(conv4_out, 
                               64, 3, name='conv4_reduce')
            upsample_conv4 = tf.image.resize_images(
                                 reduce_conv4,
                                 [pheight//4, pwidth//4])
            concat_conv3 = tf.concat([upsample_conv4,
                                 conv3_out], axis=3)
            reduce_conv3 = self.conv_layer(concat_conv3,
                                 32, 3, name='conv3_reduce')
            upsample_conv3 = tf.image.resize_images(
                                 reduce_conv3,
                                 [pheight//2, pwidth//2])
            concat_conv2 = tf.concat([upsample_conv3,
                                 conv2_out], axis=3)
            reduce_conv2 = self.conv_layer(concat_conv2,
                                 32, 3, name='conv2_reduce')
            upsample_conv2 = tf.image.resize_images(
                                 reduce_conv2,
                                 [pheight, pwidth])
            concat_conv1 = tf.concat([upsample_conv2,
                                 conv1_out], axis=3)
            reduce_conv1 = self.conv_layer(concat_conv1,
                                 2, 3, name='conv1_reduce',
                                 linear=True)
            mask_prob = tf.nn.softmax(reduce_conv1)

        return class_prob, rotation, rot_vect, mask_prob
