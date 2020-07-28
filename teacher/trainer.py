import numpy as np
import tensorflow as tf
import json
from model import Model
from random import choice
import utils
import logging
import argparse
import os
from PIL import Image, ImageEnhance, ImageFilter
logging.basicConfig(level=logging.INFO)

class Trainer(Model):
 
    def __init__(self, crops_dir, rots_dir, classes, save_dir='./models',
                 input_shape=[64, 64], batch_size=1, is_training=True):
        Model.__init__(self, input_shape, batch_size)
        self.crops_dir = crops_dir
        self.rots_dir = rots_dir
        self.rots_ground_truth = {}
        self.num_samples = {}
        self.keys = {}
        self.classes = classes
        self.save_dir = save_dir
        for cls in classes:
            json_path = os.path.join(rots_dir, cls+'.json')
            self.rots_ground_truth[cls] = \
            json.load(open(json_path))
            self.num_samples[cls] = len(self.rots_ground_truth[cls])
            self.keys[cls] = self.rots_ground_truth[cls].keys()
            logging.info('Num Samples {}: {}'.format(cls, self.num_samples[cls]))
        return

    def rand(self, loc=1.0, scale=0.7, lower=0.1):
        rand = np.random.normal(size=1, loc=loc, scale=scale)
        return max(rand, lower)

    def rot_to_prob(self, rot):
        rot_bins = self.rot_bins
        distance = np.minimum(np.minimum(np.abs(rot - rot_bins),
                              np.abs(rot - rot_bins - 2 * np.pi)),
                              np.abs(rot - rot_bins + 2 * np.pi))
        var = 0.1
        prob = np.exp(-np.square(distance) / var)
        prob = prob / np.sum(prob)
        return prob

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def augment(self, image_np, 
                rot, is_pos_cls, 
                margin_rate=0.4, 
                ori_margin=0.2,
                iou_thres=0.5):
        height, width = image_np.shape[:2]
        left = int(np.random.uniform(0, width * margin_rate))
        right = int(np.random.uniform((1-margin_rate) * width, width)) - 1
        top = int(np.random.uniform(0, height * margin_rate))
        bottom = int(np.random.uniform((1-margin_rate) * height, height)) - 1

        a_l = ori_margin
        a_r = 1.0 - ori_margin
        a_t = ori_margin
        a_b = 1.0 - ori_margin

        b_l = left * 1.0 / width
        b_r = right * 1.0 / width
        b_t = top * 1.0 / height
        b_b = bottom * 1.0 / height

        iou = self.iou([a_l, a_t, a_r, a_b],
                       [b_l, b_t, b_r, b_b])

        full = 1.0 if iou > iou_thres else 0.0
        full_prob = [1.0 - full, full]

        image_obj = Image.fromarray(image_np[top:bottom, left:right])
        image_obj = image_obj.resize(self.input_shape, 
                                     Image.BILINEAR)
        if np.random.uniform() < 0.5:
            image_obj = image_obj.filter(ImageFilter.BLUR)

        image_obj = ImageEnhance.Color(image_obj).enhance(self.rand())
        image_obj = ImageEnhance.Brightness(image_obj).enhance(self.rand())
        image_obj = ImageEnhance.Contrast(image_obj).enhance(self.rand())
        image_np = np.array(image_obj)
        if len(image_np.shape) > 2:
            image_np = np.mean(image_np, axis=-1)
        image_np = image_np * 1.0 / (np.amax(image_np) + 1e-6)
        if np.random.uniform() < 0.5:
            image_np = np.fliplr(image_np)
            rot = np.pi - rot
            rot = rot - 2 * np.pi if rot > np.pi else rot
        image_np = np.reshape(image_np, [self.input_shape[0], 
                   self.input_shape[1], 1]).astype(np.float32)
        rot_prob = self.rot_to_prob(rot)

        pos = float(is_pos_cls and iou > iou_thres)
        cls_prob = np.array([1.0 - pos, pos], dtype=np.float32)
        return image_np, rot_prob.astype(np.float32), rot, cls_prob, full_prob

    def load_data(self, pos_cls='car'):
        cls = choice(self.classes)
        file_name = choice(list(self.keys[cls]))
        image_path = os.path.join(self.crops_dir, cls, file_name+'.png')
        rot = self.rots_ground_truth[cls][file_name]
        image_np = np.array(Image.open(image_path))
        return self.augment(image_np, rot, pos_cls in cls)
       
    def load_data_batch(self):
        images = []
        rots_prob = []
        rots = []
        cls_prob = []
        full_prob = []
        for _ in range(self.batch_size):
            data = self.load_data()
            images.append(data[0])
            rots_prob.append(data[1])
            rots.append(data[2])
            cls_prob.append(data[3])
            full_prob.append(data[4])
        return images, rots_prob, rots, cls_prob, full_prob

    def train(self, iterations, global_step=0, pretrained_model=None, learning_rate=5e-5, 
              display_iter=50, eval_iter=50, save_iter=2000, decay=1e-6):
        batch_size = self.batch_size
        learning_rate_tf = tf.placeholder(dtype=tf.float32,
                               shape=())
        tf_ground_truth = tf.placeholder(dtype=tf.float32,
                          shape=[batch_size, len(self.rot_bins)])
        tf_cls_ground_truth = tf.placeholder(dtype=tf.float32,
                          shape=[batch_size, 2])
        tf_full_ground_truth = tf.placeholder(dtype=tf.float32,
                          shape=[batch_size, 2])
        _, tf_mask = tf.split(tf_cls_ground_truth, [1, 1], axis=1) 
        tf_mask_r = tf.reshape(tf_mask, (-1,))
        tf_rot_ground_truth = tf.placeholder(dtype=tf.float32,
                              shape=[batch_size])
        rot_error = tf.reduce_sum(tf_mask_r * tf.abs(self.rotation - tf_rot_ground_truth)) / \
                              (tf.reduce_sum(tf_mask) + 1e-6)
        cross_ent_rot = tf.reduce_mean(utils.cross_entropy(self.rotation_prob,
                               tf_ground_truth), axis=1)
        rot_loss = tf.reduce_sum(tf_mask_r * cross_ent_rot) / \
                              (tf.reduce_sum(tf_mask) + 1e-6)
        cls_loss = tf.reduce_mean(utils.cross_entropy(self.class_prob,
                                  tf_cls_ground_truth))
        cross_ent_full = tf.reduce_mean(tf_mask * utils.cross_entropy(
                               self.full_prob,
                               tf_full_ground_truth), axis=1) * \
                         tf.reduce_sum(tf_full_ground_truth, axis=1)
        full_loss = tf.reduce_sum(cross_ent_full) / \
                         (tf.reduce_sum(tf_mask) + 1e-6)

        weight_loss = [tf.nn.l2_loss(var) for var in tf.trainable_variables()]
        weight_loss = tf.reduce_sum(weight_loss) * decay 
        loss = rot_loss + weight_loss + cls_loss * 0.5 + full_loss
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_tf).minimize(loss)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run([tf.global_variables_initializer()])

            if pretrained_model:
                var_list = [var for var in tf.trainable_variables() if 'beta' not in var.name]
                saver_part = tf.train.Saver(var_list=var_list)
                saver_part.restore(sess, pretrained_model)
           
            for iteration in range(global_step, iterations):
                if iteration * 1.0 / iterations > 0.8:
                    learning_rate_feed = learning_rate * 0.1
                else:
                    learning_rate_feed = learning_rate
                np_image, np_rot_prob, np_rot, np_cls_prob, np_full_prob = self.load_data_batch()
              
                _, np_rot_loss, np_cls_loss, np_full_loss, np_weight_loss, np_rot_error = \
                                                 sess.run([train_op, rot_loss, cls_loss, full_loss, weight_loss, rot_error],
                                                 feed_dict={self.input_image: np_image,
                                                 tf_ground_truth: np_rot_prob,
                                                 tf_rot_ground_truth: np_rot,
                                                 tf_cls_ground_truth: np_cls_prob,
                                                 tf_full_ground_truth: np_full_prob,
                                                 learning_rate_tf: learning_rate_feed})
               
                if iteration % display_iter == 0:
       
                    logging.info('iteration: {}/{}, loss rotation: {:.3f}, class: {:.3f}, full: {:.3f}, decay: {:.3f}, error: {:.3f}'.format(
                                 iteration, iterations, np_rot_loss, np_cls_loss, np_full_loss, np_weight_loss, np_rot_error))
                
                if iteration % save_iter == 0:
                    save_path = os.path.join(self.save_dir, 
                                'iter_{}'.format(str(iteration).zfill(6)))
                    saver.save(sess, save_path)
        return

if __name__ == '__main__':
    EXPORT = False
    if EXPORT:
        batch_size = 1
        learning_rate = 1e-12
        is_training = False
    else:
        batch_size = 128
        learning_rate = 1e-5
        is_training = True
  
    classes = ['car_imagenet', 
               'car_nyc3d', 
               'all_imagenet',
               'all_imagenet',
               'all_imagenet',
               'all_imagenet']

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model', default=None)
    parser.add_argument('--global_step', default='0')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    trainer = Trainer(crops_dir='./data/Crops',
                      rots_dir='./data/Viewpoints',
                      classes=classes,
                      batch_size=batch_size,
                      is_training=is_training)
    trainer.train(160000, global_step=int(args.global_step), 
                  pretrained_model=args.model, learning_rate=learning_rate)
    
