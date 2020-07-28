import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
from reader import Reader
from source.anchor_filter import AnchorFilter
import logging
import random
import time
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class VS3D(object):

    def __init__(self, kitti, 
                 train_set, val_set,
                 is_training=False, 
                 mini_batch_size=[1024, 128]):

        self.reader = Reader(kitti)
        self.anchor_filter = AnchorFilter()
        self.is_training = is_training
        self.mini_batch_size = mini_batch_size
        with open(train_set) as f:
            indices = f.readlines()
            self.train_indices = \
                [index.strip() for index in indices]
            self.train_indices.sort()

        with open(val_set) as f:
            indices = f.readlines()
            self.val_indices = \
                [index.strip() for index in indices]
            self.val_indices.sort()
        self.endpoint = self.build()
        return

    def random_keep(self, mask, num_keep):
        curr_num = tf.reduce_sum(
                         tf.cast(mask, tf.float32))
        keep_ratio = tf.divide(num_keep,
                         curr_num + 1.0)
        rand_select = tf.random.uniform(
                         shape=tf.shape(mask),
                         minval=0,
                         maxval=1)
        keep = tf.less(rand_select, keep_ratio)
        mask = tf.logical_and(mask, keep)
        return mask

    def balance_pos_neg(self, scores, num_keep,
                        pos_thres=0.7,
                        neg_thres=0.4):
        num_keep_pos = num_keep // 2
        num_keep_neg = num_keep // 2
        all_pos = tf.greater(scores, pos_thres)
        select_pos = self.random_keep(all_pos, 
                                      num_keep_pos)
        all_neg = tf.less(scores, neg_thres)
        select_neg = self.random_keep(all_neg,
                                      num_keep_neg)
        select = tf.logical_or(select_pos, select_neg)
        return select

    def mask_out(self, mask, tensors):
        masked_tensors = [tf.boolean_mask(tensor, mask) \
                          for tensor in tensors]
        return masked_tensors

    def build(self):

        endpoint = {}
        placeholder = {}

        placeholder['sphere_map'] = tf.placeholder(
                              shape=[64, 512, 5],
                              dtype=tf.float32)
        placeholder['input_image'] = tf.placeholder(
                              shape=[384, 1248, 3],
                              dtype=tf.float32)
        placeholder['image_size'] = tf.placeholder(
                              shape=[2], 
                              dtype=tf.float32)
        placeholder['plane'] = tf.placeholder(
                              shape=[4],
                              dtype=tf.float32)
        placeholder['velo_to_cam'] = tf.placeholder(
                              shape=[4, 4],
                              dtype=tf.float32)
        placeholder['cam_to_img'] = tf.placeholder(
                              shape=[3, 4],
                              dtype=tf.float32)
        placeholder['cam_to_velo'] = tf.placeholder(
                              shape=[4, 4],
                              dtype=tf.float32)
        xyz, ranges, density = tf.split(
                              placeholder['sphere_map'],
                              [3, 1, 1], axis=-1)
        anchor_centers, scores_init = \
            self.anchor_filter.filt(xyz, placeholder['plane'], 
                              placeholder['cam_to_velo']) 

        endpoint['anchor_centers'] = anchor_centers
        endpoint['scores_init'] = scores_init

        mask = tf.greater(scores_init, 0.9 if self.is_training else 0.8)
        if self.is_training:
            mask = self.random_keep(mask, 
                       self.mini_batch_size[0])
      
        bottom_centers, rotation, class_prob, full_prob = \
            self.anchor_filter.filt_image(
                              placeholder['input_image'],
                              placeholder['plane'],
                              placeholder['cam_to_img'],
                              placeholder['image_size'],
                              mask)

        if self.is_training:
            mask_balance = self.balance_pos_neg(
                               class_prob, 
                               self.mini_batch_size[1])
            bottom_centers, rotation, class_prob, full_prob = \
                self.mask_out(mask_balance, 
                    [bottom_centers, rotation, class_prob, full_prob])


        endpoint['bottom_centers'] = bottom_centers
        endpoint['rotation'] = rotation
        endpoint['class_prob'] = class_prob
        endpoint['full_prob'] = full_prob
       
        [_, rotation_lidar, rot_vect_lidar,
        class_prob_lidar, mask_prob_lidar] = \
            self.anchor_filter.filt_lidar(
                              placeholder['sphere_map'],
                              placeholder['plane'],
                              placeholder['cam_to_velo'],
                              placeholder['velo_to_cam'],
                              mask)

        if self.is_training:
            [_, rotation_lidar, rot_vect_lidar,
                class_prob_lidar, mask_prob_lidar] = \
                    self.mask_out(mask_balance,
                        [_, rotation_lidar, rot_vect_lidar,
                            class_prob_lidar, mask_prob_lidar])
                    

        endpoint['rotation_lidar'] = rotation_lidar
        endpoint['class_prob_lidar'] = class_prob_lidar
        endpoint['rot_vect_lidar'] = rot_vect_lidar
        endpoint['mask_prob_lidar'] = mask_prob_lidar

     
        bottom_centers_aligned, point_cloud_density = \
            self.anchor_filter.points_alignment(
                              xyz,
                              bottom_centers,
                              rotation,
                              placeholder['velo_to_cam'],
                              placeholder['cam_to_velo'])
        endpoint['bottom_centers_aligned'] = bottom_centers_aligned
        endpoint['point_cloud_density'] = point_cloud_density
      

        bottom_centers_aligned_lidar, point_cloud_density_lidar = \
            self.anchor_filter.points_alignment(
                              xyz,
                              bottom_centers,
                              rotation_lidar,
                              placeholder['velo_to_cam'],
                              placeholder['cam_to_velo'])
        endpoint['bottom_centers_aligned_lidar'] = bottom_centers_aligned_lidar
        endpoint['point_cloud_density_lidar'] = point_cloud_density_lidar

      
        rotation_aligned = self.anchor_filter.rotation_align(
                              xyz,
                              bottom_centers,
                              rotation,
                              placeholder['velo_to_cam'],
                              placeholder['cam_to_velo'])
        endpoint['rotation_aligned'] = rotation_aligned

        instance_points, instance_mask = \
            self.anchor_filter.instance_mask(
                              xyz,
                              bottom_centers_aligned,
                              rotation,
                              placeholder['velo_to_cam'],
                              placeholder['cam_to_velo'])
        endpoint['instance_points'] = instance_points
        endpoint['instance_mask'] = instance_mask
      

        instance_points_lidar, instance_mask_lidar = \
            self.anchor_filter.instance_mask(
                              xyz,
                              bottom_centers_aligned_lidar,
                              rotation_lidar,
                              placeholder['velo_to_cam'],
                              placeholder['cam_to_velo'])
        endpoint['instance_points_lidar'] = instance_points_lidar
        endpoint['instance_mask_lidar'] = instance_mask_lidar
        endpoint['mask_prob_lidar'] = mask_prob_lidar

       
        nms_indices = self.anchor_filter.nms_image(
                              bottom_centers,
                              rotation,
                              tf.minimum(class_prob, full_prob),
                              placeholder['cam_to_img'],
                              placeholder['image_size'])
        endpoint['nms_indices'] = nms_indices
      
        nms_indices_lidar = self.anchor_filter.nms_image(
                              bottom_centers,
                              rotation_lidar,
                              class_prob_lidar,
                              placeholder['cam_to_img'],
                              placeholder['image_size'])
        endpoint['nms_indices_lidar'] = nms_indices_lidar
      
        class_loss, rot_loss, mask_loss, rot_error = \
            self.anchor_filter.build_loss(
                rotation, rot_vect_lidar, rotation_lidar, 
                tf.minimum(class_prob, full_prob), 
                class_prob_lidar, instance_mask, mask_prob_lidar)

        endpoint['class_loss'] = class_loss
        endpoint['rot_loss'] = rot_loss
        endpoint['mask_loss'] = mask_loss
        endpoint['rot_error'] = rot_error
        
        self.placeholder = placeholder
        
        return endpoint

    def to_kitti_line(self, bbox, center, 
                      size, rotation, score): 
        kitti_line = 'Car -1 -1 -10 ' + \
                     '{:.2f} {:.2f} {:.2f} {:.2f} '.format(
                      bbox[0], bbox[1], bbox[2], bbox[3]) + \
                     '{:.2f} {:.2f} {:.2f} '.format(
                      size[0], size[1], size[2]) + \
                     '{:.2f} {:.2f} {:.2f} '.format(
                      center[0], center[1], center[2]) + \
                     '{:.2f} {:.2f} \n'.format(
                      rotation, score)
        return kitti_line

    def to_bbox(self, center, 
                size, rotation, 
                cam_to_img, image_size):        
        R = np.array([[+np.cos(rotation), 0, +np.sin(rotation)],
                      [                0, 1,                  0],
                      [-np.sin(rotation), 0, +np.cos(rotation)]],
                      dtype=np.float32)
        h, w, l = size
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        corners = np.dot(R, [x_corners, y_corners, z_corners])
        corners = corners + center.reshape((3, 1))
        projection = np.dot(cam_to_img, np.vstack([corners, 
                                np.ones(8, dtype=np.float32)]))
        projection = (projection / projection[2])[:2]
        left   = max(np.amin(projection[0]), 0)
        right  = min(np.amax(projection[0]), image_size[1])
        top    = max(np.amin(projection[1]), 0)
        bottom = min(np.amax(projection[1]), image_size[0])
        return [left, top, right, bottom]

    def train(self, 
              model_image, 
              model_lidar=None, 
              save_dir='./runs/weights', 
              steps=160000, 
              learning_rate_init=1e-4,
              l2_weight=1e-5,
              clip_grads=False,
              clip_grads_norm=2.0,
              display_step=200,
              save_step=2000):
        class_loss = self.endpoint['class_loss']
        rot_loss = self.endpoint['rot_loss'] * 5.0
        mask_loss = self.endpoint['mask_loss'] * 2.0
        rot_error = self.endpoint['rot_error']
        global_step = tf.placeholder(shape=(), dtype=tf.int32)
        learning_rate = tf.train.exponential_decay(
                            learning_rate_init, global_step,
                                120000, 0.2, staircase=False)
        weight_loss = [tf.nn.l2_loss(var) for var \
                           in tf.trainable_variables()]
        weight_loss = tf.reduce_sum(weight_loss) * l2_weight
        total_loss = weight_loss + class_loss + rot_loss + mask_loss

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        all_vars = tf.get_collection_ref(
                           tf.GraphKeys.GLOBAL_VARIABLES) 
        var_list_image = \
                [var for var in all_vars if "lidar" not in var.name]
        var_list_lidar = \
                [var for var in all_vars if "lidar" in var.name]
        
        if clip_grads:
            grads_and_vars = opt.compute_gradients(total_loss,
                                     var_list_lidar)
            grads, tvars = zip(*grads_and_vars)
            clipped_grads, norm = tf.clip_by_global_norm(
                                      grads, clip_grads_norm)
            grads_and_vars = zip(clipped_grads, tvars)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                train_op = opt.apply_gradients(grads_and_vars)
        else:
            train_op = tf.train.AdamOptimizer(
                learning_rate=learning_rate
                    ).minimize(total_loss, var_list=var_list_lidar)
     
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver_image = tf.train.Saver(var_list=var_list_image)
            saver_lidar = tf.train.Saver(var_list=var_list_lidar)
            saver_image.restore(sess, model_image)

            if model_lidar:
                saver_lidar.restore(sess, model_lidar)

            rot_loss_np_list = []
            class_loss_np_list = []
            mask_loss_np_list = []
            rot_error_np_list = []

            for step in range(steps):
                index = random.choice(self.train_indices)
                data = self.reader.data[index]
                sphere_map = np.load(open(
                                 data['sphere_path'], 'rb'))
                image_pil = Image.open(
                                 data['image_path'])
                width, height = image_pil.size
                image_size = np.array([height, width],
                                 dtype=np.float32)
                image_np = np.array(image_pil.resize((1248, 384)),
                                 dtype=np.float32)
                cam_to_img = data['P2']
                plane = data['plane']
                velo_to_cam = np.dot(data['R0'], data['Tr'])
                cam_to_velo = np.linalg.inv(velo_to_cam)
                placeholder = self.placeholder
             
                _, weight_loss_np, class_loss_np, \
                   rot_loss_np, mask_loss_np, rot_error_np, debug_np = \
                              sess.run([train_op, weight_loss,
                                        class_loss, rot_loss, 
                                        mask_loss, rot_error, 
                                        tf.get_collection('debug')],
                              feed_dict={
                                  placeholder['sphere_map']: sphere_map,
                                  placeholder['plane']: plane,
                                  placeholder['velo_to_cam']: velo_to_cam,
                                  placeholder['cam_to_velo']: cam_to_velo,
                                  placeholder['input_image']: image_np,
                                  placeholder['image_size']: image_size,
                                  placeholder['cam_to_img']: cam_to_img,
                                  global_step: step})

                rot_loss_np_list.append(rot_loss_np)
                class_loss_np_list.append(class_loss_np)
                mask_loss_np_list.append(mask_loss_np)
                rot_error_np_list.append(rot_error_np)

                if step % display_step == 0:
                    logging.info(
                        'Step: {} / {}, '.format(step, steps) + \
                        'Loss Weight: {:.3f}, '.format(weight_loss_np) + \
                        'Class: {:.3f}, '.format(np.mean(class_loss_np_list)) + \
                        'Rotation: {:.3f}, '.format(np.mean(rot_loss_np_list)) + \
                        'Mask: {:.3f}, '.format(np.mean(mask_loss_np_list)) + \
                        'Rot Error: {:.3f}'.format(np.mean(rot_error_np_list)))

                    rot_loss_np_list = []
                    class_loss_np_list = []
                    mask_loss_np_list = []
                    rot_error_np_list= []

                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                if step % save_step == 0:
                    saver_lidar.save(sess, os.path.join(save_dir, 
                        'model_lidar_{}'.format(str(step).zfill(6))))
              
    def run(self, score_thres=0.05, 
            density_thres=0.1, save_dir=None,
            image_model=None, lidar_model=None, 
            return_pred=False, max_pred_frames=np.inf):
      
        with tf.Session() as sess:
            all_vars = tf.get_collection_ref(
                           tf.GraphKeys.GLOBAL_VARIABLES)
            var_list_image = \
                    [var for var in all_vars if "lidar" not in var.name]
            var_list_lidar = \
                    [var for var in all_vars if "lidar" in var.name]
            assert len(all_vars) == len(var_list_image + var_list_lidar)

            if image_model and not lidar_model:
                saver = tf.train.Saver(var_list=var_list_image)
                saver.restore(sess, image_model)

                rotation_tf = self.endpoint['rotation']
                centers_tf = self.endpoint['bottom_centers']
                centers_aligned_tf = self.endpoint['bottom_centers_aligned']
                scores_tf = tf.minimum(self.endpoint['class_prob'], 
                                       self.endpoint['full_prob'])
                nms_indices_tf = self.endpoint['nms_indices']
                point_cloud_density_tf = self.endpoint['point_cloud_density']
                instance_points_tf = self.endpoint['instance_points']
                instance_mask_tf = self.endpoint['instance_mask']

            elif not image_model and lidar_model:
                saver = tf.train.Saver(var_list=var_list_lidar)
                saver.restore(sess, lidar_model)

                rotation_tf = self.endpoint['rotation_lidar']
                centers_tf = self.endpoint['bottom_centers']
                centers_aligned_tf = self.endpoint['bottom_centers_aligned_lidar']
                scores_tf = self.endpoint['class_prob_lidar']
                nms_indices_tf = self.endpoint['nms_indices_lidar']
                point_cloud_density_tf = self.endpoint['point_cloud_density_lidar']
                instance_points_tf = self.endpoint['instance_points_lidar']
                instance_mask_tf = self.endpoint['instance_mask_lidar']

            elif image_model and lidar_model:
                saver = tf.train.Saver(var_list=var_list_image)
                saver.restore(sess, image_model)
                saver = tf.train.Saver(var_list=var_list_lidar)
                saver.restore(sess, lidar_model)
            else:
                raise Exception('Image or LiDAR model must be provided!')

            out_tf = [rotation_tf, centers_tf, centers_aligned_tf,
                      scores_tf, nms_indices_tf, point_cloud_density_tf,
                      instance_points_tf, instance_mask_tf]
            
            bbox_list = []
            mask_list = []
            index_list = []
            
            total_time = []
            for iindex, index in enumerate(self.val_indices):
                if iindex == max_pred_frames:
                    break
                logging.info('Inference {}'.format(index))
                data = self.reader.data[index]
                sphere_map = np.load(open(
                                 data['sphere_path'], 'rb'))
                image_pil = Image.open(
                                 data['image_path'])
                width, height = image_pil.size
                image_size = np.array([height, width],
                                 dtype=np.float32)
                image_np = np.array(image_pil.resize((1248, 384)),
                                 dtype=np.float32)
                cam_to_img = data['P2']
                plane = data['plane']
                velo_to_cam = np.dot(data['R0'], data['Tr'])
                cam_to_velo = np.linalg.inv(velo_to_cam)
                placeholder = self.placeholder
                start_time = time.time()
                out_np = sess.run(out_tf, 
                              feed_dict={
                                  placeholder['sphere_map']: sphere_map,
                                  placeholder['plane']: plane,
                                  placeholder['velo_to_cam']: velo_to_cam,
                                  placeholder['cam_to_velo']: cam_to_velo,
                                  placeholder['input_image']: image_np,
                                  placeholder['image_size']: image_size,
                                  placeholder['cam_to_img']: cam_to_img})
                total_time.append(time.time() - start_time)
                [rotation_np, centers_np, centers_aligned_np,
                 scores_np, nms_indices_np, point_cloud_density_np,
                 instance_points_np, instance_mask_np] = out_np
                if iindex % 300 == 0:
                    logging.info('Forward time: {:.3f}s, STD: {:.3f}s'.format(np.mean(total_time), np.std(total_time)))
                    total_time = []

                kitti_lines = []
                instance_points_masked = []
            
                for aind in nms_indices_np:
               
                    score = scores_np[aind]
                    density = point_cloud_density_np[aind]
                    if score < score_thres or density < density_thres:
                        continue
                    bbox = self.to_bbox(center=centers_aligned_np[aind],
                                        size=[1.45, 1.55, 4.00],
                                        rotation=rotation_np[aind],
                                        cam_to_img=cam_to_img,
                                        image_size=image_size)
                    kitti_line = self.to_kitti_line(
                                     bbox=bbox,
                                     center=centers_aligned_np[aind],
                                     size=[1.45, 1.55, 4.00],
                                     rotation=rotation_np[aind],
                                     score=score)
                    kitti_lines.append(kitti_line)
                    instance_points_masked.append(
                        instance_points_np[aind] * instance_mask_np[aind])

                if not os.path.exists(save_dir):
                    os.makedirs(os.path.join(save_dir, 'bbox'))
                    os.makedirs(os.path.join(save_dir, 'mask'))
             
                with open(os.path.join(save_dir, 
                                       'bbox', 
                                       index+'.txt'), 'w') as f:
                    f.writelines(kitti_lines)
                    f.close()

                with open(os.path.join(save_dir, 
                                      'mask', 
                                       index+'.npy'), 'wb') as f:
                    np.save(f, instance_points_masked)
                    f.close()

                if return_pred:
                    bbox_list.append(kitti_lines)
                    mask_list.append(instance_points_masked)
                    index_list.append(index)

            if return_pred:
                return bbox_list, mask_list, index_list, self.reader

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', 
                        type=str, 
                        required=True, 
                        help='train or evaluate')
    parser.add_argument('--teacher_model', 
                        type=str, 
                        default='../data/pretrained/teacher/iter_158000',
                        help='required in training.')
    parser.add_argument('--student_model', 
                        type=str, 
                        default=None,
                        help='required in testing and optional in training.')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0',
                        help='GPU to use.')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  
    if args.mode == 'train':
        vs3d = VS3D(kitti='../data/kitti/training',
                    train_set='../data/kitti/train.txt',
                    val_set='../data/kitti/val.txt',
                    is_training=True)
        vs3d.train(model_image=args.teacher_model)

    elif args.mode == 'evaluate':
        vs3d = VS3D(kitti='../data/kitti/training',
                    train_set='../data/kitti/train.txt',
                    val_set='../data/kitti/val.txt',
                    is_training=False)

        vs3d.run(save_dir='../output',
                 lidar_model=args.student_model)

    else:
        raise NotImplementedError