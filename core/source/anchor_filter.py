import numpy as np
import tensorflow as tf
from source.model import Model
from source.model_lidar import ModelLiDAR
import utils
from utils import get_normal_map

class AnchorFilter(object):

    def __init__(self, 
                 area_extend=[[-35, 35], 
                              [0, 70]],
                 area_stride=0.2):
        x_start, x_end = area_extend[0]
        z_start, z_end = area_extend[1]
        x_grid, z_grid = np.meshgrid(
                             np.arange(x_start, 
                                 x_end,
                                 step=area_stride),
                             np.arange(z_start,
                                 z_end,
                                 step=area_stride))
        self.x_grid = tf.constant(x_grid,
                          dtype=tf.float32)
        self.z_grid = tf.constant(z_grid, 
                          dtype=tf.float32)
        return

    def get_phi(self, points):
        sine = points[2] / tf.math.sqrt(points[0]**2 + 
                               points[1]**2 + 
                               points[2]**2)
        return tf.math.asin(sine)

    def get_theta(self, points):
        sine = points[1] / tf.math.sqrt(points[0]**2 + 
                               points[1]**2)
        return tf.math.asin(sine)

    def get_anchor_centers(self, 
                           plane,
                           y_offset=-0.75):
        x_grid = self.x_grid
        z_grid = self.z_grid
        a, b, c, d = plane[0], plane[1], plane[2], plane[3]
        y_grid = -(a * x_grid + c * z_grid + d) / b
        num_x, num_z = y_grid.get_shape().as_list()
        num_xz = num_x * num_z
        xyz_flatten = tf.concat([
                          tf.reshape(x_grid, (1, num_xz)),
                          tf.reshape(y_grid, (1, num_xz)) + \
                              y_offset,
                          tf.reshape(z_grid, (1, num_xz)),
                          tf.ones(shape=(1, num_xz),
                              dtype=tf.float32)],
                          axis=0)
        return xyz_flatten

    def compute_corners(self, 
                        dimensions, 
                        alpha):
        h, w, l = tf.split(tf.reshape(dimensions, 
                               (-1, 1, 3)), 
                      [1, 1, 1], axis=2)
        unrot = tf.concat([tf.concat([l/2, l/2, -l/2, -l/2, 
                                      l/2, l/2, -l/2, -l/2], 
                                      axis=2),
                           tf.concat([w/2, -w/2, -w/2, w/2, 
                                      w/2, -w/2, -w/2, w/2], 
                                      axis=2)], 
                           axis=1)
        alpha_r = tf.reshape(alpha, (-1, 1))
        zeros = tf.expand_dims(
                    tf.zeros(tf.shape(alpha_r), 
                        dtype=tf.float32), 2)

        x_rot_vect = tf.reshape(tf.concat([tf.cos(alpha_r), 
                                           tf.sin(alpha_r)], 
                                           axis=1), 
                               (-1, 2, 1))
        x_rot = tf.reduce_sum(unrot * x_rot_vect, 
                              axis=1, keepdims=True)

        z_rot_vect = tf.reshape(tf.concat([-tf.sin(alpha_r), 
                                           tf.cos(alpha_r)], 
                                           axis=1), 
                               (-1, 2, 1))
        z_rot = tf.reduce_sum(unrot * z_rot_vect, 
                              axis=1, keepdims=True)
        minus_h = tf.add(zeros, -tf.reshape(h, (-1, 1, 1)))
        y_rot = tf.concat([zeros, zeros, zeros, zeros, 
                           minus_h, minus_h, minus_h, minus_h], axis=2)
        corners_rot = tf.concat([x_rot, y_rot, z_rot], axis=1)
        return tf.stop_gradient(corners_rot)

    def corners_to_bbox(self, 
                        global_corners, 
                        calib, 
                        image_size, 
                        tf_order=True):        
        global_corners_expand = tf.reshape(
                                    global_corners, 
                                    (-1, 3, 8))
        xz = tf.reduce_sum(tf.reshape(calib[0, :3], 
                                     (1, 3, 1)) * \
                           global_corners_expand, axis=1) + \
                           calib[0, 3]
        yz = tf.reduce_sum(tf.reshape(calib[1, :3], 
                                     (1, 3, 1)) * \
                           global_corners_expand, axis=1) + \
                           calib[1, 3]
        z = tf.reduce_sum(tf.reshape(calib[2, :3], 
                                     (1, 3, 1)) * \
                           global_corners_expand, axis=1) + \
                           calib[2, 3]

        x = tf.clip_by_value(xz / (z * image_size[1] + 1e-5), 0, 1)
        y = tf.clip_by_value(yz / (z * image_size[0] + 1e-5), 0, 1)
        left = tf.reduce_min(x, axis=1, keepdims=True)
        right = tf.reduce_max(x, axis=1, keepdims=True)
        top = tf.reduce_min(y, axis=1, keepdims=True)
        bottom = tf.reduce_max(y, axis=1, keepdims=True)
        if tf_order:
            bbox = tf.concat([top, left, bottom, right], axis=1)
        else:
            bbox = tf.concat([left, top, right, bottom], axis=1)
        return bbox

    def filt(self, 
             xyz_map,
             plane, 
             cam_to_velo, 
             y_offset=-0.4,
             dr_mean=1.5,
             dr_std=0.5):
        x_grid = self.x_grid
        num_x, num_z = x_grid.get_shape().as_list()
        num_xz = num_x * num_z
        xyz_flatten = self.get_anchor_centers(
                          plane, y_offset)
        xyz_velo = tf.matmul(cam_to_velo, 
                       xyz_flatten)
        phi = self.get_phi(xyz_velo)
        phi = tf.reshape(-phi * 400 / 3 + 8, (-1, 1)) / 64
        theta = self.get_theta(xyz_velo)
        theta = tf.reshape(theta * 400 + 256, (-1, 1)) / 512
        boxes = tf.concat([phi - 0.5, theta - 0.5,
                    phi + 0.5, theta + 0.5], axis=1)
        box_indices = tf.constant(
                          np.zeros(num_xz),
                          dtype=tf.int32)
        xyz_map = tf.reshape(xyz_map, (1, 64, 512, 3))
        xyz_crop = tf.image.crop_and_resize(
                       image=xyz_map,
                       boxes=boxes,
                       box_ind=box_indices,
                       crop_size=[1, 1])
        range_anchors = tf.norm(
                            xyz_velo, 
                            axis=0, 
                            keepdims=False)
        range_crops = tf.norm(
                          xyz_crop,
                          axis=3,
                          keepdims=False)
        range_crops = tf.squeeze(range_crops)
        range_diff = range_crops + dr_mean - range_anchors
        scores = tf.math.exp(-range_diff**2 / (2*dr_std**2))
        anchor_centers = tf.transpose(xyz_flatten, [1, 0])

        return anchor_centers, scores

    def filt_shift(self,
                   xyz_map,
                   plane,
                   cam_to_velo,
                   y_offset=-0.4,
                   shift_offset=0.2,
                   dr_mean=1.5,
                   dr_std=0.5):
        centers, scores_ori = self.filt(xyz_map,
                                        plane,
                                        cam_to_velo,
                                        y_offset,
                                        dr_mean,
                                        dr_std)
        _, scores_up = self.filt(xyz_map,
                                 plane,
                                 cam_to_velo,
                                 y_offset-shift_offset,
                                 dr_mean,
                                 dr_std)
        _, scores_down = self.filt(xyz_map,
                                 plane,
                                 cam_to_velo,
                                 y_offset+shift_offset,
                                 dr_mean,
                                 dr_std)
        scores = tf.maximum(
                     tf.maximum(
                         scores_up, scores_down), 
                     scores_ori)
        return centers, scores

    def filt_image(self, 
                   image,
                   plane,
                   cam_to_img, 
                   image_size,
                   mask,
                   cube_size=[1.5, 2.5, 2.5],
                   crop_size=[64, 64],
                   cube_size_second=[1.45, 1.55, 4.00]):
        x_grid = self.x_grid
        num_x, num_z = x_grid.get_shape().as_list()
        num_xz = num_x * num_z
        xyz_flatten = self.get_anchor_centers(
                          plane, y_offset=0.0)[:3]
        xyz = tf.expand_dims(
                  tf.transpose(xyz_flatten, [1, 0]), 2)
        corners = self.compute_corners(
                      dimensions=tf.constant(
                          np.reshape(
                              cube_size, [1, 3]),
                          dtype=tf.float32),
                      alpha=tf.constant(0.0,
                          dtype=tf.float32))
        xyz = tf.boolean_mask(xyz, mask)
        kept_corners = xyz + corners
        bbox = self.corners_to_bbox(
                   kept_corners, 
                   cam_to_img,
                   image_size, 
                   tf_order=True)
        num_bbox = bbox.get_shape().as_list()[0]
        box_indices = tf.zeros_like(bbox[:, 0],
                          dtype=tf.int32)
        image = tf.expand_dims(tf.squeeze(image), 0)
        image = tf.reduce_mean(image, axis=3, 
                               keepdims=True)
        image_crops = tf.image.crop_and_resize(
                          image=image,
                          boxes=bbox,
                          box_ind=box_indices,
                          crop_size=crop_size)
        image_crops = image_crops / (tf.reduce_max(
                          image_crops, 
                          axis=[1, 2], 
                          keepdims=True) + 1e-6)
        image_model = Model(input_image=image_crops,
                          batch_size=None)
        class_prob_image = image_model.class_prob[:, 1]
        rotation_local = image_model.rotation
        bottom_centers = tf.squeeze(xyz, 2)
        rotation = rotation_local + tf.constant(np.pi / 2,
                                        dtype=tf.float32) - \
                   tf.atan2(bottom_centers[:, 2], 
                            bottom_centers[:, 0])

        corners_second = self.compute_corners(
                      dimensions=tf.constant(
                          np.reshape(
                              cube_size_second, 
                              [1, 3]),
                          dtype=tf.float32),
                      alpha=rotation)

        kept_corners_second = xyz + corners_second
        bbox_second = self.corners_to_bbox(
                   kept_corners_second,
                   cam_to_img,
                   image_size,
                   tf_order=True)
        image_crops_second = tf.image.crop_and_resize(
                          image=image,
                          boxes=bbox_second,
                          box_ind=box_indices,
                          crop_size=crop_size)
        image_crops_second = image_crops_second / (tf.reduce_max(
                          image_crops_second,
                          axis=[1, 2],
                          keepdims=True) + 1e-6)
        image_model_second = Model(input_image=image_crops_second,
                          batch_size=None)
    
        rotation_local_second = image_model_second.rotation
        rotation = rotation_local_second + tf.constant(np.pi / 2,
                                        dtype=tf.float32) - \
                   tf.atan2(bottom_centers[:, 2],
                            bottom_centers[:, 0])

        class_prob_image = image_model_second.class_prob[:, 1]
        class_prob = class_prob_image

        full_prob_image = image_model_second.full_prob[:, 1]
        full_prob = full_prob_image

        return bottom_centers, rotation, class_prob, full_prob

    def nms_image(self, 
                  bottom_centers, 
                  rotation, 
                  class_prob,
                  cam_to_img,
                  image_size,
                  cube_size=[1.45, 1.55, 4.00],
                  iou_thres=0.3,
                  max_out=128):
        corners = self.compute_corners(
                      dimensions=tf.constant(
                          np.reshape(cube_size, [1, 3]),
                          dtype=tf.float32),
                      alpha=rotation)
        xyz = tf.expand_dims(bottom_centers, 2)
        global_corners = xyz + corners
        bbox = self.corners_to_bbox(
                   global_corners,
                   cam_to_img,
                   image_size,
                   tf_order=True)
        nms_indices = tf.image.non_max_suppression(
                          bbox, class_prob,
                          max_output_size=max_out,
                          iou_threshold=iou_thres)
        return nms_indices

    def points_inside_cube(self,
                           points,
                           bottom_centers,
                           rotation,
                           cube_size=[1.45, 1.55, 4.00]):
        bottom_centers = tf.expand_dims(
                           bottom_centers, axis=2)
        points_centered = points - bottom_centers
        x, y, z = tf.split(points_centered,
                           [1, 1, 1], axis=1)
        xz = tf.concat([x, z], axis=1)
        rot = tf.reshape(rotation, (-1, 1, 1))
        center_to_front_unit_vect = \
            tf.concat([tf.math.cos(rot),
                       -tf.math.sin(rot)], axis=1)
        center_to_side_unit_vect = \
            tf.concat([tf.math.sin(rot),
                       tf.math.cos(rot)], axis=1)
        proj_front = tf.reduce_sum(
                         xz * center_to_front_unit_vect,
                         axis=1, keepdims=True)
        proj_side =  tf.reduce_sum(
                        xz * center_to_side_unit_vect,
                        axis=1, keepdims=True)
        inside_front = tf.cast(tf.less(tf.abs(proj_front), 
                                   cube_size[2] * 0.5),
                               tf.float32)
        inside_side = tf.cast(tf.less(tf.abs(proj_side), 
                                  cube_size[1] * 0.5),
                              tf.float32)
        inside_bottom = tf.cast(tf.less(y, 0.0), tf.float32)
        inside_mask = inside_front * inside_side * \
                      inside_bottom
        
        return inside_mask, proj_front, proj_side

    def instance_mask(self, 
                      xyz_map,
                      bottom_centers,
                      rotation,
                      velo_to_cam,
                      cam_to_velo,
                      cube_size=[1.45, 1.55, 4.00],
                      crop_size=[32, 32]):
        corners = self.compute_corners(
                      dimensions=tf.constant(
                          np.reshape(cube_size, [1, 3]),
                          dtype=tf.float32),
                      alpha=rotation)
        xyz = tf.expand_dims(bottom_centers, 2)
        global_corners = xyz + corners
        bbox, box_indices = self.corners_to_bbox_sphere(
                          global_corners, cam_to_velo)
        xyz_map = tf.concat([tf.reshape(xyz_map,
                                 (1, 64, 512, 3)),
                             tf.ones(shape=(1, 64, 512, 1),
                                 dtype=tf.float32)], axis=3)
        xyz_map = tf.linalg.matmul(
                      xyz_map,
                      tf.tile(
                          tf.reshape(
                              tf.transpose(
                                  velo_to_cam, [1, 0]),
                              (1, 1, 4, 4)),
                          [1, 64, 1, 1]))

        xyz_map, _ = tf.split(xyz_map, [3, 1], axis=3)
        xyz_crop = tf.image.crop_and_resize(
                       image=xyz_map,
                       boxes=bbox,
                       box_ind=box_indices,
                       crop_size=crop_size)
        xyz_crop = tf.reshape(xyz_crop, (-1,
                       crop_size[0] * crop_size[1], 3))
        xyz_crop = tf.transpose(xyz_crop,
                               [0, 2, 1])
        inside, pj_front, pj_side = self.points_inside_cube(
                       points=xyz_crop,
                       bottom_centers=bottom_centers,
                       rotation=rotation,
                       cube_size=[1.45, 1.55, 4.00])
        points, mask = xyz_crop, inside
        return points, mask

    def rotation_align(self,
                       xyz_map,
                       bottom_centers,
                       rotation,
                       velo_to_cam,
                       cam_to_velo,
                       cube_size=[1.45, 1.55, 4.00],
                       crop_size=[32, 32],
                       delta_rot_num=9,
                       delta_rot=0.03,
                       cos_thres=0.8):
        corners = self.compute_corners(
                      dimensions=tf.constant(
                          np.reshape(cube_size, [1, 3]),
                          dtype=tf.float32),
                      alpha=rotation)
        xyz = tf.expand_dims(bottom_centers, 2)
        global_corners = xyz + corners
        bbox, box_indices = self.corners_to_bbox_sphere(
                          global_corners, cam_to_velo)
        xyz_map = tf.concat([tf.reshape(xyz_map,
                                 (1, 64, 512, 3)),
                             tf.ones(shape=(1, 64, 512, 1),
                                 dtype=tf.float32)], axis=3)
        xyz_map = tf.linalg.matmul(
                      xyz_map,
                      tf.tile(
                          tf.reshape(
                              tf.transpose(
                                  velo_to_cam, [1, 0]),
                              (1, 1, 4, 4)),
                          [1, 64, 1, 1]))

        xyz_map, _ = tf.split(xyz_map, [3, 1], axis=3)
        normal_map = get_normal_map(xyz_map)
        xyz_normal = tf.concat([xyz_map, normal_map], 
                                axis=3)
        xyz_normal_crop = tf.image.crop_and_resize(
                       image=xyz_normal,
                       boxes=bbox,
                       box_ind=box_indices,
                       crop_size=crop_size)
        xyz_normal_crop = tf.reshape(xyz_normal_crop, (-1,
                       crop_size[0] * crop_size[1], 6))
        xyz_normal_crop = tf.transpose(xyz_normal_crop, 
                                      [0, 2, 1])
        inside, pj_front, pj_side = self.points_inside_cube(
                       points=xyz_normal_crop[:, :3, :],
                       bottom_centers=bottom_centers,
                       rotation=rotation,
                       cube_size=[1.45, 1.55, 4.00])
        _, _, _, nx, ny, nz = tf.split(xyz_normal_crop,
                                       6, axis=1)
      
        assert delta_rot_num % 2 == 1
        delta_rot_proposals = (tf.cast(tf.range(delta_rot_num),
                                      tf.float32) - \
                          (delta_rot_num - 1) / 2) * delta_rot
        delta_rot_proposals = tf.reshape(delta_rot_proposals,
                                  [1, delta_rot_num, 1, 1])
        rotation_r = tf.reshape(rotation, [-1, 1, 1, 1])
        rot_prop = rotation_r + delta_rot_proposals
        
        nxnz = tf.concat([nx, nz], axis=1) * inside
        nxnz = nxnz / (1e-6 + tf.norm(nxnz, 
                                      axis=1, 
                                      keepdims=True))
        nx, nz = tf.split(
                     tf.reshape(nxnz, 
                     [-1, 1, 2, crop_size[0] * crop_size[1]]),
                     [1, 1], axis=2)
        front = tf.reduce_sum(tf.cast(tf.greater( 
                 tf.cos(rot_prop) * nx - tf.sin(rot_prop) * nz, 
                   cos_thres), tf.float32), axis=3)

        left  = tf.reduce_sum(tf.cast(tf.greater( 
                 tf.sin(rot_prop) * nx + tf.cos(rot_prop) * nz, 
                  cos_thres), tf.float32), axis=3)

        back  = tf.reduce_sum(tf.cast(tf.greater(
                -tf.cos(rot_prop) * nx + tf.sin(rot_prop) * nz, 
                  cos_thres), tf.float32), axis=3)

        right = tf.reduce_sum(tf.cast(tf.greater(
                -tf.sin(rot_prop) * nx - tf.cos(rot_prop) * nz, 
                  cos_thres), tf.float32), axis=3)

        max_count = tf.reduce_max(tf.concat(
                        [front, left, back, right], axis=2),
                    axis=2)
        assert max_count.get_shape().as_list()[1] == delta_rot_num
        prob = max_count / (1e-6 + tf.reduce_sum(
                                       max_count, 
                                       axis=1, 
                                       keepdims=True))
        rot_prop = tf.reshape(rot_prop, tf.shape(prob))
        rotation_aligned = tf.reduce_sum(rot_prop * prob, axis=1)
      
        return rotation_aligned

    def points_alignment(self, 
                         xyz_map,
                         bottom_centers, 
                         rotation,
                         velo_to_cam,
                         cam_to_velo,
                         cube_size=[1.45, 1.55, 4.00],
                         crop_size=[32, 32]):
        corners = self.compute_corners(
                      dimensions=tf.constant(
                          np.reshape(cube_size, [1, 3]),
                          dtype=tf.float32),
                      alpha=rotation)
        xyz = tf.expand_dims(bottom_centers, 2)
        global_corners = xyz + corners
        bbox, box_indices = self.corners_to_bbox_sphere(
                          global_corners, cam_to_velo)
        xyz_map = tf.concat([tf.reshape(xyz_map, 
                                 (1, 64, 512, 3)),
                             tf.ones(shape=(1, 64, 512, 1), 
                                 dtype=tf.float32)], axis=3)
        xyz_map = tf.linalg.matmul(
                      xyz_map,
                      tf.tile(
                          tf.reshape(
                              tf.transpose(
                                  velo_to_cam, [1, 0]), 
                              (1, 1, 4, 4)), 
                          [1, 64, 1, 1]))

        xyz_map, _ = tf.split(xyz_map, [3, 1], axis=3)
        xyz_crop = tf.image.crop_and_resize(
                       image=xyz_map,
                       boxes=bbox,
                       box_ind=box_indices,
                       crop_size=crop_size)
        xyz_crop = tf.reshape(xyz_crop, (-1, 
                       crop_size[0] * crop_size[1], 3))
        xyz_crop = tf.transpose(xyz_crop, [0, 2, 1])
        inside, pj_front, pj_side = self.points_inside_cube(
                       points=xyz_crop,
                       bottom_centers=bottom_centers,
                       rotation=rotation,
                       cube_size=[1.45, 1.55, 4.00])
        front_max = tf.reduce_max(inside * pj_front, 
                       axis=[1, 2], keepdims=False)
        front_min = tf.reduce_min(inside * pj_front,
                       axis=[1, 2], keepdims=False)
        side_max = tf.reduce_max(inside * pj_side,
                       axis=[1, 2], keepdims=False)
        side_min = tf.reduce_min(inside * pj_side,
                       axis=[1, 2], keepdims=False)
        
        delta_front = tf.where(tf.less(
                          tf.abs(+cube_size[2]/2 - front_max),
                          tf.abs(-cube_size[2]/2 - front_min)),
                      cube_size[2]/2 - front_max,
                      -cube_size[2]/2 - front_min)
        delta_side = tf.where(tf.less(
                          tf.abs(+cube_size[1]/2 - side_max),
                          tf.abs(-cube_size[1]/2 - side_min)),
                      cube_size[1]/2 - side_max,
                      -cube_size[1]/2 - side_min)

        rot = rotation
        delta_x = tf.add(delta_front * tf.math.cos(rotation),
                         delta_side * tf.math.sin(rotation))
        delta_z = tf.add(-delta_front * tf.math.sin(rotation),
                         delta_side * tf.math.cos(rotation))
        x_ori, y_ori, z_ori = tf.split(bottom_centers, 
                                  [1, 1, 1], axis=1)
        x = delta_x + tf.squeeze(x_ori, 1)
        z = delta_z + tf.squeeze(z_ori, 1)
        y = tf.squeeze(y_ori, 1)
        bottom_centers_aligned = tf.concat([tf.expand_dims(x, 1),
                                            tf.expand_dims(y, 1),
                                            tf.expand_dims(z, 1)],
                                            axis=1)
        point_cloud_density = tf.divide(
                                  tf.reduce_sum(inside, [1, 2],
                                      keepdims=False), 
                                  crop_size[0] * crop_size[1])
        
        return bottom_centers_aligned, point_cloud_density
       
    def corners_to_bbox_sphere(self, 
                               corners_cam,
                               cam_to_velo):            
        cam_to_velo = tf.reshape(cam_to_velo, (1, 4, 4))
        x = tf.reduce_sum(tf.reshape(cam_to_velo[:, 0, :3],
                                    (1, 3, 1)) * \
                                     corners_cam, axis=1)
        x = x + tf.reshape(cam_to_velo[:, 0, 3], (1, 1))

        y = tf.reduce_sum(tf.reshape(cam_to_velo[:, 1, :3],
                                    (1, 3, 1)) * \
                                     corners_cam, axis=1)
        y = y + tf.reshape(cam_to_velo[:, 1, 3], (1, 1))

        z = tf.reduce_sum(tf.reshape(cam_to_velo[:, 2, :3],
                                    (1, 3, 1)) * \
                                     corners_cam, axis=1)
        z = z + tf.reshape(cam_to_velo[:, 2, 3], (1, 1))

        points = [tf.layers.flatten(x),
                  tf.layers.flatten(y),
                  tf.layers.flatten(z)]
        phi = tf.reshape(self.get_phi(points), tf.shape(z))
        theta = tf.reshape(self.get_theta(points), tf.shape(z))
        phi = (-phi * 400 / 3 + 8) / 64
        theta = (theta * 400 + 256) / 512

        top = tf.reduce_min(phi, axis=1, keepdims=True)
        bottom = tf.reduce_max(phi, axis=1, keepdims=True)
        left = tf.reduce_min(theta, axis=1, keepdims=True)
        right = tf.reduce_max(theta, axis=1, keepdims=True)

        bbox = tf.concat([top, left, bottom, right], axis=1) 
        box_indices = tf.cast(tf.multiply(bbox[:, 0], 0.0), tf.int32)
        return tf.stop_gradient(bbox), box_indices
        
    def filt_lidar(self, 
                   sphere_map,
                   plane,
                   cam_to_velo,
                   velo_to_cam,
                   mask,
                   cube_size=[1.5, 2.5, 2.5],
                   cube_size_second=[1.53, 1.63, 3.98],
                   roi_point_cloud_size=[32, 32]):
        xyz_map, range_density = tf.split(sphere_map,
                                     [3, 2], axis=2)
        xyz_map = tf.concat([tf.reshape(xyz_map,
                                 (1, 64, 512, 3)),
                             tf.ones(shape=(1, 64, 512, 1),
                                 dtype=tf.float32)], axis=3)
        xyz_map = tf.linalg.matmul(
                      xyz_map,
                      tf.tile(
                          tf.reshape(
                              tf.transpose(
                                  velo_to_cam, [1, 0]),
                              (1, 1, 4, 4)),
                          [1, 64, 1, 1]))
        xyz_map, _ = tf.split(xyz_map, [3, 1], axis=3)
        range_density = tf.reshape(range_density,
                                   [1, 64, 512, 2])
        sphere_map_cam = tf.concat([xyz_map, 
                                    range_density], 3)
        x_grid = self.x_grid
        num_x, num_z = x_grid.get_shape().as_list()
        num_xz = num_x * num_z
        xyz_flatten = self.get_anchor_centers(
                          plane, y_offset=0.0)[:3]
        xyz = tf.expand_dims(
                  tf.transpose(xyz_flatten, [1, 0]), 2)
        corners = self.compute_corners(
                      dimensions=tf.constant(
                          np.reshape(
                              cube_size, [1, 3]
                              ).astype(np.float32)),
                      alpha=tf.constant(0.0,
                          dtype=tf.float32))
        xyz = tf.boolean_mask(xyz, mask)
        kept_corners = xyz + corners
        bbox, box_indices = self.corners_to_bbox_sphere(
                                kept_corners, cam_to_velo)
        network_lidar = ModelLiDAR()
        rotation_init = network_lidar.build(sphere_map_cam,
                          bbox, box_indices, xyz)[1]
 
        corners_second = self.compute_corners(
                      dimensions=tf.constant(
                          np.reshape(
                              cube_size_second,
                              [1, 3]),
                          dtype=tf.float32),
                      alpha=rotation_init)
        kept_corners_second = xyz + corners_second
        bbox_second, box_indices_second = \
            self.corners_to_bbox_sphere(
                             kept_corners_second,
                             cam_to_velo)
       
        class_prob, rotation, rot_vect, mask_prob = \
            network_lidar.build(sphere_map_cam,
                           bbox, box_indices, xyz)
        bottom_centers = tf.squeeze(xyz, 2)
        class_prob = class_prob[:, 1]
        num_points = roi_point_cloud_size[0] * \
                     roi_point_cloud_size[1]
        mask_prob = tf.reshape(
                        mask_prob[:, :, :, 1],
                        [-1, 1, num_points])

        return bottom_centers, rotation, rot_vect, class_prob, mask_prob

    def rot_to_prob(self, rot,
                    rot_bins = np.arange(-np.pi, np.pi, np.pi/8)):
        rot = tf.reshape(rot, [-1, 1])
        rot_bins = tf.constant(np.reshape(rot_bins, 
                                          [1, -1]), 
                               dtype=tf.float32)
        distance = tf.minimum(tf.minimum(np.abs(rot - rot_bins),
                              tf.abs(rot - rot_bins - 2 * np.pi)),
                              tf.abs(rot - rot_bins + 2 * np.pi))
        var = 0.1
        prob = tf.exp(-tf.square(distance) / var)
        prob = prob / tf.reduce_sum(prob, axis=1, keepdims=True)
        return prob

    def rectify_scores(self, x, miu=0.70, sigma=0.15):
        alpha = tf.exp(-(1 - miu) / sigma) + 1
        x = tf.divide(alpha, tf.exp(-(x - miu) / sigma) + 1)
        return x
 
    def build_loss(self,
                   rot_true,
                   rot_vect_pred,
                   rot_pred,
                   class_true,
                   class_pred,
                   mask_true, 
                   mask_pred):
        pos_mask = self.rectify_scores(class_true)

        cls_cross_entropy = \
            utils.cross_entropy(pos_mask, class_pred)
        
        rot_vect_true = self.rot_to_prob(rot_true)
        rot_cross_entropy = tf.reduce_mean(
                                utils.cross_entropy(
                                    rot_vect_true, 
                                    rot_vect_pred), 
                                axis=1) * pos_mask
        rot_error = tf.reduce_sum(tf.abs(rot_pred - 
                                      rot_true) * pos_mask) \
                    / (1e-6 + tf.reduce_sum(pos_mask))

        mask_cross_entropy = \
            utils.cross_entropy(mask_true, mask_pred)
  
        mask_cross_entropy = tf.reduce_mean(
                             mask_cross_entropy, [1, 2])
        mask_cross_entropy = mask_cross_entropy * pos_mask

        cls_loss = tf.reduce_mean(cls_cross_entropy)
        rot_loss = tf.reduce_mean(rot_cross_entropy)
        mask_loss = tf.reduce_mean(mask_cross_entropy)
     
        return cls_loss, rot_loss, mask_loss, rot_error
