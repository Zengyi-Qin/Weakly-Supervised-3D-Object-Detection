import cv2
import os
import numpy as np
from reader import Reader
from scipy import interpolate
import logging
logging.basicConfig(level=logging.INFO)


def read_lidar(lidar_path, mat):
    lidar = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
    lidar[:, 3] = 1.0
    camera = np.dot(mat, lidar.T)
    return camera

def read_lidar_raw(lidar_path):
    lidar = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
    return lidar.T

def get_phi(points):
    sine = points[2] / np.sqrt(points[0]**2 + 
                               points[1]**2 + 
                               points[2]**2)
    return np.arcsin(sine)

def get_theta(points):
    sine = points[1] / np.sqrt(points[0]**2 + 
                               points[1]**2)
    return np.arcsin(sine)

def get_sphere_map(lidar_path):
    points_raw = read_lidar_raw(lidar_path)
    keep = points_raw[0] > 0
    points_keep = points_raw[:, keep]
    theta = get_theta(points_keep)
    phi = get_phi(points_keep)
    keep_theta = np.logical_and(theta < 0.64,
                          theta > -0.64)
    keep_phi = np.logical_and(phi < 0.06,
                              phi > -0.42)
    keep_dist = points_keep[0] < 74
    keep = np.logical_and(np.logical_and(keep_theta, 
                          keep_phi), keep_dist)
    points_keep = points_keep[:, keep]
    theta = np.round(get_theta(points_keep) * 400
                     ).astype(np.int32) + 256
    theta = np.clip(theta, 0, 511)
    phi = -np.round(get_phi(points_keep) / \
                    7.5 * 1000).astype(np.int32) + 8
    phi = np.clip(phi, 0, 63)
    sphere_map = np.zeros((64, 512, 5))
    sphere_map[phi, theta, :3] = points_keep[:3].T
    sphere_map[phi, theta, 3] = np.linalg.norm(points_keep[:3], axis=0)
    sphere_map[phi, theta, 4] = points_keep[3]
    return sphere_map

def interp_init(invalid, x):
    i = 0
    while invalid[i] and i < invalid.shape[0] - 1:
        i = i + 1
    x[:i] = x[i]
    j = x.shape[0] - 1
    while invalid[j] and j > 0:
        j = j - 1
    x[j:] = x[j]
    return x

def interp_init_lstsq(invalid, x):
    valid = np.logical_not(invalid)
    if np.sum(valid) < 3:
        return interp_init(invalid, x)
    domain = np.arange(x.shape[0])
    valid_domain = domain[valid]
    valid_x = x[valid]
    A = np.vstack([valid_domain, np.ones(len(valid_domain))]).T
    m, c = np.linalg.lstsq(A, valid_x, rcond=None)[0]
    
    i = 0
    while invalid[i] and i < invalid.shape[0] - 1:
        i = i + 1
    x[:i] = domain[:i] * m + c
    j = x.shape[0] - 1
    while invalid[j] and j > 0:
        j = j - 1
    x[j:] = domain[j:] * m + c
    return x

def inpaint_map(sphere_map):
    norm = np.linalg.norm(sphere_map, axis=-1)
    mask = (norm > 0).astype(np.int32)
    patch_width = 8
    iters = sphere_map.shape[1] // patch_width
    for c in range(sphere_map.shape[-1]):
        for i in range(iters):
            sphere_map[:, i*patch_width:(i+1)*patch_width, c] = \
            inpaint.inpaint_biharmonic(sphere_map[:, i*patch_width:(i+1)*patch_width, c], 
                                   mask[:, i*patch_width:(i+1)*patch_width])
    return sphere_map

def interp_map(sphere_map):
    if len(sphere_map.shape) == 2:
        sphere_map = np.expand_dims(sphere_map, 2)
    norm = np.linalg.norm(sphere_map, axis=-1)
    invalid = norm == 0
    for c in range(sphere_map.shape[2]):
        for x in range(sphere_map.shape[1]):
            sphere_map[:, x, c] = interp_init(invalid[:, x], sphere_map[:, x, c])
    norm = np.linalg.norm(sphere_map, axis=-1)
    valid = norm > 0
    y = np.arange(0, sphere_map.shape[0])
    for c in range(sphere_map.shape[2]):
        for x in range(sphere_map.shape[1]):
            valid_index = valid[:, x]
            valid_y = y[valid_index]
            if np.sum(valid_index) == 0:
                continue
            valid_z = sphere_map[valid_index, x, c]
            f = interpolate.interp1d(valid_y, valid_z, kind='linear')
            sphere_map[:, x, c] = f(y)
    return sphere_map

def to_eight_bits(mat):
    min_value = np.nanmin(mat)
    max_value = np.amax(mat)
    ratio = 255.0 / (max_value - min_value + 1e-12)
    mat_scaled = (mat - min_value) * ratio
    return mat_scaled.astype(np.uint8), (min_value, ratio)

def to_float_bits(mat, cache):
    min_value, ratio = cache
    mat = mat * 1.0 / ratio + min_value
    return mat

def inpaint_map(sphere_map):
    norm = np.linalg.norm(sphere_map, axis=-1)
    mask = (norm == 0).astype(np.uint8)
    for c in range(sphere_map.shape[-1]):
        eight_bits, cache = to_eight_bits(sphere_map[:, :, c])
        eight_bits = cv2.inpaint(eight_bits, mask, 5, cv2.INPAINT_TELEA)
        float_bits = to_float_bits(eight_bits, cache)
        sphere_map[:, :, c] += float_bits * mask
    return sphere_map

def main():
    LABEL_DIR = '../data/kitti/training/label_2'
    IMAGE_DIR = '../data/kitti/training/image_2'
    LIDAR_DIR = '../data/kitti/training/velodyne'
    CALIB_DIR = '../data/kitti/training/calib'
    SAVE_DIR = '../data/kitti/training/sphere'
    label_reader = Reader(IMAGE_DIR, LIDAR_DIR, LABEL_DIR, CALIB_DIR)
    total_labels = len(label_reader.indices)
    for iindex, index in enumerate(label_reader.indices):
        logging.info('Processing {} / {}'.format(iindex, total_labels))
        data = label_reader.data[index]
        sphere_map = get_sphere_map(data['lidar_path'])
     
        for c in [0, 1, 2, 3, 4]:
            sphere_map[:, :, c] = interp_map(
                sphere_map[:, :, c]).squeeze()

        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
            
        with open(os.path.join(
                 SAVE_DIR, index+'.npy'), 'wb') as f:
            np.save(f, sphere_map)

if __name__ == '__main__':
    main()
