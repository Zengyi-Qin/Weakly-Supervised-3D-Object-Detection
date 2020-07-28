import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def get_normal_map(x, area_weighted=False):
    """
    x: [bs, h, w, 3] (x,y,z) -> (nx,ny,nz)
    """
    nn = 6
    p11 = x
   
    p = tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]))
    p11 = p[:, 1:-1, 1:-1, :]
    p10 = p[:, 1:-1, 0:-2, :]
    p01 = p[:, 0:-2, 1:-1, :]
    p02 = p[:, 0:-2, 2:, :]
    p12 = p[:, 1:-1, 2:, :]
    p20 = p[:, 2:, 0:-2, :]
    p21 = p[:, 2:, 1:-1, :]

    pos = [p10, p01, p02, p12, p21, p20]

    for i in range(nn):
        pos[i] = tf.subtract(pos[i], p11)

    normals = []
    for i in range(1, nn):
        normals.append(tf.cross(pos[i%nn], pos[(i-1+nn)%nn]))
        normal = tf.reduce_sum(tf.stack(normals), axis=0)

    if not area_weighted:
        normal = tf.nn.l2_normalize(normal, 3)

    normal = tf.where(tf.is_nan(normal), 
                 tf.zeros_like(normal), normal)
    return normal

def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs  =  tf.subtract(deltas, targets)
    smooth_l1_signs = \
        tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
                    tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1

def cross_entropy(true, pred):
    true_shape = true.get_shape().as_list()[1:]
    pred_shape = pred.get_shape().as_list()[1:]
    assert true_shape == pred_shape
    ent = true * tf.log(pred + 1e-6) + \
          (1 - true) * tf.log(1 - pred + 1e-6)
    return -ent

def save_histogram(x, rotation, save_path):

    plt.figure()
    hist_step = 0.1
    bins = np.arange(0, 1.57, hist_step)
    n, bins, patches = plt.hist(x, bins=bins, 
                                density=True, 
                                facecolor='g', 
                                alpha=0.25)
    plt.axis([0, 1.57, 0, 10.0])
    plt.grid(False)
    plt.title('rotation: {:.3f}'.format(rotation))
    plt.savefig(save_path)
    plt.close()
    return

def get_corners(dimensions, location, rotation_y):
    R = np.array([[+np.cos(rotation_y), 0, +np.sin(rotation_y)], 
                  [                  0, 1,                  0],
                  [-np.sin(rotation_y), 0, +np.cos(rotation_y)]], 
                  dtype=np.float32)
    h, w, l = dimensions
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    corners_3D = np.dot(R, [x_corners, y_corners, z_corners])
    corners_3D += location.reshape((3, 1))
    return corners_3D
    
def draw_projection(corners, P2, ax, color, score, linewidth=3):
    projection = np.dot(P2, np.vstack([corners, np.ones(8, dtype=np.int32)]))
    projection = (projection / projection[2])[:2]
    orders = [[0, 1, 2, 3, 0],
              [4, 5, 6, 7, 4],
              [2, 6], [3, 7],
              [1, 5], [0, 4]]
    for order in orders:
        ax.plot(
            projection[0, order], 
            projection[1, order], 
            color=color, linewidth=linewidth, alpha=score)
    return

def draw_projected_faces(corners, P2, ax, color, score):
    projection = np.dot(P2, np.vstack([corners, np.ones(8, dtype=np.int32)]))
    projection = (projection / projection[2])[:2]
    patches = []
    orders = [[0, 1, 2, 3],
              [4, 5, 6, 7],
              [2, 6, 5, 1], 
              [3, 7, 4, 0],
              [2, 6, 7, 3],
              [1, 0, 4, 5]]
    for order in orders:
        polygon = Polygon(projection[:, order].T, True, color=color)
        patches.append(polygon)
    colors = np.ones(len(patches)) * 4
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.2*score)
    p.set_array(colors)
    p.set_clim([0, 9])
    ax.add_collection(p)
    return

def draw_mask_projection(points, P2, ax, color, score):
    num_points = points.shape[1]
    points = P2.dot(np.vstack([points, np.ones(shape=[1, num_points])]))
    points = points[:2] / points[2]
    ax.scatter(points[0], points[1], s=20, alpha=score*0.5, c=color)
    return

def draw_space(corners, ax, color, alpha, linewidth=3):
    assert corners.shape == (3, 8)
    orders = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]
    lines = np.zeros((3, 16), dtype=np.float32)
    for index, point in enumerate(orders):
        lines[:, index] = corners[:, point]
    ax.plot(-lines[0], lines[2] - 8, -lines[1], c=color, alpha=alpha, linewidth=linewidth)
    return

def draw_faces(corners, ax):
    assert corners.shape == (3, 8)
    corners_r = [-corners[0].reshape(1, 8), 
                 corners[2].reshape(1, 8) - 8,
                 -corners[1].reshape(1, 8)]
    corners_r = np.concatenate(corners_r, axis=0)
    corners_r = corners_r.T
    patches = []
    orders = [[0, 1, 2, 3],
              [4, 5, 6, 7],
              [2, 6, 5, 1],
              [3, 7, 4, 0],
              [2, 6, 7, 3],
              [1, 0, 4, 5]]
    for order in orders:
        poly3d = [corners_r[order]]
        p = Poly3DCollection(poly3d, facecolors='azure', linewidths=3, alpha=0.2)
        p.set_edgecolor('cyan')
        ax.add_collection3d(p)
    return

def draw_space_truth(corners, ax, color):
    assert corners.shape == (3, 8)
    orders = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]
    lines = np.zeros((3, 16), dtype=np.float32)
    for index, point in enumerate(orders):
        lines[:, index] = corners[:, point]
    ax.plot(-lines[0], lines[2] - 8, -lines[1], c=color, linewidth=6)
    return

def draw_point_cloud(point_cloud, ax, color, size):
    ax.scatter(-point_cloud[0], point_cloud[2] - 8, -point_cloud[1], c=color, s=size)
    return
    
def read_lidar(lidar_path, lidar_to_camera):
    lidar = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
    lidar[:, 3] = 1.0
    camera = np.dot(lidar_to_camera, lidar.T)
    return camera

def read_sphere(sphere_path, velo_to_cam):
    points = np.load(open(sphere_path, 'rb'))[:, :, :3].reshape((-1, 3)).T
    points = np.concatenate([points, np.zeros((1, points.shape[1]))], axis=0)
    points_cam = velo_to_cam.dot(points)[:3]
    return points_cam

def keep_in_image(point_cloud_camera, camera_to_image, width=1240, height=370):
    image_coor = np.dot(camera_to_image, 
                        np.concatenate([point_cloud_camera, 
                                        np.zeros(shape=(1, point_cloud_camera.shape[-1]))],
                                        axis=0))
    image_coor = image_coor[:2] / image_coor[2]
    keep = np.logical_and(np.logical_and(image_coor[0, :] > 0,
                                         image_coor[0, :] < width), 
                          np.logical_and(image_coor[1, :] > 0, 
                                         image_coor[1, :] < height))
    keep = np.logical_and(point_cloud_camera[2] > 0, keep)
    return point_cloud_camera[:, keep]
   
def sort_tracklets(tracklets):
    sort = sorted(tracklets, key=lambda x: x['location'][0])
    return sort

def norm(x):
    return np.linalg.norm(x)

def smooth_rot(rot):
    mean_sin = np.mean(np.sin(rot))
    mean_cos = np.mean(np.cos(rot))
    return np.ones_like(rot) * np.arctan2(mean_sin, mean_cos)

def parse_kitti_lines(lines):
    data = []
    for line in lines:
        elements = line.split(' ')
        if not elements[0] == 'Car':
            continue
        bbox = np.array(elements[4: 8], dtype=np.float32)
        dimensions = np.array(elements[8: 11], dtype=np.float32)
        location = np.array(elements[11: 14], dtype=np.float32)
        rotation_y = np.array(elements[14], dtype=np.float32)
        score = float(elements[15])
        data.append({'bbox': bbox,
                     'dimensions': dimensions,
                     'location': location,
                     'rotation_y': rotation_y,
                     'score': score})
    return data