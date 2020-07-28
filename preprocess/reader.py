import os
import numpy as np
from PIL import Image

class Reader(object):

    def __init__(self, image, lidar, label, calib):
 
        self.indices = []
        self.data = {}
        for label_file in os.listdir(label):
            data = {}
            data['tracklets'] = []
            index = label_file.split('.')[0]
            data['image_path'] = os.path.join(image, index + '.png')
            data['lidar_path'] = os.path.join(lidar, index + '.bin')
            self.indices.append(index)
            calib_path = os.path.join(calib, index + '.txt')
            with open(calib_path) as calib_file:
                lines = calib_file.readlines()
                P2 = np.reshape(lines[2].strip().split(' ')[1:], (3, 4)).astype(np.float32)
                
                Tr = np.zeros((4, 4), dtype=np.float32)
                Tr[3, 3] = 1.0
                Tr[:3, :] = np.reshape(lines[5].strip().split(' ')[1:], (3, 4)).astype(np.float32)
                
                R0 = np.zeros((4, 4), dtype=np.float32)
                R0[3, 3] = 1.0
                R0[:3, :3] = np.reshape(lines[4].strip().split(' ')[1:], (3, 3)).astype(np.float32)
                
                data['P2'], data['Tr'], data['R0'] = P2, Tr, R0
                calib_file.close()
            label_path = os.path.join(label, index + '.txt')
            with open(label_path) as label_file:
                lines = label_file.readlines()
                for line in lines:
                    elements = line.split(' ')
                    if not elements[0] == 'Car':
                        continue
                    bbox = np.array(elements[4: 8], dtype=np.float32)
                    dimensions = np.array(elements[8: 11], dtype=np.float32)
                    location = np.array(elements[11: 14], dtype=np.float32)
                    rotation_y = np.array(elements[14], dtype=np.float32)
                    data['tracklets'].append({'bbox': bbox,
                                              'dimensions': dimensions,
                                              'location': location,
                                              'rotation_y': rotation_y})
            self.data[index] = data
        return

    def read_lidar(self, lidar_path, lidar_to_camera):
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
        lidar[:, 3] = 1.0
        camera = np.dot(lidar_to_camera, lidar.T)
        return camera

    def keep_in_image(self, point_cloud_camera, camera_to_image, width, height):
        image_coor = np.dot(camera_to_image, 
                            np.concatenate([point_cloud_camera, 
                                        np.zeros(shape=(1, point_cloud_camera.shape[-1]))],
                                        axis=0))
        image_coor = image_coor[:2] / image_coor[2]
        keep = np.logical_and(np.logical_and(image_coor[0, :] > -1,
                                         image_coor[0, :] < width), 
                              np.logical_and(image_coor[1, :] > -1, 
                                         image_coor[1, :] < height))
        return point_cloud_camera[:, keep]
