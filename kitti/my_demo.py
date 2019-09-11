import os
import numpy as np
import glob
import cv2
from kitti_foundation import KITTI,KITTI_Util
import sys
sys.path.append('.')
import my_log as log

def print_projection_cv2(points, color, image):
    """ project converted velodyne points into camera image """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    points = points.astype(np.int32).transpose((1,0)).tolist()
    color = color.astype(np.int32).tolist()

    for pt,c in zip(points,color):
        cv2.circle(hsv_image, (pt[0], pt[1]), 2, (c, 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (int(points[0][i]), int(points[1][i])), 2, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

# root = '/media/james/MyPassport/James/dataset/KITTI/raw/2011_09_26/'
root = '/media/james/Ubuntu_Data/dataset/KITTI/raw/2011_09_26/'
velo_path = os.path.join(root, '2011_09_26_drive_0005_sync/velodyne_points/data')
image_path = os.path.join(root, '2011_09_26_drive_0005_sync/image_02/data')
v2c = os.path.join(root,'calib_velo_to_cam.txt')
c2c = os.path.join(root,'calib_cam_to_cam.txt')

v_fov, h_fov = (-24.9, 2.0), (-90, 90)

res = KITTI_Util(frame=0,camera_path=image_path, velo_path=velo_path,v2c_path=v2c, c2c_path=c2c)

img, points, color = res.velo_projection_frame(v_fov=v_fov, h_fov=h_fov)
log.warn(total = res.num_frame, img=img.shape, points=points.shape, color=color.shape)
result = print_projection_cv2(points, color, img)

cv2.imshow('projection result', result)
cv2.waitKey(0)