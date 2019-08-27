import os
import numpy as np
import glob
import cv2
import sys
sys.path.append('.')
from visualizer.kitti_foundation import Kitti,Kitti_util

def print_projection_cv2(points, color, image):
    """ project converted velodyne points into camera image """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 2, (np.int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (int(points[0][i]), int(points[1][i])), 2, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

root = '/media/james/MyPassport/James/dataset/KITTI/raw/2011_09_26/'
velo_path = os.path.join(root, '2011_09_26_drive_0005_sync/velodyne_points/data')
v2c_filepath = os.path.join(root,'calib_velo_to_cam.txt')
c2c_filepath = os.path.join(root,'calib_cam_to_cam.txt')
image_path = os.path.join(root, '2011_09_26_drive_0005_sync/image_02/data')

def pano_example1():
    """ save one frame image about velodyne dataset converted to panoramic image  """
    v_fov, h_fov = (-10.5, 2.0), (-60, 80)
    velo = Kitti_util(frame=89, velo_path=velo_path)

    frame = velo.velo_2_pano_frame(h_fov, v_fov, depth=False)

    cv2.imshow('panoramic result', frame)
    cv2.waitKey(0)

def pano_example2():
    """ save video about velodyne dataset converted to panoramic image  """
    v_fov, h_fov = (-24.9, 2.0), (-180, 160)

    velo2 = Kitti_util(frame='all', velo_path=velo_path)
    pano = velo2.velo_2_pano(h_fov, v_fov, depth=False)

    velo = Kitti_util(frame=0, velo_path=velo_path)
    velo.velo_2_pano_frame(h_fov, v_fov, depth=False)
    size = velo.surround_size

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = cv2.VideoWriter('experiment/pano_result.avi', fourcc, 25.0, size, False)

    for frame in pano:
        vid.write(frame)

    print('video saved')
    vid.release()

def topview_example1():
    """ save one frame image about velodyne dataset converted to topview image  """
    x_range, y_range, z_range = (-15, 15), (-10, 10), (-2, 2)
    velo = Kitti_util(frame=89, velo_path=velo_path)

    frame = velo.velo_2_topview_frame(x_range=x_range, y_range=y_range, z_range=z_range)

    cv2.imshow('panoramic result', frame)
    cv2.waitKey(0)

def topview_example2():
    """ save video about velodyne dataset converted to topview image  """
    x_range, y_range, z_range, scale = (-20, 20), (-20, 20), (-2, 2), 10
    size = (int((max(y_range) - min(y_range)) * scale), int((max(x_range) - min(x_range)) * scale))

    velo2 = Kitti_util(frame='all', velo_path=velo_path)
    topview = velo2.velo_2_topview(x_range=x_range, y_range=y_range, z_range=z_range, scale=scale)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = cv2.VideoWriter('experiment/topview_result.avi', fourcc, 25.0, size, False)

    for frame in topview:
        vid.write(frame)

    print('video saved')
    vid.release()

def projection_example1():
    """ save one frame about projecting velodyne points into camera image """
    v_fov, h_fov = (-24.9, 2.0), (-90, 90)

    res = Kitti_util(frame=89, camera_path=image_path, velo_path=velo_path, \
                    v2c_path=v2c_filepath, c2c_path=c2c_filepath)

    img, pnt, c_ = res.velo_projection_frame(v_fov=v_fov, h_fov=h_fov)

    result = print_projection_cv2(pnt, c_, img)

    cv2.imshow('projection result', result)
    cv2.waitKey(0)

def projection_example2():
    """ save video about projecting velodyne points into camera image """

    v_fov, h_fov = (-24.9, 2.0), (-90, 90)

    temp = Kitti(frame=0, camera_path=image_path)
    img = temp.camera_file
    size = (img.shape[1], img.shape[0])

    """ save result video """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = cv2.VideoWriter('experiment/projection_result.avi', fourcc, 25.0, size)
    test = Kitti_util(frame='all', camera_path=image_path, velo_path=velo_path, \
                      v2c_path=v2c_filepath, c2c_path=c2c_filepath)

    res = test.velo_projection(v_fov=v_fov, h_fov=h_fov)

    for frame, point, cc in res:
        image = print_projection_cv2(point, cc, frame)
        vid.write(image)

    print('video saved')
    vid.release()

def xml_example():

    xml_path = "./tracklet_labels.xml"
    xml_check = Kitti_util(xml_path=xml_path)

    tracklet_, type_ = xml_check.tracklet_info
    print(tracklet_[0])

if __name__ == "__main__":
    pano_example1()
    topview_example1()
    projection_example1()
    
    pano_example2()
    topview_example2()
    projection_example2()

