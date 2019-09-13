import os
import numpy as np
from PIL import Image
import glob
import cv2
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

def calib_velo2cam(fn_v2c):
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    with open(fn_v2c, "r") as f:
        v2c_file = f.readlines()

    for line in v2c_file:
        (key, val) = line.split(':', 1)
        if key == 'R':
            R = np.fromstring(val, sep=' ')
            R = R.reshape(3, 3)
        if key == 'T':
            T = np.fromstring(val, sep=' ')
            T = T.reshape(3, 1)
    return R, T

def calib_cam2cam(fn_c2c, mode = '02'):
    """
    If your image is 'rectified image' :
        get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)

    In this code, only P matrix info is used for rectified image
    """
    with open(fn_c2c, "r") as f:
        c2c_file = f.readlines()

    for line in c2c_file:
        (key, val) = line.split(':', 1)
        if key == ('P_rect_' + mode):
            P = np.fromstring(val, sep=' ')
            P = P.reshape(3, 4)
            # erase 4th column ([0,0,0])
            P = P[:3, :3]
    return P

def hv_in_range(m, n, fov, fov_type='h'):
    """ extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit 
        horizontal limit = azimuth angle limit
        vertical limit = elevation angle limit
    """

    if fov_type == 'h':
        return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                                np.arctan2(n, m) < (-fov[0] * np.pi / 180))
    elif fov_type == 'v':
        return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                                np.arctan2(n, m) > (fov[0] * np.pi / 180))
    else:
        raise NameError("fov type must be set between 'h' and 'v' ")

def _3d_in_range(points, x_range, y_range, z_range):
    """ extract filtered in-range velodyne coordinates based on x,y,z limit """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return points[
        np.logical_and.reduce(
            (x > x_range[0], x < x_range[1],y > y_range[0], y < y_range[1],z > z_range[0], z < z_range[1]))]


def points_filter(points, h_fov, v_fov, x_range = None, y_range = None, z_range = None):
    """
    filter points based on h,v FOV and x,y,z distance range.
    x,y,z direction is based on velodyne coordinates
    1. azimuth & elevation angle limit check
    2. x,y,z distance limit
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    h_points = hv_in_range(x, y, h_fov, fov_type='h')
    v_points = hv_in_range(d, z, v_fov, fov_type='v')
    con = np.logical_and(h_points, v_points)
    lim_x, lim_y, lim_z, lim_d = x[con], y[con], z[con], d[con]
    # x, y, z, d = lim_x, lim_y, lim_z, lim_d

    if x_range is None and y_range is None and z_range is None:
        pass
    elif x_range is not None and y_range is not None and z_range is not None:
        # extract in-range points
        temp_x, temp_y = _3d_in_range(lim_x), _3d_in_range(lim_y)
        temp_z, temp_d = _3d_in_range(lim_z), _3d_in_range(lim_d)
        lim_x, lim_y, lim_z, lim_d = temp_x, temp_y, temp_z, temp_d
    else:
        raise ValueError("Please input x,y,z's min, max range(m) based on velodyne coordinates. ")

    return lim_x, lim_y, lim_z, lim_d

def normalize_data(val, min, max, scale, depth=False, clip=False):
    """ Return normalized data """
    if clip:
        # limit the values in an array
        np.clip(val, min, max, out=val)
    if depth:
        """
        print 'normalized depth value'
        normalize values to (0 - scale) & close distance value has high value. (similar to stereo vision's disparity map)
        """
        return (((max - val) / (max - min)) * scale).astype(np.uint8)
    else:
        """
        print 'normalized value'
        normalize values to (0 - scale) & close distance value has low value.
        """
        return (((val - min) / (max - min)) * scale).astype(np.uint8)

def extract_points(points, h_fov, v_fov):
    """ extract points corresponding to FOV setting """

    # filter in range points based on fov, x,y,z range setting
    lim_x, lim_y, lim_z, lim_d = points_filter(points, h_fov, v_fov)

    # Stack arrays in sequence horizontally
    xyz_ = np.hstack((lim_x[:, None], lim_y[:, None], lim_z[:, None]))
    xyz_ = xyz_.T

    # stack (1,n) arrays filled with the number 1
    one_mat = np.full((1, xyz_.shape[1]), 1)
    xyz_ = np.concatenate((xyz_, one_mat), axis=0)

    # need dist info for points color
    color = normalize_data(lim_d, min=1, max=70, scale=120, clip=True)

    return xyz_, color

def velo_2_img_projection(points, fn_v2c, fn_c2c, h_fov = None, v_fov = None):
    """ convert velodyne coordinates to camera image coordinates """

    # rough velodyne azimuth range corresponding to camera horizontal fov
    if h_fov is None:
        h_fov = (-50, 50)
    if h_fov[0] < -50:
        h_fov = (-50,) + h_fov[1:]
    if h_fov[1] > 50:
        h_fov = h_fov[:1] + (50,)

    # R_vc = Rotation matrix ( velodyne -> camera )
    # T_vc = Translation matrix ( velodyne -> camera )
    R_vc, T_vc = calib_velo2cam(fn_v2c)

    # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
    P_ = calib_cam2cam(fn_c2c)

    """
    xyz_v - 3D velodyne points corresponding to h, v FOV limit in the velodyne coordinates
    c_    - color value(HSV's Hue vaule) corresponding to distance(m)

                [x_1 , x_2 , .. ]
    xyz_v   =   [y_1 , y_2 , .. ]
                [z_1 , z_2 , .. ]
                [ 1  ,  1  , .. ]
    """
    xyz_v, c_ = extract_points(points, h_fov, v_fov)

    """
    RT_ - rotation matrix & translation matrix
        ( velodyne coordinates -> camera coordinates )

            [r_11 , r_12 , r_13 , t_x ]
    RT_  =  [r_21 , r_22 , r_23 , t_y ]
            [r_31 , r_32 , r_33 , t_z ]
    """
    RT_ = np.concatenate((R_vc, T_vc), axis=1)

    # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
    for i in range(xyz_v.shape[1]):
        xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

    """
    xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
               [x_1 , x_2 , .. ]
    xyz_c   =  [y_1 , y_2 , .. ]
               [z_1 , z_2 , .. ]
    """
    xyz_c = np.delete(xyz_v, 3, axis=0)

    # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
    for i in range(xyz_c.shape[1]):
        xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

    """
    xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
    ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
                [s_1*x_1 , s_2*x_2 , .. ]
    xy_i    =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]
                [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
    """
    xy_i = xyz_c[::] / xyz_c[::][2]
    ans = np.delete(xy_i, 2, axis=0)

    return ans, c_

if __name__ == "__main__":
    root = '/media/james/Ubuntu_Data/dataset/KITTI/raw/2011_09_26/'
    part = '2011_09_26_drive_0001_sync'
    velo_path = os.path.join(root, '%s/velodyne_points/data/'%(part))
    image_path = os.path.join(root, '%s/image_02/data/'%(part))
    fn_v2c = os.path.join(root,'calib_velo_to_cam.txt')
    fn_c2c = os.path.join(root,'calib_cam_to_cam.txt')
    h_fov = (-40, 40)
    v_fov = (-25, 2.0)

    index = 0
    while True:
        with log.Tick():
            fn_frame = os.path.join(image_path, '%010d.jpg' % (index))
            fn_velo = os.path.join(velo_path, '%010d.bin' %(index))

            if not os.path.exists(fn_frame):
                fn_frame = os.path.join(image_path, '%010d.png' % (index))

                if not os.path.exists(fn_frame) or not os.path.exists(fn_velo):
                    print('End of the sequence')
                    break
            
            with log.Tock():
                frame = cv2.imread(fn_frame)
                # frame = np.array(Image.open(fn_frame))[:,:,::-1]
                points = np.fromfile(fn_velo, dtype=np.float32)
                points_3d = points.reshape(-1, 4)[:,:3]
            
            with log.Tock():
                points_2d, color = velo_2_img_projection(points_3d, fn_v2c, fn_c2c, h_fov, v_fov)

            result = print_projection_cv2(points_2d, color, frame)
            cv2.imshow('projection result', result)

            index += 1

        if 27 == cv2.waitKey(1):
            break
        