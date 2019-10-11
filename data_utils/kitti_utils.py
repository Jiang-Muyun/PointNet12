import os
import cv2
import json
import yaml
import numpy as np
import random
from tqdm import tqdm
import torch
from PIL import Image

class KITTI_2_Common():
    def __init__(self, model):
        self.model = model
        self.common = None
        self.kitti_names = ['road', 'sidewalk', 'building', 'wall', 'fence',
            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle'
        ]
        self.kitti_2_common = [
            'road','sidewalk','building+wall','fence','pole',
            'traffic_light+traffic_sign','vegetation',
            'terrain','person','rider','car','truck',
            'bus+train','motorcycle','bicycle','sky'
        ]
        self.kitti_colors = [
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30],[220, 220, 0], [107, 142, 35],
            [152, 251, 152], [0, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142],
            [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
        ]
        self.colors = []
        for index, c_class in enumerate(self.kitti_2_common):
            name_0 = c_class.split('+')[0]
            index_0 = self.kitti_names.index(name_0)
            self.colors.append(self.kitti_colors[index_0])
        
        self.kitti_colors = np.array(self.kitti_colors)
        self.colors = np.array(self.colors)

    def __call__(self, x):
        logits = self.model(x)
        new_size = list(logits.size())
        new_size[1] = len(self.kitti_2_common)
        self.common = torch.zeros(new_size, dtype=logits.dtype).cuda()
        
        for index, c_class in enumerate(self.kitti_2_common):
            merge = c_class.split('+')
            if len(merge) == 1:
                index_0 = self.kitti_names.index(merge[0])
                self.common[:, index] = logits[:,index_0]
            elif len(merge) == 2:
                index_0 = self.kitti_names.index(merge[0])
                index_1 = self.kitti_names.index(merge[1])
                self.common[:, index] = logits[:,[index_0,index_1]].max(1)[0]
            else:
                raise NotImplementedError("not implemented!")
        return self.common


class SemKITTI_2_Common():
    def __init__(self, model, model_name):
        self.model_name = model_name
        self.model = model
        self.common = None
        self.semkitti_names = [
            'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
            'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
            'building', 'fence', 'vegetation', 'trunk', 'terrain','pole', 'traffic-sign'
        ]
        self.semkitti_2_common = [
            'road', 'parking+sidewalk', 'building', 'fence', 'trunk+pole',
            'traffic-sign', 'vegetation', 'terrain', 'person', 'bicyclist+motorcyclist',
            'car', 'truck', 'other-vehicle', 'motorcycle', 'bicycle','other-ground'
        ]
        self.semkitti_colors = [
            [245, 150, 100],[245, 230, 100],[150, 60, 30],[180, 30, 80],
            [255, 0, 0],[30, 30, 255],[200, 40, 255],[90, 30, 150],[255, 0, 255],
            [255, 150, 255], [75, 0, 75],[75, 0, 175],[0, 200, 255],[50, 120, 255],
            [0, 175, 0],[0, 60, 135],[80, 240, 150],[150, 240, 255],[0, 0, 255]
        ]

        self.colors = []
        for index, c_class in enumerate(self.semkitti_2_common):
            name_0 = c_class.split('+')[0]
            index_0 = self.semkitti_names.index(name_0)
            self.colors.append(self.semkitti_colors[index_0])
        
        self.semkitti_colors = np.array(self.semkitti_colors)
        self.colors = np.array(self.colors)

    def __call__(self, x):
        if self.model_name == 'pointnet':
            logits, feature_transform = self.model(x)
        else:
            logits = self.model(x)

        new_size = list(logits.size())
        new_size[2] = len(self.semkitti_2_common)
        self.common = torch.zeros(new_size, dtype=logits.dtype).cuda()

        for index, c_class in enumerate(self.semkitti_2_common):
            merge = c_class.split('+')
            if len(merge) == 1:
                index_0 = self.semkitti_names.index(merge[0])
                self.common[:, :, index] = logits[:, :, index_0]
            elif len(merge) == 2:
                index_0 = self.semkitti_names.index(merge[0])
                index_1 = self.semkitti_names.index(merge[1])
                self.common[:, :, index] = logits[:, :, [index_0,index_1]].max(2)[0]
            else:
                raise NotImplementedError("not implemented!")
        
        if self.model_name == 'pointnet':
            return self.common, feature_transform
        else:
            return self.common


sem_kitti_class_names = [
    'car','bicycle','motorcycle','truck','other-vehicle',
    'person','bicyclist','motorcyclist','road','parking','sidewalk','other-ground',
    'building','fence','vegetation','trunk','terrain','pole','traffic-sign']

sem_kitti_colors = [[245, 150, 100],[245, 230, 100],[150, 60, 30],[180, 30, 80],
    [255, 0, 0],[30, 30, 255],[200, 40, 255],[90, 30, 150],[255, 0, 255],
    [255, 150, 255], [75, 0, 75],[75, 0, 175],[0, 200, 255],[50, 120, 255],
    [0, 175, 0],[0, 60, 135],[80, 240, 150],[150, 240, 255],[0, 0, 255]
]

kitti_class_names = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain','sky', 'person', 
    'rider', 'car', 'truck', 'bus', 'train','motorcycle', 'bicycle']

kitti_colors = [[128, 64, 128],[244, 35, 232],[70, 70, 70],[102, 102, 156],
    [190, 153, 153],[153, 153, 153],[250, 170, 30],[220, 220, 0],[107, 142, 35],
    [152, 251, 152],[0, 130, 180],[220, 20, 60],[255, 0, 0],[0, 0, 142],
    [0, 0, 70],[0, 60, 100],[0, 80, 100],[0, 0, 230],[119, 11, 32]
]

class Semantic_KITTI_Utils():
    def __init__(self, root, subset = 'all'):
        self.root = root

        base_path = os.path.dirname(os.path.realpath(__file__)) + '/../'

        self.R, self.T = self.calib_velo2cam(base_path+'config/calib_velo_to_cam.txt')
        self.P = self.calib_cam2cam(base_path+'config/calib_cam_to_cam.txt' ,mode="02")
        self.RT = np.concatenate((self.R, self.T), axis=1)

        self.sem_cfg = yaml.load(open(base_path+'config/semantic-kitti.yaml','r'), Loader=yaml.SafeLoader)
        
        self.class_names = self.sem_cfg['labels']
        self.learning_map = self.sem_cfg['learning_map']
        self.learning_map_inv = self.sem_cfg['learning_map_inv']
        self.learning_ignore = self.sem_cfg['learning_ignore']
        self.sem_color_map = self.sem_cfg['color_map']

        self.length = {
            '00': 4540,'01':1100,'02':4660,'03':800,'04':270,'05':2760,
            '06':1100,'07':1100,'08':4070,'09':1590,'10':1200
        }
        assert subset in ['all', 'inview'], subset

        self.subset = subset

        self.num_classes = 19
        self.index_to_name = {i:name for i,name in enumerate(sem_kitti_class_names)}
        self.name_to_index = {name:i for i,name in enumerate(sem_kitti_class_names)}

        self.kitti_index_to_name = {i:name for i,name in enumerate(kitti_class_names)}
        self.kitti_name_to_index = {name:i for i,name in enumerate(kitti_class_names)}

        self.class_names = sem_kitti_class_names

        self.kitti_colors = np.array(kitti_colors,np.uint8)
        self.kitti_colors_bgr = np.array([list(reversed(c)) for c in kitti_colors],np.uint8)

        self.colors = np.array(sem_kitti_colors,np.uint8)
        self.colors_bgr = np.array([list(reversed(c)) for c in sem_kitti_colors],np.uint8)

    def get(self, part, index, load_image = False):
        
        sequence_root = os.path.join(self.root, 'sequences/%s/'%(part))
        assert index <= self.length[part], index

        if load_image:
            fn_frame = os.path.join(sequence_root, 'image_2/%06d.png' % (index))
            assert os.path.exists(fn_frame), 'Broken dataset %s' % (fn_frame)
            self.frame_BGR = cv2.imread(fn_frame)
            self.frame = cv2.cvtColor(self.frame_BGR, cv2.COLOR_BGR2RGB)
            #self.frame_HSV = cv2.cvtColor(self.frame_BGR, cv2.COLOR_BGR2HSV)

        fn_velo = os.path.join(sequence_root, 'velodyne/%06d.bin' %(index))
        fn_label = os.path.join(sequence_root, 'labels/%06d.label' %(index))
        assert os.path.exists(fn_velo), 'Broken dataset %s' % (fn_velo)
        assert os.path.exists(fn_label), 'Broken dataset %s' % (fn_label)
            
        points = np.fromfile(fn_velo, dtype=np.float32).reshape(-1, 4)
        raw_label = np.fromfile(fn_label, dtype=np.uint32).reshape((-1))

        if raw_label.shape[0] == points.shape[0]:
            label = raw_label & 0xFFFF  # semantic label in lower half
            inst_label = raw_label >> 16  # instance id in upper half
            assert((label + (inst_label << 16) == raw_label).all()) # sanity check
        else:
            print("Points shape: ", points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")
        
        # Map to learning 20 classes
        label = np.array([self.learning_map[x] for x in label], dtype=np.int32)

        # Drop class -> 0
        drop_class_0 = np.where(label != 0)
        points = points[drop_class_0]
        label = label[drop_class_0] - 1
        assert (label >=0).all and (label<self.num_classes).all(), np.unique(label)

        if self.subset == 'inview':
            self.set_filter([-40, 40], [-20, 20])
            combined = self.points_basic_filter(points)
            points = points[combined]
            label = label[combined]

        return points, label
    
    def set_filter(self, h_fov, v_fov, x_range = None, y_range = None, z_range = None, d_range = None):
        self.h_fov = h_fov if h_fov is not None else (-180, 180)
        self.v_fov = v_fov if v_fov is not None else (-25, 20)
        self.x_range = x_range if x_range is not None else (-10000, 10000)
        self.y_range = y_range if y_range is not None else (-10000, 10000)
        self.z_range = z_range if z_range is not None else (-10000, 10000)
        self.d_range = d_range if d_range is not None else (-10000, 10000)

    def hv_in_range(self, m, n, fov, fov_type='h'):
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

    def box_in_range(self,x,y,z,d, x_range, y_range, z_range, d_range):
        """ extract filtered in-range velodyne coordinates based on x,y,z limit """
        return np.logical_and.reduce((
                x > x_range[0], x < x_range[1],
                y > y_range[0], y < y_range[1],
                z > z_range[0], z < z_range[1],
                d > d_range[0], d < d_range[1]))

    def points_basic_filter(self, points):
        """
            filter points based on h,v FOV and x,y,z distance range.
            x,y,z direction is based on velodyne coordinates
            1. azimuth & elevation angle limit check
            2. x,y,z distance limit
            return a bool array
        """
        assert points.shape[1] == 4, points.shape # [N,3]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) # this is much faster than d = np.sqrt(np.power(points,2).sum(1))

        # extract in-range fov points
        h_points = self.hv_in_range(x, y, self.h_fov, fov_type='h')
        v_points = self.hv_in_range(d, z, self.v_fov, fov_type='v')
        combined = np.logical_and(h_points, v_points)

        # extract in-range x,y,z points
        in_range = self.box_in_range(x,y,z,d, self.x_range, self.y_range, self.z_range, self.d_range)
        combined = np.logical_and(combined, in_range)

        return combined

    def calib_velo2cam(self, fn_v2c):
        """
        get Rotation(R : 3x3), Translation(T : 3x1) matrix info
        using R,T matrix, we can convert velodyne coordinates to camera coordinates
        """
        for line in open(fn_v2c, "r"):
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
        return R, T

    def calib_cam2cam(self, fn_c2c, mode = '02'):
        """
        If your image is 'rectified image' :get only Projection(P : 3x4) matrix is enough
        but if your image is 'distorted image'(not rectified image) :
            you need undistortion step using distortion coefficients(5 : D)
        In this code, only P matrix info is used for rectified image
        """
        # with open(fn_c2c, "r") as f: c2c_file = f.readlines()
        for line in open(fn_c2c, "r"):
            (key, val) = line.split(':', 1)
            if key == ('P_rect_' + mode):
                P = np.fromstring(val, sep=' ')
                P = P.reshape(3, 4)
                P = P[:3, :3]  # erase 4th column ([0,0,0])
        return P

    def project_3d_to_2d(self, pts_3d):
        assert pts_3d.shape[1] == 3, pts_3d.shape
        pts_3d = pts_3d.copy()
        
        # Concat and change shape from [N,3] to [N,4] to [4,N]
        one_mat = np.ones((pts_3d.shape[0], 1),dtype=np.float32)
        xyz_v = np.concatenate((pts_3d, one_mat), axis=1).T

        # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
        for i in range(xyz_v.shape[1]):
            xyz_v[:3, i] = np.matmul(self.RT, xyz_v[:, i])

        xyz_c = xyz_v[:3]

        # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        for i in range(xyz_c.shape[1]):
            xyz_c[:, i] = np.matmul(self.P, xyz_c[:, i])

        # normalize image(pixel) coordinates(x,y)
        xy_i = xyz_c / xyz_c[2]

        # get pixels location
        pts_2d = xy_i[:2].T
        return pts_2d

    def torch_project_3d_to_2d(self, pts_3d):
        assert pts_3d.shape[1] == 3, pts_3d.shape
        pts_3d = pts_3d.copy()
        
        # Create a [N,1] array
        one_mat = np.ones((pts_3d.shape[0], 1),dtype=np.float32)
        xyz_v = np.concatenate((pts_3d, one_mat), axis=1)

        RT = torch.from_numpy(self.RT).float().cuda()
        P = torch.from_numpy(self.P).float().cuda()
        xyz_v = torch.from_numpy(xyz_v).float().cuda()

        assert xyz_v.size(1) == 4, xyz_v.size()
    
        xyz_v = xyz_v.unsqueeze(2)
        RT_rep = RT.expand(xyz_v.size(0),3,4)
        P_rep = P.expand(xyz_v.size(0),3,3)

        xyz_c = torch.bmm(RT_rep, xyz_v)
        #log.info(xyz_c.shape, RT_rep.shape, xyz_v.shape)

        xy_v = torch.bmm(P_rep, xyz_c)
        #log.msg(xy_v.shape, P_rep.shape, xyz_c.shape)

        xy_i = xy_v.squeeze(2).transpose(1,0)
        xy_n = xy_i / xy_i[2]
        pts_2d = (xy_n[:2]).transpose(1,0)

        return pts_2d.detach().cpu().numpy()

    def draw_2d_points(self, pts_2d, colors, image = None):
        """ draw 2d points in camera image """
        assert pts_2d.shape[1] == 2, pts_2d.shape

        if image is None:
            image = self.frame.copy()
        pts = pts_2d.astype(np.int32).tolist()

        for (x,y),c in zip(pts, colors.tolist()):
            cv2.circle(image, (x, y), 2, c, -1)

        return image

    def draw_2d_top_view(self, pcd_3d, colors):
        """ draw 2d points in camera image """
        assert pcd_3d.shape[1] == 3, pcd_3d.shape

        image = np.zeros((600,800,3),dtype=np.uint8)

        for (x,y,z),c in zip(pcd_3d.tolist(), colors.tolist()):
            X = int(-x*800+600)
            Y = int(-y*800+400)
            cv2.circle(image, (Y,X), 3, c, -1)

        return image

    def get_max_index(self,part):
        return self.length[part]