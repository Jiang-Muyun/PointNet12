class PointNetColorGen(nn.Module):
    def __init__(self,num_class, input_dims=4, feature_transform=False):
        super(PointNetColorGen, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False,input_dims = input_dims, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat
        
class ColorGeneratorLoader(Dataset):
    def __init__(self, root, npoints, train = True):
        self.root = root
        self.train = train
        self.npoints = npoints
        self.np_redis = Mat_Redis_Utils()
        self.utils = Semantic_KITTI_Utils(root,'inview','learning')

        part_length = {'00': 4540,'01':1100,'02':4660,'03':800,'04':270,'05':2760,
            '06':1100,'07':1100,'08':4070,'09':1590,'10':1200}

        self.keys = []
        alias = 'gen'
        
        if self.train:
            for part in ['00','01','02','03','04','05','06','07','09','10']:
                length = part_length[part]
                for index in range(0,length,2):
                    self.keys.append('%s/%s/%06d'%(alias, part, index))
        else:
            for part in ['08']:
                length = part_length[part]
                for index in range(0,length,2):
                    self.keys.append('%s/%s/%06d'%(alias, part, index))

    def __len__(self):
            return len(self.keys)

    def get_data(self, key):
        if not self.np_redis.exists(key):
            alias, part, index = key.split('/')

            point_cloud, _ = self.utils.get(part, int(index), load_image = True)
            pts_2d = self.utils.project_3d_to_2d(point_cloud[:,:3]).astype(np.int32)
            pts_color = np.zeros((point_cloud.shape[0],3), dtype=np.float32)

            frame_shape = self.utils.frame.shape
            for i,(y,x) in enumerate(pts_2d):
                if x >= 0 and x < frame_shape[0] and y >= 0 and y < frame_shape[1]:
                    pts_color[i] = self.utils.frame_HSV[x,y]
            # img = self.utils.draw_2d_points(pts_2d, pts_color, on_black=True)
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
            pts_color[:,0] /= 180
            pts_color[:,1:] /= 255

            to_store = np.concatenate((point_cloud, pts_color),axis=1)
            self.np_redis.set(key, to_store)
            print('add', key, to_store.shape, to_store.dtype)
        else:
            data = self.np_redis.get(key)
            point_cloud = data[:,:4]
            pts_color = data[:,4:]
        
        # Unnormalized Point cloud
        return point_cloud, pts_color

    def __getitem__(self, index):
        point_cloud, label = self.get_data(self.keys[index])
        #pcd = point_cloud
        pcd = pcd_normalize(point_cloud)
        if self.train:
            pcd = pcd_jitter(pcd)

        length = pcd.shape[0]
        if length == self.npoints:
            pass
        elif length > self.npoints:
            start_idx = np.random.randint(0, length - self.npoints)
            end_idx = start_idx + self.npoints
            pcd = pcd[start_idx:end_idx]
            label = label[start_idx:end_idx]
        else:
            rows_short = self.npoints - length
            pcd = np.concatenate((pcd,pcd[0:rows_short]),axis=0)
            label = np.concatenate((label,label[0:rows_short]),axis=0)
        return pcd, label