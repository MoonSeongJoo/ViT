# -------------------------------------------------------------------
# Copyright (C) 2020 Harbin Institute of Technology, China
# Author: Xudong Lv (15B901019@hit.edu.cn)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import csv
import os
from math import radians
import cv2

import h5py
import mathutils
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import minmax_scale

from .utils import invert_pose, rotate_forward, quaternion_from_matrix ,rotate_back #, read_calib_file
from pykitti import odometry
import pykitti
import matplotlib.pyplot as plt
from torchvision.transforms import functional as tvtf
import matplotlib as mpl
import matplotlib.cm as cm

# def get_calib_kitti_odom(sequence):
#     if sequence == 0:
#         return torch.tensor([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]])
#     elif sequence == 3:
#         return torch.tensor([[721.5377, 0, 609.5593], [0, 721.5377, 172.854], [0, 0, 1]])
#     elif sequence in [5, 6, 7, 8, 9]:
#         return torch.tensor([[707.0912, 0, 601.8873], [0, 707.0912, 183.1104], [0, 0, 1]])
#     else:
#         raise TypeError("Sequence Not Available")

"""
root_dir = "/mnt/data/kitti_odometry"
calib_path ="data_odometry_calib"
image_path ="data_odometry_color"
velodyne_path = "data_odometry_velodyne"
imagegray_path = "data_odometry_gray"
poses_path = "data_odometry_poses"
val_RT_path = "data_odometry_valRT"
"""

class DatasetLidarCameraKittiOdometry(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=0.2, max_r=10., split='val', device='cpu', val_sequence='00', suf='.png'):
        super(DatasetLidarCameraKittiOdometry, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.K = {}
        self.suf = suf
        self.img_shape =(384,1280)
        self.num_kp = 500
        # print ("number of kp = " , self.num_kp)
        
        self.all_files = []
        self.sequence_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
                
        self.calib_path ="data_odometry_calib"
        self.image_path ="data_odometry_color"
        self.velodyne_path = "data_odometry_velodyne"
        self.imagegray_path = "data_odometry_gray"
        self.poses_path = "data_odometry_poses"
        self.val_RT_path = "data_odometry_valRT"
        
        self.calib_path_total = os.path.join(dataset_dir,self.calib_path,"dataset")
        self.image_path_total = os.path.join(dataset_dir,self.image_path,"dataset")
        self.imagegray_path_total = os.path.join(dataset_dir,self.imagegray_path,"dataset")
        self.velodyne_path_total = os.path.join(dataset_dir,self.velodyne_path,"dataset")
        self.poses_path_total = os.path.join(dataset_dir,self.poses_path,"dataset","poses")
        self.val_RT_path_total = os.path.join(dataset_dir,self.val_RT_path,"dataset")
        
        # self.model = CameraModel()
        # self.model.focal_length = [7.18856e+02, 7.18856e+02]
        # self.model.principal_point = [6.071928e+02, 1.852157e+02]
        # for seq in ['00', '03', '05', '06', '07', '08', '09']:
        for seq in self.sequence_list:
            # odom = odometry(self.calib_path_total,self.poses_path_total, seq)
            odom = odometry(self.calib_path_total, seq)
            calib = odom.calib
            T_cam02_velo_np = calib.T_cam2_velo #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
            self.K[seq] = calib.K_cam2 # 3x3
            # T_cam02_velo = torch.from_numpy(T_cam02_velo_np)
            # GT_R = quaternion_from_matrix(T_cam02_velo[:3, :3])
            # GT_T = T_cam02_velo[3:, :3]
            # self.GTs_R[seq] = GT_R # GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
            # self.GTs_T[seq] = GT_T # GT_T = np.array([row['x'], row['y'], row['z']])
            self.GTs_T_cam02_velo[seq] = T_cam02_velo_np #gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)

            image_list = os.listdir(os.path.join(self.image_path_total, 'sequences', seq, 'image_2'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(self.velodyne_path_total, 'sequences', seq, 'velodyne',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(self.image_path_total, 'sequences', seq, 'image_2',
                                                   str(image_name.split('.')[0])+suf)):
                    continue
                if seq == val_sequence:
                    if split.startswith('val') or split == 'test':
                        self.all_files.append(os.path.join(seq, image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train':
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            # val_RT_file = os.path.join(dataset_dir, 'sequences',
            #                            f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            val_RT_sequences_path = os.path.join(self.val_RT_path_total,"sequences")
            val_RT_file = os.path.join(self.val_RT_path_total, 'sequences',
                                       f'val_RT_left_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            
            if not os.path.exists(val_RT_sequences_path):
                os.makedirs(val_RT_sequences_path)
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([i, transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb
    def get_2D_lidar_projection(self,pcl, cam_intrinsic):
        pcl_xyz = cam_intrinsic @ pcl.T
        pcl_xyz = pcl_xyz.T
        pcl_z = pcl_xyz[:, 2]
        pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
        pcl_uv = pcl_xyz[:, :2]

        return pcl_uv, pcl_z

    def lidar_project_depth(self,pc_rotated, cam_calib, img_shape):
        pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
        #cam_intrinsic = cam_calib.numpy()
        cam_intrinsic = cam_calib
        pcl_uv, pcl_z = self.get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
        mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
                pcl_uv[:, 1] < img_shape[0] ) & (pcl_z > 0)
        mask1 = (pcl_uv[:, 1] < 188)
        pcl_uv_no_mask = pcl_uv
        pcl_z_no_mask = pcl_z
        pcl_uv = pcl_uv[mask]
        pcl_z = pcl_z[mask]
        pcl_uv = pcl_uv.astype(np.uint32)
        pcl_uv_no_mask  = pcl_uv_no_mask.astype(np.uint32) 
        pcl_z = pcl_z.reshape(-1, 1)
        depth_img = np.zeros((img_shape[0], img_shape[1], 1))
        depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
        depth_img = torch.from_numpy(depth_img.astype(np.float32))
        pcl_uv = torch.from_numpy(pcl_uv.astype(np.float32))
        pcl_uv_no_mask = torch.from_numpy(pcl_uv_no_mask.astype(np.float32))
        pcl_z_no_mask = torch.from_numpy(pcl_z_no_mask.astype(np.float32))
        #depth_img = depth_img.cuda()
        depth_img = depth_img.permute(2, 0, 1)
        points_index = np.arange(pcl_uv_no_mask.shape[0])[mask]
        points_index1 = np.arange(pcl_uv_no_mask.shape[0])[mask1]
        
        return depth_img, pcl_uv , pcl_uv_no_mask , pcl_z , mask , points_index , points_index1
        
    def trim_corrs(self, in_corrs):
        length = in_corrs.shape[0]
#         print ("number of keypoint before trim : {}".format(length))
        if length >= self.num_kp:
            mask = np.random.choice(length, self.num_kp)
            return in_corrs[mask]
        else:
            mask = np.random.choice(length, self.num_kp - length)
            return np.concatenate([in_corrs, in_corrs[mask]], axis=0)

    def knn(self, x, y ,k):
# #         print (" x shape = " , x.shape)
#         inner = -2*torch.matmul(x.transpose(-2, 1), x)
#         xx = torch.sum(x**2, dim=1, keepdim=True)
# #         print (" xx shape = " , x.shape)
#         pairwise_distance = -xx - inner - xx.transpose(4, 1)
        pairwise_distance = F.pairwise_distance(x,y)

        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
        return idx

    def two_images_side_by_side(self, img_a, img_b):
        assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
        assert img_a.dtype == img_b.dtype
        h, w, c = img_a.shape
#         b,h, w, c = img_a.shape
        canvas = np.zeros((h, 2 * w, c), dtype=img_a.dtype)
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.dtype)
        canvas[:, 0 * w:1 * w, :] = img_a
        canvas[:, 1 * w:2 * w, :] = img_b
#         canvas = np.zeros((b, h, 2 * w, c), dtype=img_a.cpu().numpy().dtype)
#         canvas[:, :, 0 * w:1 * w, :] = img_a.cpu().numpy()
#         canvas[:, :, 1 * w:2 * w, :] = img_b.cpu().numpy()

        #canvas[:, :, : , 0 * w:1 * w] = img_a.cpu().numpy()
        #canvas[:, :, : , 1 * w:2 * w] = img_b.cpu().numpy()
        return canvas
    
    # From Github https://github.com/balcilar/DenseDepthMap
    def dense_map(self, Pts ,n, m, grid):
        ng = 2 * grid + 1

        mX = np.zeros((m,n)) + np.float("inf")
        mY = np.zeros((m,n)) + np.float("inf")
        mD = np.zeros((m,n))
        mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
        mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
        mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]

        KmX = np.zeros((ng, ng, m - ng, n - ng))
        KmY = np.zeros((ng, ng, m - ng, n - ng))
        KmD = np.zeros((ng, ng, m - ng, n - ng))

        for i in range(ng):
            for j in range(ng):
                KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
                KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
                KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
        S = np.zeros_like(KmD[0,0])
        Y = np.zeros_like(KmD[0,0])

        for i in range(ng):
            for j in range(ng):
                s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
                Y = Y + s * KmD[i,j]
                S = S + s

        S[S == 0] = 1
        out = np.zeros((m,n))
        out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
        return out 
    
    def colormap(self, disp):
        """"Color mapping for disp -- [H, W] -> [3, H, W]"""
#         disp_np = disp.cpu().numpy()        # tensor -> numpy
        disp_np = disp
        vmax = np.percentile(disp_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  #magma, plasma, etc.
        colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3])
        return colormapped_im.transpose(2, 0, 1)
#         return colormapped_im
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        img_path = os.path.join(self.image_path_total, 'sequences', seq, 'image_2', rgb_name+self.suf)
        lidar_path = os.path.join(self.velodyne_path_total, 'sequences', seq, 'velodyne', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        # if self.use_reflectance:
        #     reflectance = pc[:, 3].copy()
        #     reflectance = torch.from_numpy(reflectance).float()

        RT_torch = self.GTs_T_cam02_velo[seq].astype(np.float32)

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")
        
        pc_rot = np.matmul(RT_torch, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[1, :] *= -1

#         img = Image.open(img_path).convert('RGB')
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         print ('img raw shape' , np.asarray(img).shape)
        
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
#         try:
#             img = self.custom_transform(img, img_rotation, h_mirror)
#         except OSError:
#             new_idx = np.random.randint(0, self.__len__())
#             return self.__getitem__(new_idx)
        
        
        # Rotate PointCloud for img_rotation
        if self.split == 'train':
            #R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            R = mathutils.Euler((radians(img_rotation), 0, 0))
            #R1 = R.to_matrix()
            #print ("---------euler--------" , R1)
            #Rx = mathutils.Matrix.Rotation(radians(img_rotation), 3, 'X')
            #Ry = mathutils.Matrix.Rotation(0, 3, 'Y')
            #Rz = mathutils.Matrix.Rotation(0, 3, 'Z')
            #R = Rz * Ry * Rx # ( XYZ order)
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 특정 범위 내에서 교정 매개 변수의 외란 값을 무작위로 설정
        # train 매번 무작위로 생성되며 각 시대마다 다른 매개 변수를 사용합니다.
        # 초기화시 test가 미리 설정되어 있으며 각 Epoch는 동일한 매개 변수를 사용합니다.
        #R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        R = mathutils.Euler((rotx, roty, rotz))
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R_torch, T_torch = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = self.K[seq]
        if h_mirror:
            calib[2] = (img_rgb.shape[2] / 2)*2 - calib[2]
        
#         print('img_raw_shape' , img.shape)
#         img = img.permute(1,2,0)
#         img_np = img.cpu().numpy()
#         img_np_resized = cv2.resize(img_np, (640,192), interpolation=cv2.INTER_LINEAR)

#         fig = plt.figure(figsize=(10,20))
#         plt.axis('off')
#         fig=plt.imshow(img_np_resized)
#         plt.show()
         
        real_shape = [376 , 1241 ,3]
#         real_shape = img_np_resized.shape
#         print ('real_shape=' , real_shape)
#         print('-------- pc_gt shape ----------- ' ,pc_in.shape)
        depth_gt, gt_uv ,gt_uv_nomask ,gt_z ,gt_mask , gt_points_index , gt_points_index1 = self.lidar_project_depth(pc_in, calib , real_shape) # image_shape
        depth_gt /= 80.
#         print('-------- gt_uv shape ----------- ' ,gt_uv.shape)
#         print(f' gt_uv shape = {gt_uv.shape}', end='gt_uv end \n')        
        
        R = mathutils.Quaternion(R).to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(T)
        #RT = T * R
        RT = T @ R # version change matutils * --> @ 

        pc_rotated = rotate_back(pc_in, RT) # Pc` = RT * Pc        
        
#         print('-------- pc_rotate shape ----------- ' ,pc_rotated.shape)
        depth_img, uv , uv_nomask , z , mask , points_index , points_index1 = self.lidar_project_depth(pc_rotated, calib , real_shape) # image_shape
        depth_img /= 80.
        
#         print('-------depth_img_shape-------' , depth_img.shape)
        lidarOnImage = np.hstack([uv, z])
        dense_depth_img = self.dense_map(lidarOnImage.T , 1241, 376 , 8)
        dense_depth_img = torch.tensor(dense_depth_img)
        
#         rgb_np_resized = img.resize((640, 192), Image.LANCZOS)
        img_rgb = cv2.resize(img_rgb, (640,192), interpolation=cv2.INTER_LINEAR)
        rgb_img = transforms.ToTensor()(img_rgb)
        img_bgr_resized = cv2.resize(img_bgr, (640,192), interpolation=cv2.INTER_LINEAR)
        img_gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
        img_gray = np.expand_dims(img_gray, axis=0)
        img_gray = np.transpose(img_gray,(1,2,0))
        img_gray = transforms.ToTensor()(img_gray)
        
#         rgb_img = self.transform(img)
#         rgb_img = img.resize((1280, 384), Image.LANCZOS)

        depth_gt_np = depth_gt.permute(1,2,0)
        depth_gt_np = depth_gt_np.cpu().numpy()
        depth_gt_np_resized = cv2.resize(depth_gt_np, (640,192), interpolation=cv2.INTER_LINEAR)
        input_lidar_gt_pytorch = transforms.ToTensor()(depth_gt_np_resized)
          
        dense_depth_img_np = dense_depth_img.unsqueeze(dim=0).permute(1,2,0)
        dense_depth_img_np = dense_depth_img_np.cpu().numpy().astype(np.uint8)
        dense_depth_img_np_resized = cv2.resize(dense_depth_img_np, (640,192), interpolation=cv2.INTER_LINEAR)
        dense_depth_img_np_resized = np.expand_dims(dense_depth_img_np_resized, axis=0)
        dense_depth_img_np_resized = np.transpose(dense_depth_img_np_resized,(1,2,0))
        dense_depth_img_np_resized = transforms.ToTensor()(dense_depth_img_np_resized)
        # dense_depth_img_np_resized_color = self.colormap(dense_depth_img_np_resized)
        # dense_depth_img= transforms.ToTensor()(dense_depth_img_np_resized_color)
        
        # sbs_img = self.two_images_side_by_side(img_gray, dense_depth_img_np_resized)
        # sbs_img = torch.from_numpy(sbs_img).type(torch.float32)
        # sbs_img = transforms.ToTensor()(sbs_img).type(torch.float32) #.permute(1,2,0)
        # sbs_img = tvtf.normalize(sbs_img.permute(2,1,0), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        ########## corr dataset generation #############################

        inter_gt_uv_mask = np.in1d(gt_points_index , points_index)
#         inter_gt_uv_mask1 = np.in1d(inter_gt_uv_mask , points_index1)
        inter_uv_mask    = np.in1d(points_index , gt_points_index)
#         inter_uv_mask1    = np.in1d(inter_uv_mask , gt_points_index1)

        gt_uv = gt_uv[inter_gt_uv_mask]
        uv    = uv[inter_uv_mask]   

        corrs = np.concatenate([gt_uv, uv], axis=1)
        corrs = torch.tensor(corrs)
        
        corrs[:, 0] = (0.5*corrs[:, 0])/1280
        corrs[:, 1] = (0.5*corrs[:, 1])/384
        corrs[:, 2] = (0.5*corrs[:, 2])/1280 + 0.5        
        corrs[:, 3] = (0.5*corrs[:, 3])/384
        
#         valid_mask = (corrs[:, 1] < 0.5) & (corrs[:, 3] < 0.5) 
#         corrs = corrs[valid_mask]
        if corrs.shape[0] <= self.num_kp :
            corrs = torch.zeros(self.num_kp, 4)
            corrs[:, 2] = corrs[:, 2] + 0.5
        
#         corrs = self.trim_corrs(corrs) # random 2d point-cloud trim
        corrs_knn_idx = self.knn(corrs[:,:2], corrs[:,2:], self.num_kp) # knn 2d point-cloud trim
        corrs = corrs[corrs_knn_idx]
#         print ("knn corrs shape = " , corrs.shape)

        assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
        assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
        assert (0.5 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
        assert (0.0 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()         
#         print ("corrs_MinMax_scaled_value =" , np.max(corrs) )
#         corrs -= corrs.min()
#         corrs /= corrs.max()
#         print ("-------corrs_max---------" , np.max(corrs[:,3]))
        
        if self.split == 'test':
            sample = {'rgb': img_gray, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T_torch, 'rot_error': R_torch, 'seq': int(seq), 'img_path': img_path,
                      'rgb_name': rgb_name + '.png', 'item': item, 'extrin': RT_torch,
                      'initial_RT': initial_RT , 'corrs' : corrs , 'pc_rotated' : pc_rotated , 'lidar_gt' : input_lidar_gt_pytorch,
                      'dense_depth_img' : dense_depth_img_np_resized}
        else:
            sample = {'rgb': img_gray, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T_torch, 'rot_error': R_torch, 'seq': int(seq),
                      'rgb_name': rgb_name, 'item': item, 'extrin': RT_torch , 
                      'corrs' : corrs , 'pc_rotated' : pc_rotated , 'lidar_gt' : input_lidar_gt_pytorch,
                      'dense_depth_img' : dense_depth_img_np_resized}

        return sample


class DatasetLidarCameraKittiRaw(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=15.0, split='val', device='cpu', val_sequence='2011_09_26_drive_0117_sync'):
        super(DatasetLidarCameraKittiRaw, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.max_depth = 80
        self.K_list = {}

        self.all_files = []
        date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        data_drive_list = ['0001', '0002', '0004', '0016', '0027']
        self.calib_date = {}

        for i in range(len(date_list)):
            date = date_list[i]
            data_drive = data_drive_list[i]
            data = pykitti.raw(self.root_dir, date, data_drive)
            calib = {'K2': data.calib.K_cam2, 'K3': data.calib.K_cam3,
                     'RT2': data.calib.T_cam2_velo, 'RT3': data.calib.T_cam3_velo}
            self.calib_date[date] = calib

        # date = val_sequence[:10]
        # seq = val_sequence
        # image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
        # image_list.sort()
        #
        # for image_name in image_list:
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
        #                                        str(image_name.split('.')[0]) + '.bin')):
        #         continue
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
        #                                        str(image_name.split('.')[0]) + '.jpg')):  # png
        #         continue
        #     self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))

        date = val_sequence[:10]
        test_list = ['2011_09_26_drive_0005_sync', '2011_09_26_drive_0070_sync', '2011_10_03_drive_0027_sync']
        seq_list = os.listdir(os.path.join(self.root_dir, date))

        for seq in seq_list:
            if not os.path.isdir(os.path.join(dataset_dir, date, seq)):
                continue
            image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
                                                   str(image_name.split('.')[0])+'.jpg')): # png
                    continue
                if seq == val_sequence and (not split == 'train'):
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train' and seq not in test_list:
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(dataset_dir,
                                       f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                           rotx, roty, rotz])
                    self.val_RT.append([float(i), transl_x, transl_y, transl_z,
                                         rotx, roty, rotz])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            # color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    # self.all_files.append(os.path.join(date, seq, 'image_2/data', image_name.split('.')[0]))
    def __getitem__(self, idx):
        item = self.all_files[idx]
        date = str(item.split('/')[0])
        seq = str(item.split('/')[1])
        rgb_name = str(item.split('/')[4])
        img_path = os.path.join(self.root_dir, date, seq, 'image_02/data', rgb_name+'.jpg') # png
        lidar_path = os.path.join(self.root_dir, date, seq, 'velodyne_points/data', rgb_name+'.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_lidar = pc.copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        if self.use_reflectance:
            reflectance = pc[:, 3].copy()
            reflectance = torch.from_numpy(reflectance).float()

        calib = self.calib_date[date]
        RT_cam02 = calib['RT2'].astype(np.float32)
        # camera intrinsic parameter
        calib_cam02 = calib['K2']  # 3x3

        E_RT = RT_cam02

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        pc_rot = np.matmul(E_RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[0, :] *= -1

        img = Image.open(img_path)
        # print(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        # if self.split == 'train':
        #     R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
        #     T = mathutils.Vector((0., 0., 0.))
        #     pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
            initial_RT = 0
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = calib_cam02
        # calib = get_calib_kitti_odom(int(seq))
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        # sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
        #           'tr_error': T, 'rot_error': R, 'seq': int(seq), 'rgb_name': rgb_name, 'item': item,
        #           'extrin': E_RT, 'initial_RT': initial_RT}
        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
                  'tr_error': T, 'rot_error': R, 'rgb_name': rgb_name + '.png', 'item': item,
                  'extrin': E_RT, 'initial_RT': initial_RT, 'pc_lidar': pc_lidar}

        return sample

