
import torch
import numpy as np
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.cm as cm

def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z

def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.numpy()
    # cam_intrinsic = cam_calib
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
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
    # points_index1 = np.arange(pcl_uv_no_mask.shape[0])[mask1]
    
    return depth_img, pcl_uv , pcl_z , points_index 
    
def trim_corrs(in_corrs):
    length = in_corrs.shape[0]
#         print ("number of keypoint before trim : {}".format(length))
    if length >= self.num_kp:
        mask = np.random.choice(length, self.num_kp)
        return in_corrs[mask]
    else:
        mask = np.random.choice(length, self.num_kp - length)
        return np.concatenate([in_corrs, in_corrs[mask]], axis=0)

def knn(x, y ,k):
# #         print (" x shape = " , x.shape)
#         inner = -2*torch.matmul(x.transpose(-2, 1), x)
#         xx = torch.sum(x**2, dim=1, keepdim=True)
# #         print (" xx shape = " , x.shape)
#         pairwise_distance = -xx - inner - xx.transpose(4, 1)
    pairwise_distance = F.pairwise_distance(x,y)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def two_images_side_by_side(img_a, img_b):
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
def dense_map(Pts ,n, m, grid):
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

def colormap(disp):
    """"Color mapping for disp -- [H, W] -> [3, H, W]"""
#         disp_np = disp.cpu().numpy()        # tensor -> numpy
    disp_np = disp
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  #magma, plasma, etc.
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3])
    # return colormapped_im.transpose(2, 0, 1)
    return colormapped_im

# corr dataset generation 
def corr_gen( gt_points_index, points_index, gt_uv, uv , num_kp = 500) :
    
    inter_gt_uv_mask = np.in1d(gt_points_index , points_index)
    inter_uv_mask    = np.in1d(points_index , gt_points_index)
    gt_uv = gt_uv[inter_gt_uv_mask]
    uv    = uv[inter_uv_mask] 
    corrs = np.concatenate([gt_uv, uv], axis=1)
    corrs = torch.tensor(corrs)

    corrs[:, 0] = (0.5*corrs[:, 0])/1280
    corrs[:, 1] = (0.5*corrs[:, 1])/384
    corrs[:, 2] = (0.5*corrs[:, 2])/1280 + 0.5        
    corrs[:, 3] = (0.5*corrs[:, 3])/384   

    if corrs.shape[0] <= num_kp :
        corrs = torch.zeros(num_kp, 4)
        corrs[:, 2] = corrs[:, 2] + 0.5

    corrs_knn_idx = knn(corrs[:,:2], corrs[:,2:], num_kp) # knn 2d point-cloud trim
    corrs = corrs[corrs_knn_idx]               

    assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
    assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
    assert (0.5 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
    assert (0.0 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()
    
    return corrs

# for displying correspondence matching 
def draw_corrs(self, imgs, corrs, col=(255, 0, 0)):
    imgs = utils.torch_img_to_np_img(imgs)
    out = []
    for img, corr in zip(imgs, corrs):
        img = np.interp(img, [img.min(), img.max()], [0, 255]).astype(np.uint8)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
#             corr *= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
        corr *= np.array([1280,384,1280,384])
        for c in corr:
            draw.line(c, fill=col)
        out.append(np.array(img))
    out = np.array(out) / 255.0
    return utils.np_img_to_torch_img(out) , out   