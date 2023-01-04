#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import torch
from torch import nn as nn

from quaternion_distances import quaternion_distance
from utils.utils import quat2mat, rotate_back, rotate_forward, tvector2mat


class GeometricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sx = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.sq = torch.nn.Parameter(torch.Tensor([-3.0]), requires_grad=True)
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = torch.exp(-self.sx) * loss_transl + self.sx
        total_loss += torch.exp(-self.sq) * loss_rot + self.sq
        return total_loss


class ProposedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(ProposedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        return total_loss


class L1Loss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot):
        super(L1Loss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = self.transl_loss(rot_err, target_rot).sum(1).mean()
        total_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        return total_loss


class DistancePoints3D(nn.Module):
    def __init__(self):
        super(DistancePoints3D, self).__init__()

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations
        Returns:
            The mean distance between 3D points
        """
        #start = time.time()
        total_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)

            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            total_loss += error.mean()

        #end = time.time()
        #print("3D Distance Time: ", end-start)

        return total_loss/target_transl.shape[0]

class CombinedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot , weight_point_cloud):
        super(CombinedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')        
        self.weight_point_cloud = 0.3
        self.weight_corr = 0.3
        self.weight_pose = 0.4
        self.loss = {}

#     def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        
#         # Regression loss
#         loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
#         loss_rot = self.transl_loss(rot_err, target_rot).sum(1).mean()
#         total_loss_lt = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot
        
#         # Point Cloud Distance Loss
#         total_loss = torch.tensor([0.0]).to(transl_err.device)
#         for i in range(len(point_clouds)):
#             point_cloud_gt = point_clouds[i].to(transl_err.device)
#             point_cloud_out = point_clouds[i].clone()

#             R_target = quat2mat(target_rot[i])
#             T_target = tvector2mat(target_transl[i])
#             RT_target = torch.mm(T_target, R_target)

#             R_predicted = quat2mat(rot_err[i])
#             T_predicted = tvector2mat(transl_err[i])
#             RT_predicted = torch.mm(T_predicted, R_predicted)

#             RT_total = torch.mm(RT_target.inverse(), RT_predicted)
            
#             point_cloud_out = point_cloud_out.cuda()
#             point_cloud_out = rotate_forward(point_cloud_out, RT_total)

#             error = (point_cloud_out - point_cloud_gt).norm(dim=0)
#             error.clamp(100.)
#             total_loss += error.mean()
        
#         total_loss_lp = total_loss/target_transl.shape[0]
        
#         loss = total_loss_lt + self.weight_point_cloud*total_loss_lp
        
#         return loss

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err ):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            target_transl: groundtruth of the translations
            target_rot: groundtruth of the rotations
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations
        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = self.rescale_trans*loss_transl + self.rescale_rot*loss_rot

        #start = time.time()
        point_clouds_loss = torch.tensor([0.0]).to(transl_err.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            point_cloud_out = point_clouds[i].clone()

            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)

            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_total = torch.mm(RT_target.inverse(), RT_predicted)
            point_cloud_out = point_cloud_out.cuda()
            point_cloud_out = rotate_forward(point_cloud_out, RT_total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            point_clouds_loss += error.mean()
        
#         corr_loss = torch.nn.functional.mse_loss(corr_pred, corr_target)
        
#         if mask.sum() > 0:
# #             print('enter cyclic loss sum')
#             cycle_loss = torch.nn.functional.mse_loss(cycle[mask], queries[mask])
#             corr_loss += cycle_loss        
        
        #end = time.time()
        #print("3D Distance Time: ", end-start)
        total_loss = self.weight_pose * pose_loss +\
                     self.weight_point_cloud * (point_clouds_loss/target_transl.shape[0]) # + self.weight_corr * corr_loss
#         total_loss = self.weight_pose * pose_loss +\
#                      self.weight_point_cloud * (point_clouds_loss/target_transl.shape[0]) 
        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        self.loss['point_clouds_loss'] = point_clouds_loss/target_transl.shape[0]
        # self.loss['corr_loss'] = corr_loss
        
        return self.loss