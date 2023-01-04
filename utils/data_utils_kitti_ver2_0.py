import logging

import random
import torch
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate
from .DatasetLidarCamera_Ver9_3_aws import DatasetLidarCameraKittiOdometry


logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    dataset_class = DatasetLidarCameraKittiOdometry

    dataset_train = dataset_class(args.data_folder , max_r=args.max_r, max_t=args.max_t,
                                  split='train', use_reflectance=args.use_reflectance,
                                  val_sequence=args.val_sequence)
    dataset_val = dataset_class(args.data_folder , max_r=args.max_r, max_t=args.max_t,
                                split='val', use_reflectance=args.use_reflectance,
                                val_sequence=args.val_sequence)
    n_dataset_train = len(dataset_train)
    n_dataset_val = len(dataset_val)
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(dataset_train) if args.local_rank == -1 else DistributedSampler(dataset_val)
    val_sampler = SequentialSampler(dataset_val)
    
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                #  shuffle=True,
                                                 sampler = train_sampler,
                                                 batch_size=args.train_batch_size,
                                                 num_workers=args.num_worker,
                                                #  worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                # shuffle=False, 
                                                sampler = val_sampler, 
                                                batch_size=args.eval_batch_size,
                                                num_workers=args.num_worker,
                                                # worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True) if dataset_val is not None else None

    return train_loader, val_loader , n_dataset_train , n_dataset_val

# EPOCH = 1
# def init_fn(worker_id, seed):
#     seed = seed + worker_id + EPOCH*100
#     print(f"Init worker {worker_id} with seed {seed}")
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)

def merge_inputs(queries):
    point_clouds = []
    imgs = []
    reflectances = []
    corrs =[]
    pc_rotated =[]
#     returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
#                if key != 'point_cloud' and key != 'rgb' and key != 'reflectance' }
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
           if key != 'point_cloud' and key != 'rgb' and key != 'reflectance' and key != 'corrs' and key != 'pc_rotated' }
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['rgb'])
        corrs.append(input['corrs'])
        pc_rotated.append(input['pc_rotated'])
        if 'reflectance' in input:
            reflectances.append(input['reflectance'])
    returns['point_cloud'] = point_clouds
    returns['rgb'] = imgs
    returns['corrs'] = corrs
    returns['pc_rotated'] = pc_rotated
    if len(reflectances) > 0:
        returns['reflectance'] = reflectances
    return returns