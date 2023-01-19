# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import math
import time
import numpy as np
import mathutils

from datetime import timedelta

import torch
import torch.distributed as dist
from torchvision import transforms
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils_kitti_ver3_0 import get_loader
from utils.dist_util import get_world_size

from losses_Ver9_3_aws import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss, CombinedLoss
from quaternion_distances import quaternion_distance
from image_processing_unit import lidar_project_depth , corr_gen , dense_map , colormap
from utils.utils import  rotate_back

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    loss_fn = CombinedLoss(args.rescale_transl, args.rescale_rot , args.weight_point_cloud)
    # model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model , loss_fn


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step , loss_fn , n_data_test):
    # Validation!
    
    total_val_loss = 0.
    total_val_t = 0.
    total_val_r = 0.
    local_loss = 0.0
    
    # eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
 
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (eval_loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
   
    for step, batch in enumerate(epoch_iterator):
        # batch = tuple(t.to(args.device) for t in batch)
        # x, y = batch
        start_time = time.time()
        rgb_input = []
        lidar_gt_input = []
        dense_depth_img_input =[]       
        
        # gt pose
        # batch['tr_error'] = batch['tr_error'].cuda()
        # batch['rot_error'] = batch['rot_error'].cuda()
        target_transl = batch['tr_error'].cuda()
        target_rot = batch['rot_error'].cuda()
        
        for idx in range(len(batch['rgb'])):
            rgb = batch['rgb'][idx].cuda()
            dense_depth_img = batch['dense_depth_img'][idx].cuda()
            # batch stack 
            rgb_input.append(rgb)
            dense_depth_img_input.append(dense_depth_img)
        
        rgb_input = torch.stack(rgb_input)
        dense_depth_img_input = torch.stack(dense_depth_img_input)
        dense_depth_img_input = dense_depth_img_input.permute(0,2,3,1)
        # sbs_img = torch.cat((rgb_input,dense_depth_img_input),1)    
        
        with torch.no_grad():
            # logits = model(x)[0]
            transl_err , rot_err = model(rgb_input, dense_depth_img_input)
            eval_loss = loss_fn(batch['point_cloud'], target_transl, target_rot, transl_err, rot_err)

            
            total_trasl_error = torch.tensor(0.0).cuda()
            # target_transl = torch.tensor(target_transl).cuda()
            total_rot_error = quaternion_distance(target_rot, rot_err, target_rot.device)
            total_rot_error = total_rot_error * 180. / math.pi
            for j in range(rgb_input.shape[0]):
                total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.
            
            total_err_transl = total_trasl_error.item()
            total_err_rot = total_rot_error.sum().item()
        
        total_val_t += total_err_transl
        total_val_r += total_err_rot    
        local_loss += eval_loss['total_loss'].item()
        
        if step % 50 == 0 and step != 0:
            print('Iter %d val loss = %.3f , time = %.2f' % (step, local_loss/50.,
                                                                  (time.time() - start_time)/rgb_input.shape[0]))
            local_loss = 0.0
        
        total_val_loss += eval_loss['total_loss'].item() * len(batch['rgb'])

            # eval_loss = loss_fct(logits, y)
    #         eval_losses.update(eval_loss.item())

    #         preds = torch.argmax(logits, dim=-1)

    #     if len(all_preds) == 0:
    #         all_preds.append(preds.detach().cpu().numpy())
    #         all_label.append(y.detach().cpu().numpy())
    #     else:
    #         all_preds[0] = np.append(
    #             all_preds[0], preds.detach().cpu().numpy(), axis=0
    #         )
    #         all_label[0] = np.append(
    #             all_label[0], y.detach().cpu().numpy(), axis=0
    #         )
    #     epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    # all_preds, all_label = all_preds[0], all_label[0]
    # accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Total Valid Loss: %2.5f" % (total_val_loss / n_data_test))  # n_data_test = number of validate set row
    logger.info("total traslation error: %2.5f" % (total_val_t / n_data_test))
    logger.info("total rotation error: %2.5f" % (total_val_r / n_data_test))

    writer.add_scalar("test/accuracy", scalar_value=(total_val_loss/n_data_test), global_step=global_step)
    return (total_val_loss / n_data_test)


def train(args, model , loss_fn):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader ,n_data_train , n_data_test = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, sample in enumerate(epoch_iterator):
            # batch = tuple(t.to(args.device) for t in batch)
            # print(f'batch {step+1}/{len(epoch_iterator)}', end='\r')
            # x, y = batch
            
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []
            # corrs_input =[]
            
            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()
            
            for idx in range(len(sample['rgb'])):
                real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]
                rgb = sample['rgb'][idx]
                rgb = transforms.ToTensor()(rgb)
                
                #calibrated point cloud 2d projection
                pc_lidar = sample['point_cloud'][idx].clone()
                if args.max_depth < 80.:
                    pc_lidar = pc_lidar[:, pc_lidar[0, :] < args.max_depth].clone()
                
                depth_gt, gt_uv , gt_z , gt_points_index  = lidar_project_depth(pc_lidar, sample['calib'][idx], real_shape) # image_shape
                depth_gt /= args.max_depth
                
                # mis-calibrated point cloud 2d projection
                R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
                R.resize_4x4()
                T = mathutils.Matrix.Translation(sample['tr_error'][idx])
                RT = T @ R

                pc_rotated = rotate_back(sample['point_cloud'][idx], RT) # Pc` = RT * Pc
                if args.max_depth < 80.:
                    pc_rotated = pc_rotated[:, pc_rotated[0, :] < args.max_depth].clone()
                
                depth_img, uv , z, points_index = lidar_project_depth(pc_rotated, sample['calib'][idx], real_shape) # image_shape
                depth_img /= args.max_depth
                
                lidarOnImage = np.hstack([uv, z])
                dense_depth_img = dense_map(lidarOnImage.T , real_shape[1], real_shape[0] , 8) # argument = (lidarOnImage.T , 1241, 376 , 8)
                dense_depth_img = dense_depth_img.astype(np.uint8)
                dense_depth_img_color = colormap(dense_depth_img)
                dense_depth_img_color = transforms.ToTensor()(dense_depth_img_color).type(dtype=torch.float32)
                
                # PAD ONLY ON RIGHT AND BOTTOM SIDE
                shape_pad = [0, 0, 0, 0]

                shape_pad[3] = (args.img_shape[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (args.img_shape[1] - rgb.shape[2])  # // 2 + 1

                rgb = F.pad(rgb, shape_pad)
                # depth_img = F.pad(depth_img, shape_pad)
                depth_gt = F.pad(depth_gt, shape_pad)
                dense_depth_img_color = F.pad(dense_depth_img_color, shape_pad)
                
                # # corr dataset generation 
                # corrs = corr_gen(gt_points_index, points_index , gt_uv, uv , args.num_kp)
                # corrs = corrs
                
                # batch stack 
                rgb_input.append(rgb)
                lidar_input.append(dense_depth_img_color)
                lidar_gt.append(depth_gt)
                real_shape_input.append(real_shape)
                shape_pad_input.append(shape_pad)
                pc_rotated_input.append(pc_rotated)
                # corrs_input.append(corrs)
            
            lidar_input = torch.stack(lidar_input) # lidar 2d depth map input [256,512,1]
            rgb_input = torch.stack(rgb_input) # camera input = [256,512,3]
            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input = F.interpolate(rgb_input, size=[192, 640], mode="bilinear").cuda()
            lidar_input = F.interpolate(lidar_input, size=[192, 640], mode="bilinear").cuda()
            # lidar_input = torch.cat((lidar_input,lidar_input,lidar_input), 1)
            # corrs_input = torch.stack(corrs_input).cuda()
            
            transl_err , rot_err = model(rgb_input, lidar_input)
            loss = loss_fn(sample['point_cloud'], sample['tr_error'], sample['rot_error'], transl_err, rot_err)

            if args.gradient_accumulation_steps > 1:
                loss['total_loss'] = loss['total_loss'] / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss['total_loss'], optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss['total_loss'] .backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss['total_loss'].item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step ,loss_fn , n_data_test)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    # args = parser.parse_args()
    
    import easydict
    
    args = easydict.EasyDict({
        "local_rank": -1,
        "fp16" : 'O2' ,
        "seed" : 42 ,
        "model_type" :  "ViT-B_16" ,
        "dataset" : "cifar10" ,
        "img_size" : 224 ,
        "pretrained_dir" : "checkpoint/ViT-B_16.npz" ,
        "output_dir" : "output" ,
        "name" : "LCCNet_ViT" ,
        "train_batch_size" : 1 ,
        "eval_batch_size" : 1 ,
        "gradient_accumulation_steps" : 1 ,
        "learning_rate" : 3e-2 ,
        "weight_decay" : 0 ,
        "num_steps" : 10000 ,
        "decay_type" : "cosine" ,
        "warmup_steps" : 500 ,
        "eval_every" : 100 ,
        "max_grad_norm" : 1.0 ,
        "fp16_opt_level" : 'O2' ,
        "loss_scale" : 0 ,
        "data_folder" : "/home/ubuntu/data/kitti_odometry" ,
        "max_r" : 20.0 ,
        "max_t" : 1.5 ,
        "use_reflectance" : False ,
        "val_sequence" : '06' ,
        "num_worker" : 5 ,
        "rescale_transl" : 2 ,
        "rescale_rot" : 1 ,
        "weight_point_cloud" : 0.4 ,
        "max_depth" : 80 ,
        "img_shape" : (384, 1280) ,
        "num_kp" : 500
    })

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model , loss_fn = setup(args)

    # Training
    train(args, model , loss_fn)


if __name__ == "__main__":
    main()
