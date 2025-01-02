# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.

# Usage: python enumerate_data_loader.py --config config_file
#        example config files can be found in ./configs/

import os
import time
import datetime
import argparse
import shutil
import numpy as np
import yaml
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributed as dist
from tensorboardX import SummaryWriter

from util.common_util import to_device, init_seeds
from model_architecture import PointConvFormer_Segmentation as VI_PointConv
from model_architecture import get_default_configs
import scannet_data_loader_color_DDP as scannet_data_loader
# from datasetCommon import compute_knn_after_tensorize
from sklearn.neighbors import KDTree
from pykeops.torch import LazyTensor
# from Serialize.serialize_knn import find_knn_zorder
from concurrent.futures import ThreadPoolExecutor
from knn_post_dataloader_utils import prepare, compute_knn_packed


def get_default_training_cfgs(cfg):
    '''
    Get default configurations w.r.t. the training and the dataset, note that this doesn't set the model default
    configurations that is in model_architecture.get_default_configs()
    '''
    if 'label_smoothing' not in cfg.keys():
        cfg.label_smoothing = False
    if 'accum_iter' not in cfg.keys():
        cfg.accum_iter = 1
    if 'rotate_aug' not in cfg.keys():
        cfg.rotate_aug = True
    if 'flip_aug' not in cfg.keys():
        cfg.flip_aug = False
    if 'scale_aug' not in cfg.keys():
        cfg.scale_aug = True
    if 'transform_aug' not in cfg.keys():
        cfg.transform_aug = False
    if 'color_aug' not in cfg.keys():
        cfg.color_aug = True
    if 'crop' not in cfg.keys():
        cfg.crop = False
    if 'shuffle_index' not in cfg.keys():
        cfg.shuffle_index = True
    if 'mix3D' not in cfg.keys():
        cfg.mix3D = False
    if 'DDP' not in cfg.keys():
        cfg.DDP = False
    return cfg

def get_parser():
    '''
    Get the arguments
    '''
    parser = argparse.ArgumentParser('ScanNet PointConvFormer')
    parser.add_argument(
        '--local_rank',
        default=-1,
        type=int,
        help='local_rank')
    parser.add_argument(
        '--config',
        default='./configWenxuanPCFDDPL5WarmUP.yaml',
        type=str,
        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = edict(yaml.safe_load(open(args.config, 'r')))
    cfg = get_default_configs(cfg, cfg.num_level, cfg.base_dim)
    cfg = get_default_training_cfgs(cfg)
    cfg.local_rank = args.local_rank
    cfg.config = args.config
    return cfg

def main():
    '''
    Main entry point for the script
    '''
    args = get_parser()
    file_dir = os.path.join(args.experiment_dir, '%s_SemSeg-' %
                            (args.model_name) +
                            str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    args.file_dir = file_dir

    train_data_loader, val_data_loader = scannet_data_loader.getdataLoadersDDP(args)

    # model = VI_PointConv(args).to(args.local_rank)
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")

    # Enumerate data from train_data_loader
    timing_knn = []
    itr = 10
    for t in range(itr):
        start_time = time.time()
        for i, data in enumerate(train_data_loader):
            
            features, pointclouds, target, norms, points_stored = data

            features, pointclouds, target, norms = to_device(features, non_blocking=True), to_device(
                pointclouds, non_blocking=True), to_device(
                    target, non_blocking=True), to_device(norms, non_blocking=True)
            
            # edges_self, edges_forward, edges_propagate = compute_knn_packed(pointclouds, points_stored)
            edges_self, edges_forward, edges_propagate = compute_knn_packed(pointclouds, points_stored)
            edges_self, edges_forward, edges_propagate = prepare(edges_self, edges_forward, edges_propagate)
            
        timing_knn.append(time.time()-start_time)
        print("done",t, " ", i)
        # Example of how you might process each batch of data
    
    print("Average time for keops", sum(timing_knn[1:]) / (itr-1))
       
        
    
    

if __name__ == '__main__':
    main()
