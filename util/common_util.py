## Partly adapted from StratifiedTransformer
## https://github.com/dvlab-research/Stratified-Transformer
## Copyright @ 2022 DV Lab
## MIT License, https://github.com/dvlab-research/Stratified-Transformer/blob/main/LICENSE.md

# For to_device() and replace_batchnorm():
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
## 

import os
import numpy as np
from PIL import Image
import random

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.init as initer
import torch.nn.functional as F


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


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    # output = output.reshape(output.size()).copy()
    # target = target.reshape(target.size())
    output = output.flatten()
    target = target.flatten()
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (_ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, _BatchNorm):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def convert_to_syncbn(model):
    def recursive_set(cur_module, name, module):
        if len(name.split('.')) > 1:
            recursive_set(getattr(cur_module, name[:name.find('.')]), name[name.find('.')+1:], module)
        else:
            setattr(cur_module, name, module)
    from lib.sync_bn import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            recursive_set(model, name, SynchronizedBatchNorm1d(m.num_features, m.eps, m.momentum, m.affine))
        elif isinstance(m, nn.BatchNorm2d):
            recursive_set(model, name, SynchronizedBatchNorm2d(m.num_features, m.eps, m.momentum, m.affine))
        elif isinstance(m, nn.BatchNorm3d):
            recursive_set(model, name, SynchronizedBatchNorm3d(m.num_features, m.eps, m.momentum, m.affine))


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def memory_use():
    BYTES_IN_GB = 1024 ** 3
    return 'ALLOCATED: {:>6.3f} ({:>6.3f})  CACHED: {:>6.3f} ({:>6.3f})'.format(
        torch.cuda.memory_allocated() / BYTES_IN_GB,
        torch.cuda.max_memory_allocated() / BYTES_IN_GB,
        torch.cuda.memory_reserved() / BYTES_IN_GB,
        torch.cuda.max_memory_reserved() / BYTES_IN_GB,
    )


def to_device(input_data, device='cuda', non_blocking=False):
    '''
    Move input_data to device. If the data is a list, move everything in the list to device
    '''
    if isinstance(input_data, list) and len(input_data) > 0:
        if isinstance(input_data[0], list):
            for idx in range(len(input_data)):
                for idx2 in range(len(input_data[idx])):
                    input_data[idx][idx2] = input_data[idx][idx2].to(device, non_blocking=non_blocking)
        else:
            for idx in range(len(input_data)):
                input_data[idx] = input_data[idx].to(device, non_blocking=non_blocking)

    if isinstance(input_data, torch.Tensor):
        input_data = input_data.to(device, non_blocking=non_blocking)

    return input_data


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
    else:
        torch.backends.cudnn.deterministic = False 
        torch.backends.cudnn.benchmark = True 


def smooth_loss(output, target, eps=0.1):
    w = torch.zeros_like(output).scatter(1, target.unsqueeze(1), 1)
    w = w * (1 - eps) + (1 - w) * eps / (output.shape[1] - 1)
    log_prob = F.log_softmax(output, dim=1)
    loss = (-w * log_prob).sum(dim=1).mean()
    return loss


# Inspired by the LeViT repository
# https://github.com/facebookresearch/LeViT
def replace_batchnorm(net):
    '''
    Fuse the batch normalization during inference time
    '''
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


def compute_knn_inverse(pointclouds, edges_self, edges_forward, edges_propagate):
    """
    Compute inverse k-nearest neighbor mappings for self, forward, and propagate edges using pcf_cuda

    Parameters
    ----------
    pointclouds : list of Tensor
        List of point cloud tensors at each level. Each tensor has shape [B, N, ...],
        where N is the number of points.
    edges_self : list of Tensor
        Each element is a [B, k] tensor of neighbor indices for self edges at each level.
    edges_forward : list of Tensor
        Each element is a [B, k] tensor of neighbor indices mapping from current to next level.
    edges_propagate : list of Tensor
        Each element is a [B, k] tensor of neighbor indices for propagation edges at each level.

    Returns
    -------
    inv_self : list
        [inverse_neighbors_self, inverse_k_self, inverse_idx_self], each a list of Tensors
        mapping back from neighbor to point indices for self edges.
    inv_forward : list
        [inverse_neighbors_forward, inverse_k_forward, inverse_idx_forward] for forward edges.
    inv_propagate : list
        [inverse_neighbors_propagate, inverse_k_propagate, inverse_idx_propagate] for propagate edges.
    """
    import pcf_cuda

    inverse_neighbors_self = []
    inverse_k_self = []
    inverse_idx_self = []
    for j, edges in enumerate(edges_self):
        total_points = pointclouds[j].shape[1]  # Current level point count
        inv_n, inv_k, inv_idx = pcf_cuda.compute_knn_inverse(edges, total_points)
        inverse_neighbors_self.append(inv_n)
        inverse_k_self.append(inv_k) 
        inverse_idx_self.append(inv_idx)

    inverse_neighbors_forward = []
    inverse_k_forward = []
    inverse_idx_forward = []
    for j, edges in enumerate(edges_forward):
        # For forward edges, neighbor indices refer to points in the CURRENT level (dense)
        # So total_points should be the number of points in the current level, not the next level
        total_points = pointclouds[j].shape[1]  # CURRENT level (dense) point count
        inv_n, inv_k, inv_idx = pcf_cuda.compute_knn_inverse(edges, total_points)
        inverse_neighbors_forward.append(inv_n)
        inverse_k_forward.append(inv_k)
        inverse_idx_forward.append(inv_idx)

    inverse_neighbors_propagate = []
    inverse_k_propagate = []
    inverse_idx_propagate = []
    for j, edges in enumerate(edges_propagate):
        # For propagate edges, total points is number of points in the current level
        total_points = pointclouds[j].shape[1]  # Current level point count
        inv_n, inv_k, inv_idx = pcf_cuda.compute_knn_inverse(edges, total_points)
        inverse_neighbors_propagate.append(inv_n)
        inverse_k_propagate.append(inv_k)
        inverse_idx_propagate.append(inv_idx)

    inverse_neighbors_self = to_device(inverse_neighbors_self, non_blocking=True)
    inverse_k_self = to_device(inverse_k_self, non_blocking=True)
    inverse_idx_self = to_device(inverse_idx_self, non_blocking=True)

    inverse_neighbors_forward = to_device(inverse_neighbors_forward, non_blocking=True)
    inverse_k_forward = to_device(inverse_k_forward, non_blocking=True)
    inverse_idx_forward = to_device(inverse_idx_forward, non_blocking=True)

    inverse_neighbors_propagate = to_device(inverse_neighbors_propagate, non_blocking=True)
    inverse_k_propagate = to_device(inverse_k_propagate, non_blocking=True)
    inverse_idx_propagate = to_device(inverse_idx_propagate, non_blocking=True)

    inv_self = [inverse_neighbors_self, inverse_k_self, inverse_idx_self]
    inv_forward = [inverse_neighbors_forward, inverse_k_forward, inverse_idx_forward]
    inv_propagate = [inverse_neighbors_propagate, inverse_k_propagate, inverse_idx_propagate]

    return inv_self, inv_forward, inv_propagate