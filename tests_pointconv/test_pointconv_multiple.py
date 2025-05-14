import numpy as np
import torch
import torch.nn as nn
import pcf_cuda
import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from easydict import EasyDict

from knn_post_dataloader_utils import compute_knn_packed, prepare
from layers import PointConv
from util.common_util import compute_knn_inverse
# voxelize point clouds
from util.voxelize import voxelize



def prepare_inputs(cfg):
    """
    Generate random point clouds with different number of points 
    and compute knn for each point cloud for `forward()`.
    """
    xyz = []
    feats = []
    # Generate random point clouds with different number of points
    # choosing from [MIN_NUM_POINTS, MAX_NUM_POINTS]
    for _ in range(cfg.NUM_POINT_CLOUDS):
        num_points = np.random.randint(cfg.MIN_NUM_POINTS, cfg.MAX_NUM_POINTS)
        xyz.append(np.random.randn(num_points, 3))
        feats.append(np.random.randn(num_points, cfg.NUM_FEATURES))

    xyz_downsampled = [[] for _ in range(cfg.NUM_DOWNSAMPLING_LAYERS)]
    for i in range(cfg.NUM_DOWNSAMPLING_LAYERS):
        for j in range(cfg.NUM_POINT_CLOUDS):
            voxel_idx = voxelize(xyz[j], voxel_size=cfg.VOXEL_SIZE*(cfg.DOWNSAMPLE_FACTOR**i), hash_type='fnv', mode='random')
            xyz_downsampled[i].append(xyz[j][voxel_idx])

    # merge xyz and xyz_downsampled
    xyz = [xyz] + xyz_downsampled

    # offset for each point cloud for packed representation
    offset = []
    for i in range(cfg.NUM_DOWNSAMPLING_LAYERS + 1):
        level_offset = [len(xyz[i][j]) for j in range(cfg.NUM_POINT_CLOUDS)]
        offset.append(level_offset)
    
    # unsqueeze xyz
    unsqueezed_xyz = []
    for level in xyz:
        unsqueezed_points = [torch.from_numpy(points).to(torch.float32).unsqueeze(0).to(cfg.device) for points in level]
        unsqueezed_xyz.append(torch.cat(unsqueezed_points, dim=1))
    xyz = unsqueezed_xyz

    # feats 
    feats = np.concatenate(feats, axis=0)
    feats = torch.from_numpy(feats).to(torch.float32).unsqueeze(0).to(cfg.device)

    # Compute kNN
    nei_self_list, nei_forward_list, nei_propagate_list = compute_knn_packed(xyz, offset, cfg.K_self, cfg.K_forward, cfg.K_propagate)
    nei_self_list, nei_forward_list, nei_propagate_list = prepare(nei_self_list, nei_forward_list, nei_propagate_list)

    return xyz, feats, nei_self_list, nei_forward_list, nei_propagate_list

def compute_knn_inverse_packed(xyz, nei_inds, offset, cfg):
    """
    Compute knn inverse for each point cloud for packed representation.
    """
    inv_neighbors_list = []
    inv_k_list = []
    inv_idx_list = []
    for i in range(cfg.NUM_POINT_CLOUDS):
        num_points = xyz[:, offset[i]:offset[i+1], :].shape[1]
        inv_neighbors, inv_k, inv_idx = pcf_cuda.compute_knn_inverse(nei_inds[:, offset[i]:offset[i+1]]-offset[i], num_points)
        inv_neighbors_list.append(inv_neighbors+offset[i])
        inv_k_list.append(inv_k+offset[i])
        inv_idx_list.append(inv_idx+offset[i])
    inv_neighbors_list = torch.cat(inv_neighbors_list, dim=1)
    inv_k_list = torch.cat(inv_k_list, dim=1)
    inv_idx_list = torch.cat(inv_idx_list, dim=1)
    return inv_neighbors_list, inv_k_list, inv_idx_list

def test_pconv(cfg):
    pconv_1 = PointConv(
        in_channel=cfg.NUM_FEATURES,
        out_channel=cfg.OUT_CHANNEL,
        cfg=cfg,
        weightnet=[cfg.NUM_FEATURES, 16],
        USE_VI=None
    ).to(cfg.device)

    pconv_2 = PointConv(
        in_channel=cfg.OUT_CHANNEL,
        out_channel=cfg.OUT_CHANNEL*4,
        cfg=cfg,
        weightnet=[cfg.NUM_FEATURES, 16],
        USE_VI=None
    ).to(cfg.device)

    pconv_3 = PointConv(
        in_channel=cfg.OUT_CHANNEL*4,
        out_channel=cfg.OUT_CHANNEL*16,
        cfg=cfg,
        weightnet=[cfg.NUM_FEATURES, 16],
        USE_VI=None
    ).to(cfg.device)

    xyz, feats, nei_self_list, nei_forward_list, nei_propagate_list = prepare_inputs(cfg)
    inv_neighbors_list, inv_k_list, inv_idx_list = compute_knn_inverse(xyz, nei_self_list, nei_forward_list, nei_propagate_list)
    print(f"Original feats shape: {feats.shape}")
    # First Layer: Applying PointConv from original resolution
    # point cloud to first downsampled point cloud.
    feats, _ = pconv_1(dense_xyz=xyz[0],
                      dense_feats=feats,
                      nei_inds=nei_self_list[1],
                      sparse_xyz=xyz[1],
                      inv_neighbors=inv_neighbors_list[0],
                      inv_k=inv_k_list[0],
                      inv_idx=inv_idx_list[0])
    print(f"First downsampled feats shape: {feats.shape}")

    # Second Layer: Applying PointConv from first downsampled point cloud
    # to second downsampled point cloud.
    feats, _ = pconv_2(dense_xyz=xyz[1],
                      dense_feats=feats,
                      nei_inds=nei_self_list[2],
                      sparse_xyz=xyz[2],
                      inv_neighbors=inv_neighbors_list[1],
                      inv_k=inv_k_list[1],
                      inv_idx=inv_idx_list[1])
    print(f"Second downsampled feats shape: {feats.shape}")

    # Third Layer: Applying PointConv from second downsampled point cloud
    # to third downsampled point cloud.
    feats, _ = pconv_3(dense_xyz=xyz[2],
                      dense_feats=feats,
                      nei_inds=nei_self_list[3],
                      sparse_xyz=xyz[3],
                      inv_neighbors=inv_neighbors_list[2],
                      inv_k=inv_k_list[2],
                      inv_idx=inv_idx_list[2])  

    print(f"Third downsampled feats shape: {feats.shape}")

    # dummy loss to perform `.backward()`
    target = torch.randn(feats.shape).to(cfg.device)
    print(f"output shape: {feats.shape}, target shape: {target.shape}")
    loss = torch.nn.MSELoss()(feats, target)
    loss.backward()

if __name__ == "__main__":
    # Load config from yaml file
    with open('test_configs/pointconv_packed.yaml', 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))

    test_pconv(cfg)


