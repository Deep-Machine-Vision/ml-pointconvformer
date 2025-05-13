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

from knn_post_dataloader_utils import compute_knn
from layers import PointConv



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
        xyz.append(torch.randn(num_points, 3).to(cfg.device))
        feats.append(torch.randn(num_points, cfg.NUM_FEATURES).to(cfg.device))

    # compute knn for each point cloud
    nei_inds = [compute_knn(xyz[i], xyz[i], cfg.K) for i in range(cfg.NUM_POINT_CLOUDS)]

    # offset for each point cloud for packed representation
    offset = [len(xyz[i]) for i in range(cfg.NUM_POINT_CLOUDS)]
    # cumsum of offsets
    offset = [0] + list(np.cumsum(offset))

    # add offset to nei_inds
    for i in range(cfg.NUM_POINT_CLOUDS):
        nei_inds[i] += offset[i]

    # convert xyz and feat to packed representation
    xyz = torch.cat(xyz, dim=0).unsqueeze(0)
    feats = torch.cat(feats, dim=0).unsqueeze(0)
    nei_inds = torch.cat(nei_inds, dim=0).unsqueeze(0)
    
    return xyz, feats, nei_inds, offset

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
    pconv = PointConv(
        in_channel=cfg.NUM_FEATURES,
        out_channel=cfg.OUT_CHANNEL,
        cfg=cfg,
        weightnet=[cfg.NUM_FEATURES, 16],
        USE_VI=None
    ).to(cfg.device)

    xyz, feats, nei_inds, offset = prepare_inputs(cfg)
    inv_neighbors_list, inv_k_list, inv_idx_list = compute_knn_inverse_packed(xyz, nei_inds, offset, cfg)

    output, _ = pconv(dense_xyz=xyz,
                      dense_feats=feats,
                      nei_inds=nei_inds,
                      inv_neighbors=inv_neighbors_list,
                      inv_k=inv_k_list,
                      inv_idx=inv_idx_list)
    
    # dummy loss to perform `.backward()`
    target = torch.randn(output.shape).to(cfg.device)
    print(f"output shape: {output.shape}, target shape: {target.shape}")
    loss = torch.nn.MSELoss()(output, target)
    loss.backward()

if __name__ == "__main__":
    # Load config from yaml file
    with open('test_configs/pointconv_packed.yaml', 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))

    test_pconv(cfg)


