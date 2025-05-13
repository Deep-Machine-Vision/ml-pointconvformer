import torch
import torch.nn as nn
import pcf_cuda

import yaml
from easydict import EasyDict
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knn_post_dataloader_utils import compute_knn
from layers import PointConv


def test_pconv(cfg):
    pconv = PointConv(
        in_channel=cfg.NUM_FEATURES,
        out_channel=cfg.OUT_CHANNEL,
        cfg=cfg,
        weightnet=[cfg.NUM_FEATURES, 16],
        USE_VI=None
    ).to(cfg.device)

    xyz = torch.randn(cfg.NUM_POINTS, 3).cuda()
    feats = torch.randn(cfg.NUM_POINTS, cfg.NUM_FEATURES).cuda()
    nei_inds = compute_knn(xyz, xyz, cfg.K)

    # Add batch dimension
    xyz = xyz.unsqueeze(0)
    feats = feats.unsqueeze(0)
    nei_inds = nei_inds.unsqueeze(0)
    
    inv_neighbors, inv_k, inv_idx = pcf_cuda.compute_knn_inverse(nei_inds, cfg.NUM_POINTS)
    
    output, _ = pconv(dense_xyz=xyz,
                      dense_feats=feats,
                      nei_inds=nei_inds,
                      inv_neighbors=inv_neighbors,
                      inv_k=inv_k,
                      inv_idx=inv_idx)
    
    # dummy loss to perform `.backward()`
    target = torch.randn(1, cfg.NUM_POINTS, cfg.OUT_CHANNEL).cuda()
    print(f"output shape: {output.shape}, target shape: {target.shape}")
    loss = torch.nn.MSELoss()(output, target)
    loss.backward()

if __name__ == "__main__":
    # Load config from yaml file
    with open('test_configs/pointconv_single.yaml', 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))

    test_pconv(cfg)


