import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pykeops.torch import LazyTensor
from layers import FasterPointConv, PointConv


def knn(pts1, pts2, nsample, sorted=False):
    """
    Input:
        pts1: query point set, [B, S, C]
        pts2: other point set, [B, N, C]
        nsample: number of nearest neighbors to sample
        sorted: whether to sort by nearest to farthest
    Return:
        nn_dists: nearest neigbor distances, [B, S, nsample]
        nn_idx: grouped points index, [B, S, nsample]
    """
    B, S, C = pts1.shape
    _, N, _ = pts2.shape

    x_i = LazyTensor(pts1.view(B, S, 1, C))
    y_j = LazyTensor(pts2.view(B, 1, N, C))

    D_ij = ((x_i - y_j)**2).sum(-1)**0.5
    #distances_i = D_ij.Kmin(nsample, dim=2)
    distances_i, indices_i = D_ij.Kmin_argKmin(nsample, dim=2)

    return distances_i, indices_i.long()

def createInverse(neighborMat):
  N = 1024+1 # neighborMat.shape[0]
  K = neighborMat.shape[1]
  neigh = [ [] for n in range(N)]
  inv_k = [ [] for n in range(N)]

  for r in range(neighborMat.shape[0]):
    for c in range(K):
      neigh[neighborMat[r,c]].append(r)
      inv_k[neighborMat[r,c]].append(c)
  
  
  idx = [len(x) for x in neigh]
  print(idx, "-------")
  cum_sum = 0
  for i in range(N):
    if idx[i] == 0:
      idx[i] = -1
    else:
      temp = idx[i]
      idx[i] = cum_sum 
      cum_sum = temp+cum_sum

  idx.append(N*K)
  neighbors = np.hstack(neigh).flatten()
  inv_k = np.hstack(inv_k).flatten()
  return neighbors, np.array(inv_k), np.array(idx)

def round_matrix(neighbor_inds):
    # create round matrix
    ineigh = []
    ik = []
    iidx = []
    for b in range(B):
        inv_neighbors, inv_k, inv_idx = createInverse(neighbor_inds[b])
        ineigh.append(torch.from_numpy(inv_neighbors))
        ik.append(torch.from_numpy(inv_k))
        iidx.append(torch.from_numpy(inv_idx))

    inv_neighbors = torch.stack(ineigh, dim=0).cuda()
    inv_k = torch.stack(ik, dim=0).cuda()
    inv_idx = torch.stack(iidx, dim=0).cuda()

    return inv_neighbors, inv_k, inv_idx


# Config
B = 16   # batch size
N = 1024   # number of points 
C_in = 3    # number of input channels
C_out = 512 # number of output channels
K=16

WARMUP_STEPS = 2
ITERS = 10
USE_FPCONV = True

fpconv1 = FasterPointConv(in_channel=C_in, out_channel=C_out, weightnet=[3, 16]).cuda()

pconv1 = PointConv(in_channel=C_in, out_channel=C_out, weightnet=[3, 16]).cuda()

# count number of trainable parameters: 
assert sum(p.numel() for p in fpconv1.parameters() if p.requires_grad) == sum(p.numel() for p in pconv1.parameters() if p.requires_grad)

# copy weights
fpconv1.load_state_dict(pconv1.state_dict())

input_points = torch.tensor(torch.randn(B, N, 3).clone().detach(), device="cuda", requires_grad=False)
input_points_ds = torch.tensor(torch.randn(B, N//2, 3).clone().detach(), device="cuda", requires_grad=False)
input_feat = torch.tensor(torch.randn(B, N, C_in).clone().detach(), device="cuda", requires_grad=False)
gt_feat = torch.tensor(torch.randn(B, N//2, C_out).clone().detach(), device="cuda", requires_grad=False)

# get neighbors
_, neighbor_inds = knn(input_points_ds, input_points, nsample=K) # b n k
print(neighbor_inds)

#if USE_FPCONV:
inv_neighbors, inv_k, inv_idx = round_matrix(neighbor_inds)
neighbor_inds = torch.tensor(neighbor_inds).to(input_feat.device)

print(f"Inv Neighbors \n: {inv_neighbors}")
print(f"Inv K \n: {inv_k}")
print(f"Inv Idx \n: {inv_idx}")

total_time = 0

for i in range(WARMUP_STEPS + ITERS):
    if i > WARMUP_STEPS:
        # torch.cuda.synchronize()
        start = time.time()
        #if USE_FPCONV:
        output = fpconv1(input_points, input_feat, inv_neighbors.long(), inv_k.long(), inv_idx, neighbor_inds, sparse_xyz=input_points_ds)
        #else:
            # output = pconv1(input_points, input_feat, neighbor_inds)
        output2 = pconv1(input_points, input_feat, neighbor_inds, sparse_xyz=input_points_ds)
        # import pdb; pdb.set_trace()
        loss = F.mse_loss(output[0], gt_feat)
        loss2 = F.mse_loss(output2[0], gt_feat)
        loss.backward()
        loss2.backward()
        print(loss.item(), loss2.item(), fpconv1.linear.c.weight.grad.mean().item(), pconv1.linear.c.weight.grad.mean().item())
        pconv1.zero_grad()
        fpconv1.zero_grad()
        # breakpoint()
        
        
        total_time += (time.time() - start)

print("Total time: {:.3f} ms".format(total_time/ITERS*1000))