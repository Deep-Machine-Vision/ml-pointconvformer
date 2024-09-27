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

def createInverse(neighborMat, inp_points):
  N =  inp_points # neighborMat.shape[0]
  K = neighborMat.shape[1]
  neigh = [ [] for n in range(N)]
  inv_k = [ [] for n in range(N)]

  for r in range(neighborMat.shape[0]):
    for c in range(K):
      neigh[neighborMat[r,c]].append(r)
      inv_k[neighborMat[r,c]].append(c)
  
  
  idx = [len(x) for x in neigh]
  cum_sum = 0
  for i in range(N):
    if idx[i] == 0:
      idx[i] = -1
    else:
      temp = idx[i]
      idx[i] = cum_sum 
      cum_sum = temp+cum_sum

  idx.append(neighborMat.shape[0]*K)
  neighbors = np.hstack(neigh).flatten()
  inv_k = np.hstack(inv_k).flatten()
  return neighbors, np.array(inv_k), np.array(idx)

def round_matrix(neighbor_inds, inp_points):
    # create round matrix
    ineigh = []
    ik = []
    iidx = []
    for b in range(B):
        inv_neighbors, inv_k, inv_idx = createInverse(neighbor_inds[b], inp_points) 
        ineigh.append(torch.from_numpy(inv_neighbors))
        ik.append(torch.from_numpy(inv_k))
        iidx.append(torch.from_numpy(inv_idx))

    inv_neighbors = torch.stack(ineigh, dim=0).cuda()
    inv_k = torch.stack(ik, dim=0).cuda()
    inv_idx = torch.stack(iidx, dim=0).cuda()

    return inv_neighbors, inv_k, inv_idx


# Config
B = 32   # batch size
N = 1024   # number of points 
C_in = 3    # number of input channels
C_out = 64 # number of output channels
C_final = 256
K=16


WARMUP_STEPS = 100
ITERS = 200
USE_FPCONV = True

fpconv1 = FasterPointConv(in_channel=C_in, out_channel=C_out, weightnet=[3, 16]).cuda()
fpconv2 = FasterPointConv(in_channel=C_out, out_channel=C_out*2, weightnet=[3, 16]).cuda()
fpconv3 = FasterPointConv(in_channel=C_out*2, out_channel=C_final, weightnet=[3, 16]).cuda()


pconv1 = PointConv(in_channel=C_in, out_channel=C_out, weightnet=[3, 16]).cuda()
pconv2 = PointConv(in_channel=C_out, out_channel=C_out*2, weightnet=[3, 16]).cuda()
pconv3 = PointConv(in_channel=C_out*2, out_channel=C_final, weightnet=[3, 16]).cuda()


# count number of trainable parameters: 
assert sum(p.numel() for p in fpconv1.parameters() if p.requires_grad) == sum(p.numel() for p in pconv1.parameters() if p.requires_grad)

# copy weights
fpconv1.load_state_dict(pconv1.state_dict())
fpconv2.load_state_dict(pconv2.state_dict())
fpconv3.load_state_dict(pconv3.state_dict())

inp = torch.randn(B, N, 3).clone().detach().requires_grad_(False).cuda()
inp_feat = torch.randn(B, N, C_in).clone().detach().requires_grad_(True).cuda()

inp1= torch.randn(B, N, 3).clone().detach().requires_grad_(False).cuda()
inp2= torch.randn(B, N, 3).clone().detach().requires_grad_(False).cuda()
inp3= torch.randn(B, N, 3).clone().detach().requires_grad_(False).cuda()

gt_feat = torch.randn(B, N, C_final).clone().detach().requires_grad_(False).cuda() 

# get neighbors
_, neighbor_inds = knn(inp1, inp, nsample=K) # b n k
_, neighbor_inds1 = knn(inp2, inp1, nsample=K) # b n k
_, neighbor_inds2 = knn(inp3, inp2, nsample=K) # b n k

# _, neighbor_inds = knn(inp, inp, nsample=K) # b n k


#if USE_FPCONV:
inv_neighbors, inv_k, inv_idx = round_matrix(neighbor_inds, inp.shape[1])
neighbor_inds = torch.tensor(neighbor_inds).to(inp_feat.device)

inv_neighbors1, inv_k1, inv_idx1 = round_matrix(neighbor_inds1, inp1.shape[1])
neighbor_inds1 = torch.tensor(neighbor_inds1).to(inp_feat.device)

inv_neighbors2, inv_k2, inv_idx2 = round_matrix(neighbor_inds2, inp2.shape[1])
neighbor_inds2 = torch.tensor(neighbor_inds2).to(inp_feat.device)

pconv_time = 0
fpconv_time = 0
idx = 0

for i in range(WARMUP_STEPS + ITERS):
    
    if i > WARMUP_STEPS:
        idx += 1
        torch.cuda.synchronize()
        start_time = time.time()
        output = pconv1(inp, inp_feat, neighbor_inds, sparse_xyz=inp1)[0]
        output = pconv2(inp1, output, neighbor_inds1, sparse_xyz=inp2)[0]
        output = pconv3(inp2, output, neighbor_inds2, sparse_xyz=inp3)[0]
        # loss = F.mse_loss(output, gt_feat)
        loss = -output.mean()
        loss.backward()
        pconv_time += time.time() - start_time
        print(f"Loss PConv: {loss.item()} || Grad: {pconv3.linear.c.weight.grad.mean().item()}") # || Grad: {pconv2.linear.c.weight.grad.mean().item()}
        pconv1.zero_grad()
        pconv2.zero_grad()
        pconv3.zero_grad()

print(f"----------------------")

for i in range(WARMUP_STEPS + ITERS):
    if i > WARMUP_STEPS:
        torch.cuda.synchronize()
        start_time = time.time()
        output = fpconv1(inp, inp_feat, inv_neighbors.long(), inv_k.long(), inv_idx, neighbor_inds, sparse_xyz=inp1)[0]
        output = fpconv2(inp1, output, inv_neighbors1.long(), inv_k1.long(), inv_idx1, neighbor_inds1, sparse_xyz=inp2)[0]
        output = fpconv3(inp2, output, inv_neighbors2.long(), inv_k2.long(), inv_idx2, neighbor_inds2, sparse_xyz=inp3)[0]
        # loss = F.mse_loss(output, gt_feat)
        loss = -output.mean()
        loss.backward()
        fpconv_time += time.time() - start_time
        print(f"Loss Faster-PConv: {loss.item()} || Grad: {fpconv3.linear.c.weight.grad.mean().item()} ") # || Grad: {fpconv2.linear.c.weight.grad.mean().item()}
        
        fpconv1.zero_grad()
        fpconv2.zero_grad()
        fpconv3.zero_grad()
        
print(f"PConv time: {pconv_time / idx} :::: Faster-PConv time: {fpconv_time / idx} :::: Speed up : {pconv_time / fpconv_time}")



"""

mlp1 = nn.Sequential(
   nn.Linear(10, 256),
   nn.ReLU(),
   nn.Linear(256, 256),
   nn.Sigmoid()
).cuda()

inp = torch.tensor(torch.randn(B, 10).clone().detach(), device="cuda", requires_grad=False)
gt = torch.tensor(torch.randn(B, 256).clone().detach(), device="cuda", requires_grad=False)

for i in range(50):
   mlp1.zero_grad()
   output = mlp1(inp)
   loss = F.mse_loss(output, gt)
   loss.backward()
   print(f"Loss: {loss.item()} || Grad Linear 1: {mlp1[0].weight.grad.mean().item()} || Grad Linear 2: {mlp1[2].weight.grad.mean().item()}")
   del loss

"""