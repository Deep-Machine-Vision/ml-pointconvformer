import time
from einops import rearrange, repeat
from termcolor import cprint
import torch
import pcf_cuda
from pykeops.torch import LazyTensor
import numpy as np


def createInverse(neighborMat):
  N = neighborMat.shape[0]
  K = neighborMat.shape[1]
  neigh = [ [] for n in range(N)]
  inv_k = [ [] for n in range(N)]

  for r in range(N):
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

  idx.append(N*K)

  neighbors = np.hstack(neigh).flatten()
  inv_k = np.hstack(inv_k).flatten()
  
  return neighbors, np.array(inv_k), np.array(idx)

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

def index_points(points, idx):
    """

    Input:
        points: input points data, shape [B, N, C]
        idx: sample index data, shape [B, S] / [B, S, K]
    Return:
        new_points:, indexed points data, shape [B, S, C] / [B, S, K, C]
    """
    device = points.device
    BB = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(BB, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def construct_unique_neighors(b, n, k):
    from scipy.linalg import circulant
    neighbors = circulant(torch.arange(n).numpy()) # n x n
    neighbors = neighbors[:, :k]
    neighbors = torch.from_numpy(neighbors)
    return repeat(neighbors, 'n k -> b n k', b=b).cuda()
    # offset = 0
    # neighbors = []
    # base = torch.arange(k)
    # for nn in range(n):
    #     neighbors.append(base + offset)
    #     offset += k
    #     if offset > n-1:
    #         offset = 0
    # neighbors = torch.stack(neighbors, dim=0)
    # print(neighbors)
    # print(f"Count percentage: {count_collisions(neighbors) / (n*k):.2f}")

    # return repeat(neighbors, 'n k -> b n k', b=b).cuda()

        
def count_collisions(neighbors):
    # neighors: N x K
    n, k = neighbors.shape
    count = 0
    for kk in range(k):
        seen = [False] * n
        for nn in range(n):
            if not seen[neighbors[nn, kk]]:
                seen[neighbors[nn, kk]] = True
        count += sum(seen)
    return count

def create_no_collision_matrix(neighbor_mat):
    n, k = neighbor_mat.shape
    num_rounds = 2*k
    rounds = -torch.ones((n, num_rounds), dtype=int)
    indices = torch.arange((2*k), dtype=int).to(rounds.device)
    indices = repeat(indices, "d -> n d", n=n)
    indices[:, k:] = -1
    rounds[0:n, 0:k] = neighbor_mat
    for c in range(num_rounds):
        seen = torch.zeros((n), dtype=bool)
        for r in range(n):
            if rounds[r, c] == -1:
                continue
            elif not seen[rounds[r, c]]:
                seen[rounds[r, c]] = True
            else:
                for cc in range(c+1, num_rounds):
                    if rounds[r, cc] == -1:
                        rounds[r, c], rounds[r, cc] = rounds[r, cc].clone(), rounds[r, c].clone()
                        indices[r, c], indices[r, cc] = indices[r, cc].clone(), indices[r, c].clone()
                        break
                    elif not seen[rounds[r, cc]]:
                        seen[rounds[r, cc]] = True
                        rounds[r, c], rounds[r, cc] = rounds[r, cc].clone(), rounds[r, c].clone()
                        indices[r, c], indices[r, cc] = indices[r, cc].clone(), indices[r, c].clone()
                        break
    return rounds, indices

B = 64    # batch size
N = 1024   # number of points 
C_in = 32     # number of input channels
C_mid = 8    # number of intermediate channels
C_out = 64  # number of output channels (should be > 8)
C_add = 3 # number of additional features
K = 16 # number of neighbors

# Time keeping
WARMUP_STEPS = 5
REPEATS = 50
forward_pconv = 0.
backward_pconv = 0.
backward_pconv_opt = 0.
backward_torch = 0.


torch.manual_seed(420)

# initialize inputs
input = torch.tensor(torch.randn(B, N, C_in), device="cuda", requires_grad=True)
input2 = torch.tensor(torch.randn(B, N, C_in), device="cuda", requires_grad=True)
weights = torch.tensor(torch.randn(B, N, K, C_mid), device="cuda", requires_grad=True)

# get neighbors
_, neighbor_inds = knn(input, input, nsample=K) # b n k
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



gathered_feat = index_points(input, neighbor_inds)
#print(gathered_feat.shape)

add_feat = torch.randn([B, N, K, C_add], device="cuda",requires_grad=True)

pytorch_check = True
if pytorch_check:
  #pytorch forward and backward
  gathered_feat2 = torch.cat([gathered_feat, add_feat], dim=-1)
  new_pconv = torch.matmul(input=gathered_feat2.permute(0, 1, 3, 2).contiguous(), other=weights)
  new_pconv = new_pconv.view(new_pconv.shape[0], new_pconv.shape[1], -1)

  grad = torch.randn_like(new_pconv)
  temp = torch.mul(new_pconv, grad)
  fin_func = torch.sum(temp)
  
  start = time.time()
  fin_func.backward()
  torch.cuda.synchronize()
  end = time.time()
  print("Torch backward (x1): {:.3f} ms".format((end-start)*1000))

  grad_input = input.grad.clone().detach()
  grad_add = add_feat.grad.clone().detach()
  grad_weights = weights.grad.clone().detach()
  

  output_pconv = pcf_cuda.pconv_forward(input, neighbor_inds, weights, add_feat)
  torch.cuda.synchronize()

  pconv_backward_opt = pcf_cuda.pconv_backward_opt(grad,
                                                  input, 
                                                  inv_neighbors.long(),
                                                  inv_k.long(),
                                                  inv_idx.long(),
                                                  neighbor_inds.long(),
                                                  weights,
                                                  add_feat)
  torch.cuda.synchronize()

  
  pconv_backward = pcf_cuda.pconv_backward(grad,
                                             input, 
                                             neighbor_inds,
                                             weights,
                                             add_feat)
  torch.cuda.synchronize()

  
  print("======== Torch Comaprisons ==========")
  print("Norm(pconv_cuda.forward, pconv_torch.forward) = ",torch.norm(new_pconv-output_pconv).item())
  print("\n Default")
  print("Norm(pconv_cuda.input_grad, pconv_torch.input_grad) = ",torch.norm(grad_input-pconv_backward[0]).item())
  print("Norm(pconv_cuda.weight_grad, pconv_torch.weight_grad) = ",torch.norm(grad_weights-pconv_backward[1]).item())
  print("Norm(pconv_cuda.add_grad, pconv_torch.add_grad) = ",torch.norm(grad_add-pconv_backward[2]).item())
  print("\n Updated")
  print("Norm(pconv_cuda_opt.input_grad, pconv_torch.input_grad) = ",torch.norm(grad_input-pconv_backward_opt[0]).item())
  print("Norm(pconv_cuda_opt.weight_grad, pconv_torch.weight_grad) = ",torch.norm(grad_weights-pconv_backward_opt[1]).item())
  print("Norm(pconv_cuda_opt.add_grad, pconv_torch.add_grad) = ",torch.norm(grad_add-pconv_backward_opt[2]).item())
  print("==================")

  #b=0
  #i=0
  #for n in range(N):
  #    print((inv_idx[b,n+1]-inv_idx[b,n]).item(),(grad_input[b,n,:]-pconv_backward_opt[0][b,n,:]))

  

  #for b in range(B):
  #  for n in range(N):
  #    print()
  #    print(grad_input[b,n,:])
  #    print(pconv_backward_opt[0][b,n,:], torch.norm(grad_input[b,n,:] - pconv_backward_opt[0][b,n,:]))
  #    print(pconv_backward[0][b,n,:], torch.norm(grad_input[b,n,:] - pconv_backward[0][b,n,:]))

#input = input.clone().detach()
#weights = weights.clone().detach()
#add_feat = add_feat.clone().detach()
finite_diff = False
if finite_diff:
  output_pconv = pcf_cuda.pconv_forward(input, neighbor_inds, weights, add_feat)
  grad = torch.zeros_like(output_pconv)
  
  err_def = 0
  err_opt = 0
  n = 0
  print("h\t\t opt\t\t def")
  h = 0.00001
  for bo in range(B):
    for io in range(N):
      for co in range(C_out):
        grad[bo][io][co] = 1
        pconv_backward = pcf_cuda.pconv_backward(grad,
                                             input, 
                                             neighbor_inds,
                                             weights,
                                             add_feat)
        torch.cuda.synchronize()
        pconv_backward_opt = pcf_cuda.pconv_backward_opt(grad,
                                                  input, 
                                                  inv_neighbors,
                                                  inv_k,
                                                  inv_idx,
                                                  neighbor_inds,
                                                  weights,
                                                  add_feat)
        torch.cuda.synchronize()
        grad[bo][io][co]=0
        for b in range(B):
          for i in range(N):
            for c in range(C_in):
              x = pcf_cuda.pconv_forward(input, neighbor_inds, weights, add_feat)[bo][io][co].item()
              torch.cuda.synchronize()
              input[b][i][c] += h
              xh = pcf_cuda.pconv_forward(input, neighbor_inds, weights, add_feat)[bo][io][co].item()
              torch.cuda.synchronize()
              diff = (xh-x)/h
              input[b][i][c] -= h
              err_def += abs(diff - pconv_backward[0][b][i][c].item())
              err_opt += abs(diff - pconv_backward_opt[0][b][i][c].item())
              n += 1
              print("{:.5f}\t\t{:.5f}\t\t{:.5f}".format(diff, pconv_backward_opt[0][b][i][c].item(), pconv_backward[0][b][i][c].item()))

    print("Average Abs Error from Finite Diff: default {:.4} \t opt {:.4}".format(err_def/n, err_opt/n))
    print("===============")


for i in range(WARMUP_STEPS+REPEATS):


    ########################################
    # PConv forward
    ########################################

    start = time.time()
    output_pconv = pcf_cuda.pconv_forward(input, neighbor_inds, weights, add_feat)
    torch.cuda.synchronize()
    end = time.time()
    if i > WARMUP_STEPS:
        forward_pconv += end - start

    if i == 0:
      grad = torch.randn_like(output_pconv)

    
    
    ########################################
    # PConv backward regular
    ########################################
    start = time.time()
    pconv_backward = pcf_cuda.pconv_backward(grad,
                                             input, 
                                             neighbor_inds,
                                             weights,
                                             add_feat)
    torch.cuda.synchronize()
    end = time.time()

    if i > WARMUP_STEPS:
        backward_pconv += end - start

    ########################################
    # PConv backward optimized
    ########################################
    start=time.time()
    pconv_backward_opt = pcf_cuda.pconv_backward_opt(grad,
                                             input, 
                                             inv_neighbors.long(),
                                             inv_k.long(),
                                             inv_idx.long(),
                                             neighbor_inds.long(),
                                             weights,
                                             add_feat)
    torch.cuda.synchronize()
    end = time.time()

    if i > WARMUP_STEPS:
        backward_pconv_opt += end - start

    if i == -1: #never
      #print(pconv_backward[0][0,:,:]-pconv_backward_opt[0][0,:,:])
      original_backward_norm = torch.norm(pconv_backward[0])
      optimized_backward_norm = torch.norm(pconv_backward_opt[0])
      diff = torch.norm(pconv_backward[0]-pconv_backward_opt[0])
      print("grad_norm_input (reg, opt, diff):",original_backward_norm.item(), optimized_backward_norm.item(), diff.item())

      original_backward_norm = torch.norm(pconv_backward[1])
      optimized_backward_norm = torch.norm(pconv_backward_opt[1])
      diff = torch.norm(pconv_backward[1]-pconv_backward_opt[1])
      print("grad_norm_weights (reg, opt, diff):",original_backward_norm.item(), optimized_backward_norm.item(), diff.item())

      original_backward_norm = torch.norm(pconv_backward[2])
      optimized_backward_norm = torch.norm(pconv_backward_opt[2])
      diff = torch.norm(pconv_backward[2]-pconv_backward_opt[2])
      print("grad_norm_adds (reg, opt, diff):",original_backward_norm.item(), optimized_backward_norm.item(), diff.item())

    
    
print("PConv forward: {:.3f} ms".format(forward_pconv/REPEATS*1000))
print("PConv backward: {:.3f} ms".format(backward_pconv/REPEATS*1000))
print("PConv_opt backward: {:.3f} ms".format(backward_pconv_opt/REPEATS*1000))
print("Speedup Ratio: {:.3f}".format(backward_pconv/backward_pconv_opt))
