import torch
import numpy as np
from sklearn.neighbors import KDTree
from pykeops.torch import LazyTensor

def knn_keops(ref_points, query_points, K):
    """
    Compute k-nearest neighbors using KeOps, and return indices as NumPy arrays.
    """
    # Ensure inputs are PyTorch tensors
    if isinstance(ref_points, np.ndarray):
        ref_points = torch.tensor(ref_points, dtype=torch.float32)
    if isinstance(query_points, np.ndarray):
        query_points = torch.tensor(query_points, dtype=torch.float32)
    
    # # Move tensors to CUDA if available
    # ref_points = ref_points.cuda()
    # query_points = query_points.cuda()
    
    # Compute pairwise squared distances using LazyTensor
    ref_lazy = LazyTensor(ref_points[:, None, :])  # Mx1xD
    query_lazy = LazyTensor(query_points[None, :, :])  # 1xNxD
    distances = ((ref_lazy - query_lazy) ** 2).sum(-1)  # Pairwise squared distances (MxN)
    
    # Get top K minimum distances and corresponding indices
    indices = distances.argKmin(K, dim=0)  # Shape: (N, K)
    # indices = indices.cpu().numpy()
    del ref_lazy, query_lazy, distances
    # torch.cuda.empty_cache()
    # Convert the indices to a NumPy array
    return indices  # Move to CPU and convert to NumPy

def compute_knn(ref_points, query_points, K, dilated_rate=1, method='keops'):
    """
    Compute KNN
    Input:
        ref_points: reference points (MxD)
        query_points: query points (NxD)
        K: the amount of neighbors for each point
        dilated_rate: If set to larger than 1, then select more neighbors and then choose from them
        (Engelmann et al. Dilated Point Convolutions: On the Receptive Field Size of Point Convolutions on 3D Point Clouds. ICRA 2020)
        method: Choose between two approaches: Scikit-Learn ('sklearn') or nanoflann ('nanoflann'). In general nanoflann should be faster, but sklearn is more stable
    Output:
        neighbors_idx: for each query point, its K nearest neighbors among the reference points (N x K)
    """
    num_ref_points = ref_points.shape[0]

    if num_ref_points < K or num_ref_points < dilated_rate * K:
        num_query_points = query_points.shape[0]
        inds = np.random.choice(
            num_ref_points, (num_query_points, K)).astype(
            np.int32)

        return inds
    if method == 'sklearn':
        print("Using sklearn")
        kdt = KDTree(ref_points)
        neighbors_idx = kdt.query(
            query_points,
            k=K * dilated_rate,
            return_distance=False)
    elif method == 'keops':
        # print("Using keops")
        neighbors_idx = knn_keops(ref_points, query_points, K*dilated_rate)

    # elif method == 'nanoflann':
    #     neighbors_idx = batch_neighbors(
    #         query_points, ref_points, [
    #             query_points.shape[0]], [num_ref_points], K * dilated_rate)
    else:
        raise Exception('compute_knn: unsupported knn algorithm')
    if dilated_rate > 1:
        neighbors_idx = np.array(
            neighbors_idx[:, ::dilated_rate], dtype=np.int32)

    return neighbors_idx

def tensorizeTensorList(tensor_list):
    """
    Make all tensors inside a list have an additional batch dimension.
    """
    ret_list = []
    for tensor_item in tensor_list:
        if tensor_item is None:
            ret_list.append(None)
        else:
            ret_list.append(tensor_item.unsqueeze(0))
    return ret_list


def tensorize(edges_self, edges_forward, edges_propagate):
    """
    Tensorize transforms a batch of multiple edge sets into a single tensor.
    This focuses only on `edges_self`, `edges_forward`, and `edges_propagate`.
    """
    edges_self = tensorizeTensorList(edges_self)
    edges_forward = tensorizeTensorList(edges_forward)
    edges_propagate = tensorizeTensorList(edges_propagate)

    return edges_self, edges_forward, edges_propagate


def listToBatch(edges_self, edges_forward, edges_propagate):
    """
    ListToBatch transforms a batch of multiple edge sets into a single set.
    Focused on `edges_self`, `edges_forward`, and `edges_propagate`.
    This version uses PyTorch tensors.
    """
    num_sample = len(edges_self)

    # Initialize batches for the first sample
    edgesSelfBatch = edges_self[0]
    edgesForwardBatch = edges_forward[0]
    edgesPropagateBatch = edges_propagate[0]
    
    # Track the cumulative number of points stored
    points_stored = [val.shape[0] for val in edges_self[0]]

    for i in range(1, num_sample):
        for j in range(len(edges_forward[i])):
            # Handle edges_forward
            tempMask = edges_forward[i][j] == -1
            edges_forwardAdd = edges_forward[i][j] + points_stored[j]
            edges_forwardAdd[tempMask] = -1
            edgesForwardBatch[j] = torch.cat([edgesForwardBatch[j], edges_forwardAdd], dim=0)

            # Handle edges_propagate
            tempMask2 = edges_propagate[i][j] == -1
            edges_propagateAdd = edges_propagate[i][j] + points_stored[j + 1]
            edges_propagateAdd[tempMask2] = -1
            edgesPropagateBatch[j] = torch.cat([edgesPropagateBatch[j], edges_propagateAdd], dim=0)

        for j in range(len(edges_self[i])):
            # Handle edges_self
            tempMask3 = edges_self[i][j] == -1
            edges_selfAdd = edges_self[i][j] + points_stored[j]
            edges_selfAdd[tempMask3] = -1
            edgesSelfBatch[j] = torch.cat([edgesSelfBatch[j], edges_selfAdd], dim=0)

            # Update points_stored
            points_stored[j] += edges_self[i][j].shape[0]

    return edgesSelfBatch, edgesForwardBatch, edgesPropagateBatch

def prepare(edges_self, edges_forward, edges_propagate):
    """
    Prepare data coming from data loader (lists of numpy arrays) into torch tensors ready to send to training.
    Focused on `edges_self`, `edges_forward`, and `edges_propagate`.
    """
    edges_self_out, edges_forward_out, edges_propagate_out = \
        listToBatch(edges_self, edges_forward, edges_propagate)

    edges_self_out, edges_forward_out, edges_propagate_out = \
        tensorize(edges_self_out, edges_forward_out, edges_propagate_out)

    return edges_self_out, edges_forward_out, edges_propagate_out


def make_cumulative(original):
    
    for i in range(0, len(original)):
        temp_list = []
        temp_list.append(0)
        for itr in original[i]:
            temp_list.append(temp_list[-1] + itr)
        original[i] = temp_list
    # for i in range(1, len(original)):
        
            
    return original

def compute_knn_packed(pointclouds, points_stored, K_self=16, K_forward=16, K_propagate=16):
    """
    Compute kNN for the given pointclouds after tensorization and batching.
    Input:
        pointclouds: list of pointclouds from different grid sizes, each of shape [N_j, 3]
        K_self: number of neighbors within each subsampling level
        K_forward: number of neighbors from one subsampling level to the next one (downsampling)
        K_propagate: number of neighbors from one subsampling level to the previous one (upsampling)
    Outputs:
        nei_forward_list: downsampling kNN neighbors (K_forward neighbors for each point)
        nei_propagate_list: upsampling kNN neighbors (K_propagate neighbors for each point)
        nei_self_list: kNN neighbors within the same layer
    """

    
    n = len(points_stored[0])

    nei_forward_list   = [[] for _ in range(n)]
    nei_propagate_list = [[] for _ in range(n)]
    nei_self_list      = [[] for _ in range(n)]

    points_stored = make_cumulative(points_stored)
   
    for i in range(n):

        temp_points = []
        temp_points.append(pointclouds[0][:, points_stored[0][i]:points_stored[0][i+1], :])
        temp_points.append(pointclouds[1][:, points_stored[1][i]:points_stored[1][i+1], :])
        temp_points.append(pointclouds[2][:, points_stored[2][i]:points_stored[2][i+1], :])
        temp_points.append(pointclouds[3][:, points_stored[3][i]:points_stored[3][i+1], :])
        temp_points.append(pointclouds[4][:, points_stored[4][i]:points_stored[4][i+1], :])
        
        nei_forward_list_temp   = []
        nei_self_list_temp      = []
        nei_propagate_list_temp = []

        for j in range(len(pointclouds)):
            temp_points[j] = temp_points[j].squeeze(0)
            if j == 0:
                nself = compute_knn(temp_points[j], temp_points[j], K_self)
                nei_self_list_temp.append(nself)
            else:
                nei_forward   = compute_knn(temp_points[j-1], temp_points[j], K_forward)
                nei_propagate = compute_knn(temp_points[j], temp_points[j-1], K_propagate)
                nself         = compute_knn(temp_points[j], temp_points[j], K_self)
                
                nei_forward_list_temp.append(nei_forward)
                nei_propagate_list_temp.append(nei_propagate)
                nei_self_list_temp.append(nself)
        
        nei_forward_list[i]   = nei_forward_list_temp
        nei_propagate_list[i] = nei_propagate_list_temp
        nei_self_list[i]      = nei_self_list_temp
        # print(len(nei_self_list_temp))

    return nei_self_list, nei_forward_list, nei_propagate_list
