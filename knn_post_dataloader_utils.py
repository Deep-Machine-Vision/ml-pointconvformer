import torch
import numpy as np
from sklearn.neighbors import KDTree
from pykeops.torch import LazyTensor
from cuvs.neighbors import brute_force
import cupy as cp



def knn_cuvs_brute_force(ref_points, query_points, K):
    """
    Compute k-nearest neighbors using brute force, and return indices.
    """
    ref_points = cp.asarray(ref_points)
    query_points = cp.asarray(query_points)
    index = brute_force.build(ref_points, metric='sqeuclidean')

    dist, ind = brute_force.search(index, query_points, K)
    ind = cp.asnumpy(ind)
    return ind

def knn_keops(ref_points, query_points, K):
    """
    Compute k-nearest neighbors using KeOps, and return indices.
    """
    if isinstance(ref_points, np.ndarray):
        ref_points = torch.tensor(ref_points, dtype=torch.float32)
    if isinstance(query_points, np.ndarray):
        query_points = torch.tensor(query_points, dtype=torch.float32)
    
    
    ref_lazy = LazyTensor(ref_points[:, None, :])  # Mx1xD
    query_lazy = LazyTensor(query_points[None, :, :])  # 1xNxD
    distances = ((ref_lazy - query_lazy) ** 2).sum(-1)  # Pairwise squared distances (MxN)
   
    indices = distances.argKmin(K, dim=0)
    
    del ref_lazy, query_lazy, distances
    assert isinstance(indices, torch.Tensor), "indices is not a torch.Tensor"

    return indices

def compute_knn(ref_points, query_points, K, dilated_rate=1, method='keops'):
    """
    Compute KNN
    Input:
        ref_points: reference points (MxD)
        query_points: query points (NxD)
        K: the amount of neighbors for each point
        dilated_rate: If set to larger than 1, then select more neighbors and then choose from them
        (Engelmann et al. Dilated Point Convolutions: On the Receptive Field Size of Point Convolutions on 3D Point Clouds. ICRA 2020)
    Output:
        neighbors_idx: for each query point, its K nearest neighbors among the reference points (N x K)
    """
    num_ref_points = ref_points.shape[0]


    if num_ref_points < K or num_ref_points < dilated_rate * K:
        num_query_points = query_points.shape[0]
        inds = np.random.choice(
            num_ref_points, (num_query_points, K)).astype(
            np.int32)
        # Convert to torch tensor if ref_points is a torch tensor
        if isinstance(ref_points, torch.Tensor):
            inds = torch.tensor(inds).to(ref_points.device)
        return inds
    if method == 'sklearn':
        kdt = KDTree(ref_points)
        neighbors_idx = kdt.query(
            query_points,
            k=K * dilated_rate,
            return_distance=False)
    
    elif method == 'keops':
        neighbors_idx = knn_keops(ref_points, query_points, K*dilated_rate)
    
    elif method == 'nvidia_cuvs_brute_force':
        neighbors_idx = knn_cuvs_brute_force(ref_points, query_points, K*dilated_rate)
    else:
        raise Exception('compute_knn: unsupported knn algorithm')
    if dilated_rate > 1:
        neighbors_idx = np.array(
            neighbors_idx[:, ::dilated_rate], dtype=np.int32)

    if method == 'keops':
        assert isinstance(neighbors_idx, torch.Tensor), "neighbors_idx is not a torch.Tensor"
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
    """
    edges_self = tensorizeTensorList(edges_self)
    edges_forward = tensorizeTensorList(edges_forward)
    edges_propagate = tensorizeTensorList(edges_propagate)

    return edges_self, edges_forward, edges_propagate


def listToBatch(edges_self, edges_forward, edges_propagate):
    """
    ListToBatch transforms a batch of multiple edge sets into a single set.
    """
    num_sample = len(edges_self)

    # Initialize batches for the first sample
    edgesSelfBatch = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in edges_self[0]]
    edgesForwardBatch = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in edges_forward[0]]
    edgesPropagateBatch = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in edges_propagate[0]]
    
    # Track the cumulative number of points stored
    points_stored = [val.shape[0] for val in edges_self[0]]

    for i in range(1, num_sample):
        for j in range(len(edges_forward[i])):
            # Handle edges_forward
            edges_forwardAdd = torch.tensor(edges_forward[i][j]) if isinstance(edges_forward[i][j], np.ndarray) else edges_forward[i][j]
            tempMask = edges_forwardAdd == -1
            edges_forwardAdd = edges_forwardAdd + points_stored[j]
            edges_forwardAdd[tempMask] = -1
            edgesForwardBatch[j] = torch.cat([edgesForwardBatch[j], edges_forwardAdd], dim=0)

            # Handle edges_propagate
            edges_propagateAdd = torch.tensor(edges_propagate[i][j]) if isinstance(edges_propagate[i][j], np.ndarray) else edges_propagate[i][j]
            tempMask2 = edges_propagateAdd == -1
            edges_propagateAdd = edges_propagateAdd + points_stored[j + 1]
            edges_propagateAdd[tempMask2] = -1
            edgesPropagateBatch[j] = torch.cat([edgesPropagateBatch[j], edges_propagateAdd], dim=0)

        for j in range(len(edges_self[i])):
            # Handle edges_self
            edges_selfAdd = torch.tensor(edges_self[i][j]) if isinstance(edges_self[i][j], np.ndarray) else edges_self[i][j]
            tempMask3 = edges_selfAdd == -1
            edges_selfAdd = edges_selfAdd + points_stored[j]
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



def compute_knn_packed(pointclouds, points_stored, K_self, K_forward, K_propagate):
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


    points_stored = [np.cumsum([0] + sublist).tolist() for sublist in points_stored]   
    
    for i in range(n):

        temp_points = []
        for j in range(len(pointclouds)):
            temp_points.append(pointclouds[j][:, points_stored[j][i]:points_stored[j][i+1], :])
        
        nei_forward_list_temp   = []
        nei_self_list_temp      = []
        nei_propagate_list_temp = []

        for j in range(len(pointclouds)):
            temp_points[j] = temp_points[j].squeeze(0)
            if j == 0:
                nself = compute_knn(temp_points[j], temp_points[j], K_self[j])
                nei_self_list_temp.append(nself)
            else:
                nei_forward   = compute_knn(temp_points[j-1], temp_points[j], K_forward[j])
                nei_propagate = compute_knn(temp_points[j], temp_points[j-1], K_propagate[j])
                nself         = compute_knn(temp_points[j], temp_points[j], K_self[j])
                
                nei_forward_list_temp.append(nei_forward)
                nei_propagate_list_temp.append(nei_propagate)
                nei_self_list_temp.append(nself)
        
        nei_forward_list[i]   = nei_forward_list_temp
        nei_propagate_list[i] = nei_propagate_list_temp
        nei_self_list[i]      = nei_self_list_temp
       

    return nei_self_list, nei_forward_list, nei_propagate_list
