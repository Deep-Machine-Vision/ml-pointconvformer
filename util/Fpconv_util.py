import numpy as np

def createInverse(neighborMat, inp_points):
    N = inp_points  # Total number of points
    K = neighborMat.shape[1]  # Number of neighbors per point

    # Initialize lists to store neighbors and inverse indices
    neigh = [[] for _ in range(N)]
    inv_k = [[] for _ in range(N)]

    for r in range(neighborMat.shape[0]):
        for c in range(K):
            # Ensure neighborMat[r, c] is a scalar integer
            point_index = neighborMat[r, c]
            neigh[point_index].append(r)
            inv_k[point_index].append(c)

    # Prepare the indexing for the flattened structure
    idx = [len(x) for x in neigh]
    cum_sum = 0
    for i in range(N):
        if idx[i] == 0:
            idx[i] = -1
        else:
            temp = idx[i]
            idx[i] = cum_sum
            cum_sum = temp + cum_sum

    # Add the total number of neighbors as the final index
    idx.append(neighborMat.shape[0] * K)

    # Flatten the list of neighbors and inverse k values
    neighbors = np.hstack(neigh).flatten()
    inv_k = np.hstack(inv_k).flatten()

    return neighbors, np.array(inv_k), np.array(idx)

def round_matrix(neighbor_inds, inp_points):
    # Create round matrix
    inv_neighbors_list = []
    inv_k_list = []
    inv_idx_list = []

    for i in range(len(neighbor_inds)):
        # Call createInverse for each item in neighbor_inds
        inv_neighbors, inv_k, inv_idx = createInverse(neighbor_inds[i], inp_points[i].shape[0])
        
        # Append the NumPy arrays directly to the lists
        inv_neighbors_list.append(inv_neighbors)
        inv_k_list.append(inv_k)
        inv_idx_list.append(inv_idx)

    return inv_neighbors_list, inv_k_list, inv_idx_list
