//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//
#pragma once

#include <torch/extension.h>
#include <stdio.h>
#include <vector>

namespace pcf {
namespace knn {

/**
 * @brief Count neighbors for each point in the point cloud
 * 
 * @param neighbor_inds Tensor containing neighbor indices
 * @param counts Output tensor to store neighbor counts
 * @param total_points Total number of points in the point cloud
 * @param start_point Starting point index
 * @param batch_idx Batch index
 */
__global__ void count_neighbors_kernel(
    const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> neighbor_inds,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> counts,
    const int total_points,
    const int start_point,
    const int batch_idx
);

/**
 * @brief Compute inverse index mapping for KNN
 * 
 * @param counts Tensor containing neighbor counts
 * @param inv_idx Output tensor to store inverse indices
 * @param total_points Total number of points in the point cloud
 * @param batch_idx Batch index
 */
__global__ void compute_inv_idx_kernel(
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> counts,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inv_idx,
    const int total_points,
    const int batch_idx
);

/**
 * @brief Fill inverse mapping for KNN
 * 
 * @param neighbor_inds Tensor containing neighbor indices
 * @param inv_neighbors Output tensor to store inverse neighbors
 * @param inv_k Output tensor to store inverse k values
 * @param running_counts Running counts tensor
 * @param inv_idx Inverse index tensor
 * @param total_points Total number of points in the point cloud
 * @param start_point Starting point index
 * @param batch_idx Batch index
 */
__global__ void fill_inverse_kernel(
    const torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> neighbor_inds,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inv_neighbors,
    torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> inv_k,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> running_counts,
    torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inv_idx,
    const int total_points,
    const int start_point,
    const int batch_idx
);

/**
 * @brief Forward pass for KNN inverse computation
 * 
 * @param neighbor_inds Neighbor indices tensor
 * @param total_points Total number of points
 * @return Vector of output tensors (inverse neighbors, inverse k, inverse indices)
 */
std::vector<torch::Tensor> knn_inverse_cuda_forward(
    torch::Tensor neighbor_inds,
    const int total_points
);

} // namespace knn
} // namespace pcf
