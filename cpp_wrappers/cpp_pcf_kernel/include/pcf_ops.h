//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//
#pragma once

#include <torch/extension.h>
#include <vector>

namespace pcf {
namespace pcf_ops {

/**
 * @brief Kernel: Forward pass for PointConvFormer (PCF)
 * 
 * @param input Input features [B x N x C_in]
 *        B = batch size, N = number of points, C_in = input channels
 * @param neighbor_inds Neighbor indices [B x N x K]
 *        K = number of neighbors per point
 * @param guidance Guidance weights [B x N x K x num_heads]
 * @param weights Weight [B x N x K x C_mid]
 *        C_mid = mid channels
 * @param output Output [B x N x (C_mid*C_in)]
 */
template <typename scalar_t>
__global__ void pcf_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ guidance,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ weights,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ output
);

/**
 * @brief Kernel: Backward pass for PointConvFormer (PCF)
 * 
 * @param grad_output [B x N x (C_mid * C_in)]
 *        B = batch size, N = number of points, C_in = input channels, C_mid = mid channels
 * @param input Input features [B x N x C_in]
 * @param neighbor_inds Neighbor indices [B x N x K]
 *        K = number of neighbors per point
 * @param guidance Guidance weights [B x N x K x num_heads]
 * @param weights Kernel weights [B x N x K x C_mid]
 * @param grad_input Gradient of input tensor [B x N x C_in]
 * @param grad_guidance Gradient of guidance tensor [B x N x K x num_heads]
 * @param grad_weights Gradient of weights tensor [B x N x K x C_mid]
 */
template <typename scalar_t>
__global__ void pcf_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ grad_output,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ guidance,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ weights,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ grad_input,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ grad_guidance,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ grad_weights
);

torch::Tensor pcf_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights
);

std::vector<torch::Tensor> pcf_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights
);

} // namespace pcf_ops
} // namespace pcf
