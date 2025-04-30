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
 * @brief Forward pass for PointConvFormer (PCF)
 * 
 * @tparam scalar_t Data type for tensor elements
 * @param input Input features tensor [B x N x C_in]
 * @param neighbor_inds Neighbor indices tensor [B x N x K]
 * @param guidance Guidance weights tensor [B x N x K x num_heads]
 * @param weights Weight tensor [B x N x K x C_mid]
 * @param output Output tensor [B x N x (C_mid*C_in)]
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
 * @brief Backward pass for PointConvFormer (PCF)
 * 
 * @tparam scalar_t Data type for tensor elements
 * @param grad_output Gradient of output tensor
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param guidance Guidance weights tensor
 * @param weights Weight tensor
 * @param grad_input Gradient of input tensor
 * @param grad_guidance Gradient of guidance tensor
 * @param grad_weights Gradient of weights tensor
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

/**
 * @brief Forward pass for PCF operation
 * 
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param guidance Guidance weights tensor
 * @param weights Weight tensor
 * @return Output tensor
 */
torch::Tensor pcf_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights
);

/**
 * @brief Backward pass for PCF operation
 * 
 * @param grad_output Gradient of output tensor
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param guidance Guidance weights tensor
 * @param weights Weight tensor
 * @return Vector of gradient tensors (grad_input, grad_guidance, grad_weights)
 */
std::vector<torch::Tensor> pcf_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights
);

} // namespace pcf_ops
} // namespace pcf
