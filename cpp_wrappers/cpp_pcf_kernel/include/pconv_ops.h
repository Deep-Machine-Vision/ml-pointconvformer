//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//
#pragma once

#include <torch/extension.h>
#include <vector>

namespace pcf {
namespace pconv_ops {

/**
 * @brief Forward pass for Point Convolution (PConv)
 * 
 * @tparam scalar_t Data type for tensor elements
 * @param input Input features tensor [B x M x C_in]
 * @param neighbor_inds Neighbor indices tensor [B x N x K]
 * @param weights Weight tensor [B x N x K x C_mid]
 * @param additional_features Additional features tensor [B x N x K x C_add]
 * @param output Output tensor [B x N x (C_mid*(C_in+C_add))]
 */
template <typename scalar_t>
__global__ void pconv_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ weights,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ additional_features,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ output
);

/**
 * @brief Forward pass for fused Point Convolution and Linear layer
 * 
 * @tparam scalar_t Data type for tensor elements
 * @param input Input features tensor [B x M x C_in]
 * @param neighbor_inds Neighbor indices tensor [B x N x K]
 * @param weights Weight tensor [B x N x K x C_mid]
 * @param additional_features Additional features tensor [B x N x K x C_add]
 * @param linear_weights Linear layer weights [C_out x (C_mid*(C_in+C_add))]
 * @param linear_bias Linear layer bias [C_out]
 * @param final_output Final output tensor [B x N x C_out]
 * @param pconv_output PConv output tensor [B x N x (C_mid*(C_in+C_add))]
 */
template <typename scalar_t>
__global__ void pconv_linear_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ weights,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ additional_features,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ linear_weights,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> __restrict__ linear_bias,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ final_output,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ pconv_output
);

/**
 * @brief Backward pass for Point Convolution
 * 
 * @tparam scalar_t Data type for tensor elements
 * @param grad_output
 * @param input Input features tensor [B x M x C_in]
 * @param neighbor_inds Neighbor indices tensor [B x N x K]
 * @param weights Weight tensor [B x N x K x C_mid]
 * @param additional_features Additional features tensor [B x N x K x C_add]
 * @param grad_input
 * @param grad_weights
 * @param grad_additional
 */
template <typename scalar_t>
__global__ void pconv_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ weights,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ additional_features,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_input,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_weights,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_additional
);

/**
 * @brief Backward pass for Fused Point Convolution + Linear Layer
 * 
 * @tparam scalar_t Data type for tensor elements
 * @param grad_output
 * @param input Input features tensor [B x M x C_in]
 * @param neighbor_inds Neighbor indices tensor [B x N x K]
 * @param weights Weight tensor [B x N x K x C_mid]
 * @param additional_features Additional features tensor [B x N x K x C_add]
 * @param linear_weights Linear layer weights [C_out x (C_mid*(C_in+C_add))]
 * @param pconv_output PConv output tensor [B x N x (C_mid*(C_in+C_add))]
 * @param grad_input
 * @param grad_weights
 * @param grad_additional
 * @param grad_linear_weights
 * @param grad_linear_bias
 */
template <typename scalar_t>
__global__ void pconv_linear_cuda_backward_kernel(
        const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_output,
        const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ input,
        const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> __restrict__ neighbor_inds,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ weights,
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ additional_features,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> __restrict__ linear_weights,
        const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ pconv_output,
        torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_input,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_weights,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_additional,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> __restrict__ grad_linear_weights,
        torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> __restrict__ grad_linear_bias
);

/**
 * @brief Backward pass for Fused Point Convolution + Linear Layer with Inverse Indices
 * 
 * @tparam scalar_t Data type for tensor elements
 * @param grad_output
 * @param input Input features tensor [B x M x C_in]
 * @param inverse_neighbor
 * @param inverse_neighbor_k
 * @param inverse_neighbor_idx
 * @param neighbor_inds Neighbor indices tensor [B x N x K]
 * @param weights Weight tensor [B x N x K x C_mid]
 * @param additional_features Additional features tensor [B x N x K x C_add]
 * @param linear_weights Linear layer weights [C_out x (C_mid*(C_in+C_add))]
 * @param pconv_output PConv output tensor [B x N x (C_mid*(C_in+C_add))]
 * @param grad_input
 * @param grad_weights
 * @param grad_additional
 * @param grad_linear_weights
 * @param grad_linear_bias
 */
template <typename scalar_t>
__global__ void pconv_linear_fused_cuda_backward_kernel_opt(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output,
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
        const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inverse_neighbor,
        const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> inverse_neighbor_k,
        const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inverse_neighbor_idx,
        const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> neighbor_inds,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> additional_features,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> linear_weights,
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> pconv_output,
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_input,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_weights,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_additional,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_linear_weights,
        torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_linear_bias
);

/**
 * @brief (Input Only points) Backward pass for Fused Point Convolution + Linear Layer with Inverse Indices
 * 
 * @tparam scalar_t Data type for tensor elements
 * @param grad_output
 * @param input Input features tensor [B x M x C_in]
 * @param inverse_neighbor
 * @param inverse_neighbor_k
 * @param inverse_neighbor_idx
 * @param weights Weight tensor [B x N x K x C_mid]
 * @param linear_weights Linear layer weights [C_out x (C_mid*(C_in+C_add))]
 * @param grad_input
 */
template <typename scalar_t>
__global__ void input_only_backward_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output,
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input,
        const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inverse_neighbor,
        const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> inverse_neighbor_k,
        const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> inverse_neighbor_idx,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> linear_weights,
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_input,
        const int N, const int Nout, const int C_in, const int C_mid, const int C_add, const int C_out, const int input_only_points
);

/**
 * @brief Scattered-Gather using Neighbor Indices around Input points
 * 
 * @tparam scalar_t Data type for tensor elements
 * @param input Input features tensor [B x M x C_in]
 * @param additional_features Additional features tensor [B x N x K x C_add]
 * @param neighbor_inds Neighbor indices tensor [B x N x K]
 * @param concatenated_output
 */
__global__ void gather_kernel(
    const float* input,
    const float* additional_features,
    const int64_t* neighbor_inds,
    float* concatenated_output,
    int B, int M, int Nout, int K, int C_in, int C_add
);

/**
 * @brief Forward pass for PConv operation
 * 
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param weights Weight tensor
 * @param additional_features Additional features tensor
 * @return Output tensor
 */
torch::Tensor pconv_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features
);

/**
 * @brief Forward pass for fused PConv and Linear operation
 * 
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param weights Weight tensor
 * @param additional_features Additional features tensor
 * @param linear_weights Linear layer weights
 * @param linear_bias Linear layer bias
 * @return Vector of output tensors (final_output, pconv_output)
 */
std::vector<torch::Tensor> pconv_linear_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias
);

/**
 * @brief Backward pass for PConv operation
 * 
 * @param grad_output Gradient of output tensor
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param weights Weight tensor
 * @param additional_features Additional features tensor
 * @return Vector of gradient tensors
 */
std::vector<torch::Tensor> pconv_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features
);

/**
 * @brief Backward pass for fused PConv and Linear operation
 * 
 * @param grad_output Gradient of output tensor
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param weights Weight tensor
 * @param additional_features Additional features tensor
 * @param linear_weights Linear layer weights
 * @param pconv_output PConv output tensor
 * @return Vector of gradient tensors
 */
std::vector<torch::Tensor> pconv_linear_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor pconv_output
);

/**
 * @brief Optimized backward pass for fused PConv and Linear operation
 * 
 * @param grad_output Gradient of output tensor
 * @param input Input features tensor
 * @param inverse_neighbor Inverse neighbor tensor
 * @param inverse_neighbor_k Inverse neighbor k tensor
 * @param inverse_neighbor_idx Inverse neighbor index tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param weights Weight tensor
 * @param additional_features Additional features tensor
 * @param linear_weights Linear layer weights
 * @param pconv_output PConv output tensor
 * @return Vector of gradient tensors
 */
std::vector<torch::Tensor> pconv_linear_opt_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor inverse_neighbor,
    torch::Tensor inverse_neighbor_k,
    torch::Tensor inverse_neighbor_idx,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor pconv_output
);

/**
 * @brief Forward pass for PConv and Linear using CUTLASS
 * 
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param weights Weight tensor
 * @param additional_features Additional features tensor
 * @param linear_weights Linear layer weights
 * @param linear_bias Linear layer bias
 * @return Vector of output tensors
 */
std::vector<torch::Tensor> pconv_linear_cutlass_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias
);

} // namespace pconv_ops
} // namespace pcf
