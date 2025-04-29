//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//
#pragma once

#include <torch/extension.h>

namespace pcf {

/**
 * @brief Macro to check if tensor is on CUDA device
 */
#define CHECK_CUDA(x) {TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")}

/**
 * @brief Macro to check if tensor is contiguous
 */
#define CHECK_CONTIGUOUS(x) {TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")}

/**
 * @brief Macro to check if tensor is on CUDA device and contiguous
 */
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/**
 * @brief Forward pass for PCF
 * 
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param guidance Guidance weights tensor
 * @param weights Weight tensor
 * @return Output tensor
 */
torch::Tensor pcf_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights
);

/**
 * @brief Backward pass for PCF
 * 
 * @param grad_output Gradient of output tensor
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param guidance Guidance weights tensor
 * @param weights Weight tensor
 * @return Vector of gradient tensors
 */
std::vector<torch::Tensor> pcf_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights
);

/**
 * @brief Forward pass for Point Convolution (PConv)
 * 
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param weights Weight tensor
 * @param additional_features Additional features tensor
 * @return Output tensor
 */
torch::Tensor pconv_forward(
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
 * @return Vector of output tensors
 */
std::vector<torch::Tensor> pconv_linear_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias
);

/**
 * @brief Backward pass for PConv
 * 
 * @param grad_output Gradient of output tensor
 * @param input Input features tensor
 * @param neighbor_inds Neighbor indices tensor
 * @param weights Weight tensor
 * @param additional_features Additional features tensor
 * @return Vector of gradient tensors
 */
std::vector<torch::Tensor> pconv_backward(
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
std::vector<torch::Tensor> pconv_linear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor pconv_output
);

/**
 * @brief Compute KNN inverse mapping
 * 
 * @param neighbor_inds Neighbor indices tensor
 * @param total_points Total number of points
 * @return Vector of output tensors
 */
std::vector<torch::Tensor> compute_knn_inverse(
    torch::Tensor neighbor_inds,
    const int total_points
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
std::vector<torch::Tensor> pconv_linear_opt_backward(
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
std::vector<torch::Tensor> pconv_linear_cutlass(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias
);

} // namespace pcf
