//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

__host__ __device__ inline int nextPowerOf2(int n) {
    n--;           // Decrement n to handle the case when n is already a power of 2
    n |= n >> 1;   // Set all bits after the highest set bit to 1
    n |= n >> 2;   // Continue setting bits
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;  // Add 1 to get the next power of 2
}

// Helper function to convert float to half
__global__ void convert_to_half(
    const float* input,
    half* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}


// Simple counting to get the number of neighbors for each point
__global__ void count_neighbors_kernel(
        const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> neighbor_inds,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> counts,
        const int total_points,
        const int start_point,
        const int batch_idx
) {
        const int local_point_idx = blockIdx.y;
        const int point_idx = start_point + local_point_idx;
        const int tid = threadIdx.x;
        const int K = neighbor_inds.size(2);

        // Process K neighbors in strided fashion
        for (int k = tid; k < K; k += blockDim.x) {
                const int neighbor = neighbor_inds[batch_idx][point_idx][k];
                if (neighbor >= 0 && neighbor < total_points) {
                        atomicAdd(&counts[neighbor], 1);
                }
        }
}

// Compute inv_idx for counts
__global__ void compute_inv_idx_kernel(
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> counts,
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> inv_idx,
        const int total_points,
        const int batch_idx
) {
        inv_idx[batch_idx][0] = 0;
        for (int i = 0; i < total_points; i++) {
                inv_idx[batch_idx][i + 1] = inv_idx[batch_idx][i] + counts[i];
        }
}

// Fill inverse neighbors and k indices
__global__ void fill_inverse_kernel(
        const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> neighbor_inds,
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> inv_neighbors,
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> inv_k,
        torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> running_counts,
        torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> inv_idx,
        const int total_points,
        const int start_point,
        const int batch_idx
) {
        const int local_point_idx = blockIdx.y;
        const int point_idx = start_point + local_point_idx;
        const int tid = threadIdx.x;
        const int K = neighbor_inds.size(2);

        // Process K neighbors in strided fashion
        for (int k = tid; k < K; k += blockDim.x) {
                const int neighbor = neighbor_inds[batch_idx][point_idx][k];
                if (neighbor >= 0 && neighbor < total_points) {
                        const int pos = atomicAdd(&running_counts[neighbor], 1);
                        const int idx = inv_idx[batch_idx][neighbor] + pos;

                        inv_neighbors[batch_idx][idx] = point_idx;
                        inv_k[batch_idx][idx] = k;
                }
        }
}


// First goal is to get forward to work to speed up the inference
// Second goal is to get backward to work to save GPU memory so that we can deal with 2cm better and have larger batch sizes

template <typename scalar_t>
__global__ void pcf_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ guidance,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ weights,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ output)
{
/* input: B x N x C_in tensor, B = batch size, N = number of points, C_in = number of channels, input features
   neighbor_inds: B x N x K tensor, K = neighborhood size, indices of the neighborhood of each point
   guidance: B x N x K x num_heads tensor, guidance weight for each point in each neighborhood
   weights: B x N x K x C_mid, C_mid = number of middle channels
   output: B x N x (C_mid*C_in), final output of the PCF layer
   
   This implements a fused layer of:
  sum_{i} input[b,n][neighbor_inds[i]][j] * guidance[b,n,head[j],i] * weights[b,n,k,i]
   It outputs a tensor of shape B x N x (C_mid * C_in)
   It avoids serializing the input hence faster than naive pyTorch implementation
  */
  	int i,k,ii,jj,kk, iter0;
	const int B = input.size(0);
	const int N = input.size(1);
	const int Nout = neighbor_inds.size(1);
	const int C_in = input.size(2);
	const int K = neighbor_inds.size(2);
	const int C_mid = weights.size(3);
	const int increment = blockDim.x / C_mid;
	const int num_heads = guidance.size(3);
  	/* parallelize ii and i on blocks */
  	// Supposedly blockIdx.x should go up to B * N
  	for (iter0 = blockIdx.x; iter0< B * Nout; iter0+= gridDim.x)
  	{
  		// ii is the index on the batch dimension
  	  	ii = iter0 / Nout;
  	  	// i is the index on the point dimension
  	  	i = iter0 % Nout;
		// Suppose each point is a block, then split it into threads
		// output channels is at least 8, C_mid is usually at least 16, so we are safe dividing on this dimension
                // C_mid is at most 16, so for sure each C_mid gets its own thread, and maybe we have many threads for the same C_mid
                jj = threadIdx.x / increment;
		// Throw out the excessive threads
		if (jj >= C_mid)
			continue;
		#pragma unroll
		for(kk=threadIdx.x % increment;kk<C_in;kk+=increment)
		{
			scalar_t partial_sum = 0.0;
			long cur_head = kk % num_heads;
			#pragma unroll
			for (k=0;k<K;k++)
			{
                        	scalar_t real_weight = weights[ii][i][k][jj] * guidance[ii][i][k][cur_head];
				partial_sum += input[ii][neighbor_inds[ii][i][k]][kk] * real_weight; 
			}
			output[ii][i][jj + kk*C_mid] = partial_sum;
		}
         }
}

template <typename scalar_t>
__global__ void pconv_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ weights,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ additional_features,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ output)
{
/* input: B x M x C_in tensor, B = batch size, M = number of points in the original point cloud, C_in = number of channels, input features
   neighbor_inds: B x N x K tensor, K = neighborhood size, indices of the neighborhood of each point
   weights: B x N x K x C_mid, C_mid = number of middle channels
   additional_features: B x N x K x C_add, additional features that do not require indexing
   output: B x N x (C_mid*C_in), final output of the PCF layer
   
   This implements a fused layer of:
  concat(sum_{i} input[b,n][neighbor_inds[i]][j] * weights[b,n,k,i], sum_{i} additional_features[b,n][i][j] * weights[b,n,k,i]
   It outputs a tensor of shape B x N x (C_mid * (C_in + C_add)
   It avoids serializing the input hence faster than naive pyTorch implementation
  */

        extern __shared__ unsigned char memory[];
        scalar_t* shared_mem = reinterpret_cast<scalar_t*>(memory);

        const int B = input.size(0);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
        const int K = neighbor_inds.size(2);
        const int C_mid = weights.size(3);
        const int C_add = additional_features.size(3);

        // Each thread block handles one point
        const int batch_idx = blockIdx.x / Nout;
        const int point_idx = blockIdx.x % Nout;
        const int tid = threadIdx.x;

        // Shared memory
        scalar_t* shared_input = shared_mem;
        scalar_t* shared_additional = shared_input + K * C_in;

        // Load input features
        for (int k = 0; k < K; k++) {
                const int n_idx = neighbor_inds[batch_idx][point_idx][k];
                for (int c = tid; c < C_in; c += blockDim.x) {
                        shared_input[k * C_in + c] = input[batch_idx][n_idx][c];
                }
        }

        // Load additional features
        if (C_add > 0) {
                for (int k = 0; k < K; k++) {
                        for (int c = tid; c < C_add; c += blockDim.x) {
                                shared_additional[k * C_add + c] = additional_features[batch_idx][point_idx][k][c];
                        }
                }
        }

        __syncthreads();

        // PConv
        const int total_channels = C_mid * (C_in + C_add);
        for (int c = tid; c < total_channels; c += blockDim.x) {
                const int mid_idx = c % C_mid;
                const int in_idx = c / C_mid;

                scalar_t sum = 0;
                for (int k = 0; k < K; k++) {
                        if (in_idx < C_in) {
                                sum += shared_input[k * C_in + in_idx] * weights[batch_idx][point_idx][k][mid_idx];
                        } else {
                                sum += shared_additional[k * C_add + (in_idx - C_in)] * weights[batch_idx][point_idx][k][mid_idx];
                        }
                }
                output[batch_idx][point_idx][c] = sum;
        }
}

template <typename scalar_t>
__global__ void pconv_linear_cuda_forward_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ input,
        const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> __restrict__ neighbor_inds,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ weights,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ additional_features,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ linear_weights,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> __restrict__ linear_bias,
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ final_output,
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ pconv_output)
{
/*
    This kernel implements a fused Point Convolution (PConv) and Linear layer.
    
    Input Tensors:
    - input: [B x M x C_in] Input features for each point
        B = batch size, M = number of points, C_in = input channels
    - neighbor_inds: [B x N x K] Indices of K neighbors for each point
        N = number of output points, K = neighborhood size
    - weights: [B x N x K x C_mid] Weight matrix for point convolution
        C_mid = middle/intermediate channels
    - additional_features: [B x N x K x C_add] Additional point features
        C_add = number of additional feature channels
    - linear_weights: [C_out x (C_mid*(C_in + C_add))] Linear layer weights
    - linear_bias: [C_out] Linear layer bias
    
    Output:
    - final_output: [B x N x C_out] Final output after PConv and Linear layer
    
    Steps:
    1. PConv: For each point, aggregate features from its neighbors using weights
    2. Concatenate results from input features and additional features
    3. Apply linear transformation to the concatenated features
*/
        extern __shared__ unsigned char memory[];
        scalar_t* shared_mem = reinterpret_cast<scalar_t*>(memory);

        const int K = neighbor_inds.size(2);
        const int C_in = input.size(2);
        const int C_mid = weights.size(3);
        const int C_add = additional_features.size(3);
        const int C_out = linear_weights.size(0);
        
        const int batch_idx = blockIdx.x / gridDim.y;   // Current batch
        const int point_idx = blockIdx.y;               // Current point in point cloud
        const int tid = threadIdx.x;                    // Thread ID within block
        const int warp_id = tid / 32;                   // Warp ID (32 threads per warp)
        const int lane_id = tid % 32;                   // Lane ID within the warp

        // Shared memory
        scalar_t* shared_input = shared_mem;
        scalar_t* shared_additional = shared_input + (K * C_in);
        scalar_t* shared_weights = shared_additional + (K * C_add);
        scalar_t* shared_intermediate = shared_weights + (K * C_mid);   // For PConv output

        // Load input features
        #pragma unroll
        for (int i = tid; i < K * C_in; i += blockDim.x) {
                const int k = i / C_in;         // Neighbor index
                const int c = i % C_in;         // Channel index
                const int n_idx = neighbor_inds[batch_idx][point_idx][k];
                shared_input[i] = input[batch_idx][n_idx][c];
        }

        // Load additional features
        #pragma unroll
        for (int i = tid; i < K * C_add; i += blockDim.x) {
                const int k = i / C_add;
                const int c = i % C_add;
                shared_additional[i] = additional_features[batch_idx][point_idx][k][c];
        }

        // Load weights
        #pragma unroll
        for (int i = tid; i < K * C_mid; i += blockDim.x) {
                const int k = i / C_mid;
                const int c = i % C_mid;
                shared_weights[i] = weights[batch_idx][point_idx][k][c];
        }

        __syncthreads();

        // PConv -> parallelize across output channels
        const int total_channels = C_mid * (C_in + C_add);      // Total channels after concat

        #pragma unroll
        for (int c = tid; c < total_channels; c += blockDim.x) {
                const int mid_idx = c % C_mid;
                const int in_idx = c / C_mid;
                
                scalar_t sum = 0;
                #pragma unroll
                for (int k = 0; k < K; k++) {
                if (in_idx < C_in) {
                        sum += shared_input[k * C_in + in_idx] * shared_weights[k * C_mid + mid_idx];
                } else {
                        sum += shared_additional[k * C_add + (in_idx - C_in)] * shared_weights[k * C_mid + mid_idx];
                }
                }
                shared_intermediate[c] = sum;
                pconv_output[batch_idx][point_idx][c] = sum;  // Store PConv output
        }

        __syncthreads();

        // Linear layer -> each thread handles one output channel
        if (tid < C_out) {
                scalar_t sum = 0;
                #pragma unroll
                for (int c = 0; c < total_channels; c++) {
                sum += shared_intermediate[c] * linear_weights[tid][c];
                }
                final_output[batch_idx][point_idx][tid] = sum + linear_bias[tid];
        }
}

template <typename scalar_t>
__global__ void pcf_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ guidance,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ weights,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_input,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_guidance,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_weights)
/* 
   grad_output: B x N x (C_mid * C_in), the gradient derived from above
   input: B x N x C_in tensor, B = batch size, N = number of points, C_in = number of channels, input features
   neighbor_inds: B x N x K tensor, K = neighborhood size, indices of the neighborhood of each point
   guidance: B x N x K x num_heads tensor, guidance weight for each point in each neighborhood
   weights: B x N x K x C_mid , C_mid = number of middle channels
   grad_input: same shape as input, gradient of input
   grad_guidance: same shape as guidance, gradient of guidance
   grad_weights: same shape as weights, gradient of weights

   Forward is:
    sum_{i} input[b,n][neighbor_inds[i]][j] * guidance[b,n,head[j],i] * weights[b,n,k,i]
*/

{
        int i,j,k, ii, kk,iter0;
        const int B = input.size(0);
        const int N = input.size(1);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
        const int K = neighbor_inds.size(2);
        const int C_mid = weights.size(3);
        const int increment = blockDim.x / C_mid;
        const int num_heads = guidance.size(3);
	const int cur_mid = threadIdx.x / increment;
	if (cur_mid >= C_mid)
		return;
        // Supposedly blockIdx.x should go up to B * N
        for (iter0 = blockIdx.x; iter0< B * Nout; iter0+= gridDim.x)
        {
                // ii is the index on the batch dimension
                ii = iter0 / Nout;
                // i is the index on the point dimension
                i = iter0 % Nout;
		k = threadIdx.x % K;
		scalar_t weight_grad_temp = 0.0;
		// Max number of heads
		scalar_t guidance_grad_temp[32];
		scalar_t cur_compute;
		for (kk=0;kk<num_heads;kk++)
			guidance_grad_temp[kk] = 0.0;
		#pragma unroll
		for (kk=0;kk<C_in;kk++)
		{
			long cur_channel = cur_mid * C_in + kk;
			long cur_head = kk % num_heads;
			cur_compute = grad_output[ii][i][cur_channel] * weights[ii][i][k][cur_mid];
       			guidance_grad_temp[cur_head] += cur_compute * input[ii][neighbor_inds[ii][i][k]][kk];
			weight_grad_temp += grad_output[ii][i][cur_channel] * guidance[ii][i][k][cur_head] * input[ii][neighbor_inds[ii][i][k]][kk];
		// It would be quite expensive to store this in shared memory so use this naive approach for now, using atomicAdd to avoid racing conditions
			atomicAdd(&grad_input[ii][neighbor_inds[ii][i][k]][kk],  guidance[ii][i][k][cur_head] * cur_compute);
		}
		grad_weights[ii][i][k][cur_mid] = weight_grad_temp;
		#pragma unroll
		for (kk=0;kk<num_heads;kk++)
			atomicAdd(&grad_guidance[ii][i][k][kk],guidance_grad_temp[kk]);
		__syncthreads();
	}
}

template <typename scalar_t>
__global__ void pconv_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_output,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ input,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> __restrict__ neighbor_inds,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ weights,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ additional_features,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> __restrict__ grad_input,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_weights,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> __restrict__ grad_additional)
/* 
   grad_output: B x N x (C_mid * C_in), the gradient derived from above
   input: B x N x C_in tensor, B = batch size, N = number of points, C_in = number of channels, input features
   neighbor_inds: B x N x K tensor, K = neighborhood size, indices of the neighborhood of each point
   weights: B x N x K x C_mid, C_mid = number of middle channels
   additional_features: B x N x K x C_add, additional features that do not need gather
   grad_input: same shape as input, gradient of input
   grad_weights: same shape as weights, gradient of weights

   Forward is:
    sum_{i} input[b,n][neighbor_inds[i]][j] * weights[b,n,k,i]
*/

{
        int i,j,k, ii, kk,iter0;
        const int B = input.size(0);
        const int N = input.size(1);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
        const int K = neighbor_inds.size(2);
        const int C_mid = weights.size(3);
	const int C_add = additional_features.size(3);
        const int increment = blockDim.x / C_mid;
        const int cur_mid = threadIdx.x / increment;
        if (cur_mid >= C_mid)
                return;
        // Supposedly blockIdx.x should go up to B * N
        for (iter0 = blockIdx.x; iter0< B * Nout; iter0+= gridDim.x)
        {
                // ii is the index on the batch dimension
                ii = iter0 / Nout;
                // i is the index on the point dimension
                i = iter0 % Nout;
                k = threadIdx.x % K;
                scalar_t weight_grad_temp = 0.0;
                scalar_t cur_compute;
                #pragma unroll
                for (kk=0;kk<C_in;kk++)
                {
                        long cur_channel = cur_mid * (C_in + C_add) + kk;
                        cur_compute = grad_output[ii][i][cur_channel] * weights[ii][i][k][cur_mid];
                        weight_grad_temp += grad_output[ii][i][cur_channel] * input[ii][neighbor_inds[ii][i][k]][kk];
                // It would be quite expensive to store this in shared memory so use this naive approach for now, using atomicAdd to avoid racing conditions
                        atomicAdd(&grad_input[ii][neighbor_inds[ii][i][k]][kk], cur_compute);
                }
		for (kk=0;kk<C_add;kk++)
		{
			long cur_channel = cur_mid * (C_in + C_add) + kk + C_in;
			cur_compute = grad_output[ii][i][cur_channel] * weights[ii][i][k][cur_mid];
			weight_grad_temp += grad_output[ii][i][cur_channel] * additional_features[ii][i][k][kk];
			grad_additional[ii][i][k][kk] = cur_compute;
		}
                grad_weights[ii][i][k][cur_mid] = weight_grad_temp;
	}
}

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
        torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> __restrict__ grad_linear_bias)
{

        extern __shared__ unsigned char memory[];
        scalar_t* shared_grad = reinterpret_cast<scalar_t*>(memory);

        const int B = input.size(0);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
        const int K = neighbor_inds.size(2);
        const int C_mid = weights.size(3);
        const int C_add = additional_features.size(3);
        const int C_out = grad_output.size(2);
        const int total_channels = C_mid * (C_in + C_add);

        const int batch_idx = blockIdx.x;
        const int point_idx = blockIdx.y;
        const int tid = threadIdx.x;
        const int increment = blockDim.x / C_mid;

        // grad_x = grad_output * W^T for intermediate gradient
        for (int c = tid; c < total_channels; c += blockDim.x) {
                scalar_t sum = 0;
                for (int c_out = 0; c_out < C_out; c_out++) {
                sum += grad_output[batch_idx][point_idx][c_out] * linear_weights[c_out][c];
                }
                shared_grad[c] = sum;
        }

        __syncthreads();

        // additional features gradient - each thread processes one (k, c_mid, c_add) combination
        if (tid < K * C_mid * C_add) {
                const int k = tid % K;
                const int mid_idx = (tid / K) % C_mid;
                const int add_idx = tid / (K * C_mid);
                
                const int grad_idx = mid_idx * (C_in + C_add) + C_in + add_idx;
                grad_additional[batch_idx][point_idx][k][add_idx] = 
                shared_grad[grad_idx] * weights[batch_idx][point_idx][k][mid_idx];
        }

        __syncthreads();

        // input features gradient and weight gradient
        if (tid < K * C_mid) {
                const int k = tid % K;
                const int c_mid = tid / K;
                scalar_t weight_grad = 0;

                // Input features gradient
                for (int c = 0; c < C_in; c++) {
                const int grad_idx = c_mid * (C_in + C_add) + c;
                const scalar_t grad_val = shared_grad[grad_idx];
                atomicAdd(&grad_input[batch_idx][neighbor_inds[batch_idx][point_idx][k]][c],
                        grad_val * weights[batch_idx][point_idx][k][c_mid]);
                weight_grad += grad_val * input[batch_idx][neighbor_inds[batch_idx][point_idx][k]][c];
                }

                // Additional features contribution to weight gradient
                for (int c = 0; c < C_add; c++) {
                const int grad_idx = c_mid * (C_in + C_add) + C_in + c;
                weight_grad += shared_grad[grad_idx] * additional_features[batch_idx][point_idx][k][c];
                }

                grad_weights[batch_idx][point_idx][k][c_mid] = weight_grad;
        }

        // Linear layer gradients
        if (tid < C_out) {
                scalar_t bias_grad = grad_output[batch_idx][point_idx][tid];
                
                // Linear weights gradient
                for (int c = 0; c < total_channels; c++) {
                atomicAdd(&grad_linear_weights[tid][c],
                        bias_grad * pconv_output[batch_idx][point_idx][c]);
                }
                
                // Linear bias gradient
                atomicAdd(&grad_linear_bias[tid], bias_grad);
        }
}

torch::Tensor pcf_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights
)
{
	const int B = input.size(0);
	const int N = input.size(1);
	const int Nout = neighbor_inds.size(1);
	const int C_in = input.size(2);
	const int C_mid =  weights.size(3);
	const int numBlocks = B * Nout;
	const int numThreads = C_mid * C_in > 256 ? 256 : C_mid * C_in;
        auto output = torch::zeros({B,Nout,C_mid*C_in}, input.type());
	AT_DISPATCH_FLOATING_TYPES(output.type(), "pcf_cuda_forward_kernel", ([&] {
	pcf_cuda_forward_kernel<scalar_t><<<numBlocks, numThreads>>>(
		input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
		neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
		guidance.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
		weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
	}));
	return output;
}

std::vector<torch::Tensor> pcf_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor guidance,
    torch::Tensor weights)
// gradient should be same shape as input
// grad_output: B x N x (C_mid * C_in)
{
    const int B = input.size(0);
    const int N = input.size(1);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int C_mid =  weights.size(3);
    const int K = neighbor_inds.size(2);
    const int numBlocks = B * Nout;
    const int numThreads = C_mid * K;
    auto grad_input = torch::zeros_like(input);
    auto grad_guidance = torch::zeros_like(guidance);
    auto grad_weights = torch::zeros_like(weights);
    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "pcf_cuda_backward_kernel", ([&] {
	pcf_cuda_backward_kernel<scalar_t><<<numBlocks,numThreads>>>(
        grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
        guidance.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_guidance.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
	}));
    return {grad_input,grad_guidance,grad_weights};
}

torch::Tensor pconv_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features
)
{
        const int B = input.size(0);
        const int N = input.size(1);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
	const int C_add = additional_features.size(3);
        const int C_mid =  weights.size(3);
        const int numBlocks = B * Nout;
        const int numThreads = C_mid * C_in > 256 ? 256 : C_mid * C_in;

        // shared memory size for input and additional features
        const int K = neighbor_inds.size(2);
        const int shared_mem_size = (K * (C_in + C_add)) * sizeof(float);

        auto output = torch::zeros({B, Nout, C_mid * (C_in + C_add)}, input.type());

        AT_DISPATCH_FLOATING_TYPES(output.type(), "pconv_cuda_forward_kernel", ([&] {
        pconv_cuda_forward_kernel<scalar_t><<<numBlocks, numThreads, shared_mem_size>>>(
                input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
		additional_features.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
        return output;
}

std::vector<torch::Tensor> pconv_linear_cuda_forward(
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features,
    torch::Tensor linear_weights,
    torch::Tensor linear_bias
)
{
        const int B = input.size(0);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
        const int K = neighbor_inds.size(2);
        const int C_mid = weights.size(3);
        const int C_add = additional_features.size(3);
        const int C_out = linear_weights.size(0);

        auto pconv_output = torch::zeros({B, Nout, C_mid * (C_in + C_add)}, input.options());
        auto final_output = torch::zeros({B, Nout, C_out}, input.options());

        const int total_work = std::max({K * C_in, K * C_add, K * C_mid, C_out});
        const int thread_count = std::min(256, nextPowerOf2(total_work));

        dim3 grid(B, Nout);

        const int shared_mem_size = (
                (K * C_in) +                    // shared_input
                (K * C_add) +                   // shared_additional  
                (K * C_mid) +                   // shared_weights
                (C_mid * (C_in + C_add))        // shared_intermediate
        ) * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES(input.type(), "pconv_linear_cuda_forward_kernel", ([&] {
                pconv_linear_cuda_forward_kernel<scalar_t><<<grid, thread_count, shared_mem_size>>>(
                input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                additional_features.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                linear_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                linear_bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                final_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                pconv_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));

        return {final_output, pconv_output};
}

std::vector<torch::Tensor> pconv_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor neighbor_inds,
    torch::Tensor weights,
    torch::Tensor additional_features)
// gradient should be same shape as input
// grad_output: B x N x (C_mid * C_in)
{
    const int B = input.size(0);
    const int N = input.size(1);
    const int Nout = neighbor_inds.size(1);
    const int C_in = input.size(2);
    const int C_mid =  weights.size(3);
    const int K = neighbor_inds.size(2);
    const int numBlocks = B * Nout;
    const int numThreads = C_mid * K;
    auto grad_input = torch::zeros_like(input);
    auto grad_weights = torch::zeros_like(weights);
    auto grad_additional = torch::zeros_like(additional_features);
    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "pconv_cuda_backward_kernel", ([&] {
        pconv_cuda_backward_kernel<scalar_t><<<numBlocks,numThreads>>>(
        grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
	additional_features.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        grad_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
	grad_additional.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        }));
    return {grad_input, grad_weights, grad_additional};
}

std::vector<torch::Tensor> pconv_linear_cuda_backward(
        torch::Tensor grad_output,
        torch::Tensor input,
        torch::Tensor neighbor_inds,
        torch::Tensor weights,
        torch::Tensor additional_features,
        torch::Tensor linear_weights,
        torch::Tensor pconv_output)
{
        const int B = input.size(0);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
        const int K = neighbor_inds.size(2);
        const int C_mid = weights.size(3);
        const int C_add = additional_features.size(3);
        const int C_out = grad_output.size(2);

        auto grad_input = torch::zeros_like(input);
        auto grad_weights = torch::zeros_like(weights);
        auto grad_additional = torch::zeros_like(additional_features);
        auto grad_linear_weights = torch::zeros_like(linear_weights);
        auto grad_linear_bias = torch::zeros({C_out}, input.options());

        const int total_work = std::max({K * C_in, K * C_add, K * C_mid, C_out});
        const int thread_count = std::min(256, nextPowerOf2(total_work));

        dim3 grid(B, Nout);

        const int shared_mem_size = (C_mid * (C_in + C_add)) * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "pconv_linear_cuda_backward_kernel", ([&] {
        pconv_linear_cuda_backward_kernel<scalar_t><<<grid, thread_count, shared_mem_size>>>(
                grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                neighbor_inds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                additional_features.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                linear_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                pconv_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                grad_input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                grad_weights.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                grad_additional.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                grad_linear_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                grad_linear_bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
        }));

        return {grad_input, grad_weights, grad_additional, grad_linear_weights, grad_linear_bias};
}


std::vector<torch::Tensor> knn_inverse_cuda_forward(
        torch::Tensor neighbor_inds,
        const int total_points) 
{
        const int B = neighbor_inds.size(0);
        const int N = neighbor_inds.size(1);
        const int K = neighbor_inds.size(2);

        // Convert input to int if its not already
        auto neighbor_inds_int = neighbor_inds.to(torch::kInt32);

        auto inv_neighbors = torch::zeros({B, N * K}, neighbor_inds_int.options());
        auto inv_k = torch::zeros({B, N * K}, neighbor_inds_int.options());
        auto inv_idx = torch::zeros({B, total_points + 1}, neighbor_inds_int.options());

        const int threads_per_block = 512;
        const int points_per_grid = 65535;
        const int num_y_grids = (N + points_per_grid - 1) / points_per_grid;

        // Process each batch separately - we do this to save memory for large point clouds
        for (int b = 0; b < B; b++) {
                auto counts = torch::zeros({total_points}, neighbor_inds_int.options());
                auto running_counts = torch::zeros({total_points}, neighbor_inds_int.options());

                // Count neighbors for this batch
                for (int grid_idx = 0; grid_idx < num_y_grids; grid_idx++) {
                        const int start_point = grid_idx * points_per_grid;
                        const int points_this_grid = std::min(points_per_grid, N - start_point);

                        const dim3 block(threads_per_block);
                        const dim3 grid(1, points_this_grid);  // only 1 batch

                        count_neighbors_kernel<<<grid, block>>>(
                                neighbor_inds_int.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
                                counts.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                total_points,
                                start_point,
                                b
                        );
                }

                // Get inv_idx for this batch
                compute_inv_idx_kernel<<<1, 1>>>(
                        counts.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                        inv_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                        total_points,
                        b
                );

                // Fill inverse mapping for this batch
                for (int grid_idx = 0; grid_idx < num_y_grids; grid_idx++) {
                        const int start_point = grid_idx * points_per_grid;
                        const int points_this_grid = std::min(points_per_grid, N - start_point);

                        const dim3 block(threads_per_block);
                        const dim3 grid(1, points_this_grid);  // only 1 batch

                        fill_inverse_kernel<<<grid, block>>>(
                                neighbor_inds_int.packed_accessor32<int,3,torch::RestrictPtrTraits>(),
                                inv_neighbors.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                inv_k.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                running_counts.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                inv_idx.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
                                total_points,
                                start_point,
                                b
                        );
                }

                // counts and running_counts are automatically freed here
        }

        return {inv_neighbors.to(neighbor_inds.dtype()), 
                inv_k.to(neighbor_inds.dtype()), 
                inv_idx.to(neighbor_inds.dtype())};
}