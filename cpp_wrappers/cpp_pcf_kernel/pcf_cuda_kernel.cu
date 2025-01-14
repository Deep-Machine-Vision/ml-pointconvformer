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
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ final_output)
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

        const int K = neighbor_inds.size(2);
        const int C_in = input.size(2);
        const int C_mid = weights.size(3);
        const int C_add = additional_features.size(3);
        const int C_out = linear_weights.size(0);
        
        const int batch_idx = blockIdx.x / gridDim.y;
        const int point_idx = blockIdx.y;
        const int tid = threadIdx.x;
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;

        // Shared memory
        scalar_t* shared_input = shared_mem;
        scalar_t* shared_additional = shared_input + (K * C_in);
        scalar_t* shared_weights = shared_additional + (K * C_add);
        scalar_t* shared_intermediate = shared_weights + (K * C_mid);

        // Load input features
        #pragma unroll
        for (int i = tid; i < K * C_in; i += blockDim.x) {
                const int k = i / C_in;
                const int c = i % C_in;
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
        const int total_channels = C_mid * (C_in + C_add);

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

// template <typename scalar_t>
// __global__ void pconv_linear_cuda_backward_kernel(
//     const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ grad_output_linear,
//     const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ pconv_output,
//     const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ input,
//     const torch::PackedTensorAccessor32<long, 3, torch::RestrictPtrTraits> __restrict__ neighbor_inds,
//     const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ weights,
//     const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ additional_features,
//     const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ linear_weights,
//     const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> __restrict__ linear_bias,
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> __restrict__ grad_input,
//     torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ grad_weights,
//     torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ grad_additional,
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ grad_linear_weights,
//     torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> __restrict__ grad_linear_bias)
// /* 
//    grad_output: B x N x (C_mid * C_in), the gradient derived from above
//    input: B x N x C_in tensor, B = batch size, N = number of points, C_in = number of channels, input features
//    neighbor_inds: B x N x K tensor, K = neighborhood size, indices of the neighborhood of each point
//    weights: B x N x K x C_mid, C_mid = number of middle channels
//    additional_features: B x N x K x C_add, additional features that do not need gather
//    grad_input: same shape as input, gradient of input
//    grad_weights: same shape as weights, gradient of weights

//    Forward is:
//     sum_{i} input[b,n][neighbor_inds[i]][j] * weights[b,n,k,i]
// */

// {
//         int i, j, k, ii, kk, iter0;
//         const int B = input.size(0);
//         const int N = input.size(1);
//         const int Nout = neighbor_inds.size(1);
//         const int C_in = input.size(2);
//         const int K = neighbor_inds.size(2);
//         const int C_mid = weights.size(3);
// 	const int C_add = additional_features.size(3);
//         const int increment = blockDim.x / C_mid;
//         const int cur_mid = threadIdx.x / increment;
//         const int C_out = linear_weights.size(0);

//         extern __shared__ unsigned char memory[];
//         scalar_t* shared_grad_output_pconv = reinterpret_cast<scalar_t*>(memory);

//         if (cur_mid >= C_mid)
//                 return;

//         // Compute grad_output_pconv from grad_output_linear
//         for (iter0 = blockIdx.x; iter0 < B * Nout; iter0 += gridDim.x) {
//                 ii = iter0 / Nout;  // Batch index
//                 i = iter0 % Nout;   // Point index
//                 k = threadIdx.x % K;

//                 scalar_t grad_temp = 0.0;

//                 for (kk = 0; kk < C_out; kk++) {
//                         grad_temp += grad_output_linear[ii][i][kk] * linear_weights[kk][cur_mid];
//                         atomicAdd(&grad_linear_weights[kk][cur_mid], grad_output_linear[ii][i][kk] * pconv_output[ii][i][cur_mid]);
//                 }

//                 // grad_output_pconv[ii][i][cur_mid] = grad_temp;
//                 shared_grad_output_pconv[threadIdx.x] = grad_temp;
//         }

//         __syncthreads();

//         // Supposedly blockIdx.x should go up to B * N
//         for (iter0 = blockIdx.x; iter0 < B * Nout; iter0 += gridDim.x) {
//                 // ii is the index on the batch dimension
//                 ii = iter0 / Nout;
//                 // i is the index on the point dimension
//                 i = iter0 % Nout;
//                 k = threadIdx.x % K;
//                 scalar_t weight_grad_temp = 0.0;
//                 scalar_t cur_compute;

//                 #pragma unroll
//                 for (kk = 0; kk < C_in; kk++) {
//                         long cur_channel = cur_mid * (C_in + C_add) + kk;
//                         cur_compute = shared_grad_output_pconv[threadIdx.x] * weights[ii][i][k][cur_mid];
//                         weight_grad_temp += shared_grad_output_pconv[threadIdx.x] * input[ii][neighbor_inds[ii][i][k]][kk];
//                 // It would be quite expensive to store this in shared memory so use this naive approach for now, using atomicAdd to avoid racing conditions
//                         atomicAdd(&grad_input[ii][neighbor_inds[ii][i][k]][kk], cur_compute);
//                 }

// 		for (kk = 0; kk < C_add; kk++) {
// 			long cur_channel = cur_mid * (C_in + C_add) + kk + C_in;
// 			cur_compute = shared_grad_output_pconv[threadIdx.x] * weights[ii][i][k][cur_mid];
// 			weight_grad_temp += shared_grad_output_pconv[threadIdx.x] * additional_features[ii][i][k][kk];
// 			grad_additional[ii][i][k][kk] = cur_compute;
// 		}
//                 grad_weights[ii][i][k][cur_mid] = weight_grad_temp;

//                 __syncthreads();
// 	}
// }

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

        // Calculate shared memory size for input and additional features
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

torch::Tensor pconv_linear_cuda_forward(
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

        auto output = torch::zeros({B, Nout, C_out}, 
                                input.options());

        // Optimal thread count
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
                output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));

        return output;
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

// std::vector<torch::Tensor> pconv_linear_cuda_backward(
//     torch::Tensor grad_output,           // B x N x (C_mid * C_out)
//     torch::Tensor input,                 // B x N x C_in
//     torch::Tensor neighbor_inds,         // B x N x K
//     torch::Tensor weights,               // B x N x K x C_mid
//     torch::Tensor additional_features,   // B x N x K x C_add
//     torch::Tensor linear_weights,        // C_out x C_mid
//     torch::Tensor linear_bias,
//     torch::Tensor pconv_output)          // B x N x C_mid (from forward pass)
// {
//     const int B = input.size(0);
//     const int N = input.size(1);
//     const int Nout = neighbor_inds.size(1);
//     const int C_in = input.size(2);
//     const int C_mid = weights.size(3);
//     const int C_out = linear_weights.size(0);
//     const int K = neighbor_inds.size(2);

//     const int numBlocks = B * Nout;             // Blocks for each batch-point combination
//     const int numThreads = C_mid * K;           // Threads for each mid-channel and neighbor
//     const size_t shared_mem_size = sizeof(float) * C_mid; // Shared memory size per block

//     auto grad_input = torch::zeros_like(input);
//     auto grad_weights = torch::zeros_like(weights);
//     auto grad_additional = torch::zeros_like(additional_features);
//     auto grad_linear_weights = torch::zeros_like(linear_weights);
//     auto grad_linear_bias = torch::zeros_like(linear_bias);

//     AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "pconv_linear_cuda_backward_kernel", ([&] {
//         pconv_linear_cuda_backward_kernel<scalar_t><<<numBlocks, numThreads, shared_mem_size>>>(
//             grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
//             pconv_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
//             input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
//             neighbor_inds.packed_accessor32<long, 3, torch::RestrictPtrTraits>(),
//             weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
//             additional_features.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
//             linear_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
//             linear_bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
//             grad_input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
//             grad_weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
//             grad_additional.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
//             grad_linear_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
//             grad_linear_bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
//     }));

//     return {grad_input, grad_weights, grad_additional, grad_linear_weights, grad_linear_bias};
// }
