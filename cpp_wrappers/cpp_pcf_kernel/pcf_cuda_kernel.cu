//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2022-2023 Apple Inc. All Rights Reserved.
//
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <vector>

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

        int i, k, ii, jj, kk, iter0;
        const int B = input.size(0);
        const int N = input.size(1);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
        const int K = neighbor_inds.size(2);
        const int C_mid = weights.size(3);
	const int C_add = additional_features.size(3);
        const int increment = blockDim.x / C_mid;

        scalar_t* shared_input = shared_mem; // Shared memory for input
        scalar_t* shared_additional = shared_mem + K * C_in; // Shared memory for additional_features

        /* parallelize ii and i on blocks */
        // Supposedly blockIdx.x should go up to B * N
        for (iter0 = blockIdx.x; iter0 < B * Nout; iter0 += gridDim.x) {
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

                // Load input data into shared memory
                for (k = threadIdx.x; k < K * C_in; k += blockDim.x) {
                        int k_idx = k / C_in;
                        int c_idx = k % C_in;
                        shared_input[k] = input[ii][neighbor_inds[ii][i][k_idx]][c_idx];
                }

                // Load additional features into shared memory
                if (C_add > 0) {
                        for (k = threadIdx.x; k < K * C_add; k += blockDim.x) {
                                int k_idx = k / C_add;
                                int c_idx = k % C_add;
                                shared_additional[k] = additional_features[ii][i][k_idx][c_idx];
                        }
                }

                __syncthreads(); // Synchronize threads after loading shared memory

                // This is our main non-contiguous memory access because we need to sparse gather the input
                // But maybe we coalesce memory across all the threads in the block?
                scalar_t partial_sum_1 = 0.0;
                for(kk = threadIdx.x % increment; kk < C_in; kk += increment) {
                        #pragma unroll
                        for (k = 0; k < K; k++) {
                                partial_sum_1 += shared_input[k * C_in + kk] * weights[ii][i][k][jj];
                        }
                        output[ii][i][jj + kk * C_mid] = partial_sum_1;
                        partial_sum_1 = 0.0;
                }
                if (C_add > 0) {
                        scalar_t partial_sum_2 = 0.0;
                        for(kk = threadIdx.x % increment; kk < C_add; kk += increment) {
                                #pragma unroll
                                for (k = 0; k < K; k++) {
                                        partial_sum_2 += shared_additional[k * C_add + kk] * weights[ii][i][k][jj];
                                }
                                output[ii][i][jj + (kk + C_in) * C_mid] = partial_sum_2;
                                partial_sum_2 = 0.0;
                        }
                }
                // At this point we would have fully populated the interm array of C_mid * C_in

                __syncthreads();
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

        int i, k, ii, jj, kk, iter0;
        const int B = input.size(0);
        const int N = input.size(1);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
        const int K = neighbor_inds.size(2);
        const int C_mid = weights.size(3);
	const int C_add = additional_features.size(3);
        const int C_out = linear_weights.size(0);
        const int increment = blockDim.x / C_mid;

        // Shared memory
        scalar_t* shared_input = shared_mem;                            // input
        scalar_t* shared_additional = shared_mem + K * C_in;            // additional_features
        scalar_t* shared_fused_output = shared_additional + K * C_add;  // fused output

        /* parallelize ii and i on blocks */
        // Supposedly blockIdx.x should go up to B * N
        for (iter0 = blockIdx.x; iter0 < B * Nout; iter0 += gridDim.x) {
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

                // Load input data into shared memory
                for (k = threadIdx.x; k < K * C_in; k += blockDim.x) {
                        int k_idx = k / C_in;
                        int c_idx = k % C_in;
                        shared_input[k] = input[ii][neighbor_inds[ii][i][k_idx]][c_idx];
                }

                // Load additional features into shared memory
                if (C_add > 0) {
                        for (k = threadIdx.x; k < K * C_add; k += blockDim.x) {
                                int k_idx = k / C_add;
                                int c_idx = k % C_add;
                                shared_additional[k] = additional_features[ii][i][k_idx][c_idx];
                        }
                }

                for (int c = threadIdx.x; c < C_mid * (C_in + C_add); c += blockDim.x) {
                        shared_fused_output[c] = 0.0;
                }

                __syncthreads(); // Sync threads after loading into shared memory

                // PConv
                for (kk = threadIdx.x % increment; kk < C_in; kk += increment) {
                        #pragma unroll
                        for (k = 0; k < K; k++) {
                                shared_fused_output[jj + kk * C_mid] += shared_input[k * C_in + kk] * weights[ii][i][k][jj];
                        }
                }

                if (C_add > 0) {
                        for (kk = threadIdx.x % increment; kk < C_add; kk += increment) {
                                #pragma unroll
                                for (k = 0; k < K; k++) {
                                        shared_fused_output[jj + (kk + C_in) * C_mid] += shared_additional[k * C_add + kk] * weights[ii][i][k][jj];
                                }
                        }
                }

                __syncthreads(); // Sync threads before applying linear layer

                // Linear
                for (int c_out = threadIdx.x; c_out < C_out; c_out += blockDim.x) {
                        scalar_t linear_sum = 0.0;
                        for (int c_in = 0; c_in < C_mid * (C_in + C_add); c_in++) {
                                linear_sum += shared_fused_output[c_in] * linear_weights[c_out][c_in];
                        }
                        final_output[ii][i][c_out] = linear_sum + linear_bias[c_out];
                }

                __syncthreads();
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
        const int N = input.size(1);
        const int Nout = neighbor_inds.size(1);
        const int C_in = input.size(2);
	const int C_add = additional_features.size(3);
        const int C_mid =  weights.size(3);
        const int C_out = linear_weights.size(0);
        const int numBlocks = B * Nout;
        const int numThreads = C_mid * C_in > 256 ? 256 : C_mid * C_in;

        // Calculate shared memory size for input, additional features and intermediates
        const int K = neighbor_inds.size(2);
        // const int shared_mem_size = (K * (C_in + C_add)) * sizeof(float);
        const int shared_mem_size = (K * C_in + K * C_add + C_mid * (C_in + C_add)) * sizeof(float);

        auto output = torch::zeros({B, N, C_out}, input.type());

        AT_DISPATCH_FLOATING_TYPES(output.type(), "pconv_linear_cuda_forward_kernel", ([&] {
        pconv_linear_cuda_forward_kernel<scalar_t><<<numBlocks, numThreads, shared_mem_size>>>(
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
    return {grad_input,grad_weights,grad_additional};
}

