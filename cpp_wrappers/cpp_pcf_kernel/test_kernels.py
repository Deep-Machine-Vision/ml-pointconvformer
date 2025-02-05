import time
import torch
import pcf_cuda
import torch.cuda.profiler as profiler


class PConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, weightnet, additional_features):
        neighbor_inds.requires_grad = False
        output = pcf_cuda.pconv_forward(input_feat, neighbor_inds, weightnet, additional_features)
        ctx.save_for_backward(input_feat, neighbor_inds, weightnet, additional_features)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_additional = pcf_cuda.pconv_backward(
            grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, grad_weight, grad_additional

class PConvLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, weightnet, additional_features, linear_weights, linear_bias):
        neighbor_inds.requires_grad = False
        output, pconv_output = pcf_cuda.pconv_linear_forward(input_feat, neighbor_inds, weightnet, additional_features, 
                                             linear_weights, linear_bias)
        
        ctx.save_for_backward(input_feat, neighbor_inds, weightnet, additional_features, 
                            linear_weights, pconv_output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        input_feat, neighbor_inds, weightnet, additional_features, linear_weights, pconv_output = saved
        
        grads = pcf_cuda.pconv_linear_backward(
            grad_output.contiguous(), input_feat, neighbor_inds, weightnet,
            additional_features, linear_weights, pconv_output)
        
        return grads[0], None, grads[1], grads[2], grads[3], grads[4]

class PConv(torch.nn.Module):
    def __init__(self):
        super(PConv, self).__init__()

    @staticmethod
    def forward(input_features, neighbor_inds, weightnet, additional_features=None):
        if additional_features is None:
            additional_features = torch.zeros(input_features.shape[0], input_features.shape[1], 
                                           neighbor_inds.shape[2], 0, device=input_features.device)
        return PConvFunction.apply(input_features, neighbor_inds, weightnet, additional_features)

class PConvLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PConvLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, input_features, neighbor_inds, weightnet, additional_features=None):
        if additional_features is None:
            additional_features = torch.zeros(input_features.shape[0], input_features.shape[1], 
                                           neighbor_inds.shape[2], 0, device=input_features.device)
        return PConvLinearFunction.apply(input_features, neighbor_inds, weightnet, additional_features,
                                       self.linear.weight, self.linear.bias)


def test_pconv_linear():
    path = "/nfs/stak/users/sivakuml/hpc-memory/cutlass/data/500000pts"
    device = torch.device("cuda")
    
    feats_x = torch.load(f"{path}/feat_x.pt").to(device).contiguous()
    nei_inds = torch.load(f"{path}/nei_inds.pt").to(device).contiguous()
    weights = torch.load(f"{path}/weights.pt").to(device).contiguous()
    feat_pe = torch.load(f"{path}/feat_pe.pt").to(device).contiguous()
    linear_weights = torch.load(f"{path}/linear_weights.pt").to(device).contiguous()
    linear_bias = torch.load(f"{path}/linear_bias.pt").to(device).contiguous()

    # Params
    out_channel = 64
    last_ch = 16
    weightnet_dim = 16

    pconv = PConv()
    linear_layer = torch.nn.Linear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2).to(device)
    linear_layer.weight.data.copy_(linear_weights)
    linear_layer.bias.data.copy_(linear_bias)

    pconv_linear = PConvLinear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2)
    pconv_linear.linear.weight.data.copy_(linear_weights)
    pconv_linear.linear.bias.data.copy_(linear_bias)
    pconv_linear = pconv_linear.cuda()

    # Create dummy gradient for backward pass
    grad_output = torch.randn_like(linear_layer(torch.empty(feats_x.shape[0], feats_x.shape[1], 
                                              (out_channel // 4 + last_ch) * weightnet_dim, device=device)))

    # Forward & Backward (Unfused)
    pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
    out_unfused = linear_layer(pconv_out)
    out_unfused.backward(grad_output)

    # Store unfused gradients
    unfused_grads = {
        "input": feats_x.grad.clone(),
        "weightnet": weights.grad.clone(),
        "additional": feat_pe.grad.clone(),
        "linear_weight": linear_layer.weight.grad.clone(),
        "linear_bias": linear_layer.bias.grad.clone()
    }

    # Clear gradients
    feats_x.grad = None
    weights.grad = None
    feat_pe.grad = None
    linear_layer.weight.grad = None
    linear_layer.bias.grad = None

    # Forward & Backward (Fused)
    out_fused = pconv_linear(feats_x, nei_inds, weights, feat_pe)
    out_fused.backward(grad_output)

    torch.cuda.synchronize()

    # Store fused gradients
    fused_grads = {
        "input": feats_x.grad.clone(),
        "weightnet": weights.grad.clone(),
        "additional": feat_pe.grad.clone(),
        "linear_weight": pconv_linear.linear.weight.grad.clone(),
        "linear_bias": pconv_linear.linear.bias.grad.clone()
    }
    
    # Compare outputs
    print("\nForward output max difference:", 
          torch.max(torch.abs(out_unfused - out_fused)).item())
    
    # Compare gradients
    print("\nGradient differences (max absolute error):")
    for key in unfused_grads:
        diff = torch.max(torch.abs(unfused_grads[key] - fused_grads[key])).item()
        print(f"{key:13} grad diff: {diff:.6f}")
        print(f"{key:13} stats:")
        print(f"  Unfused - min: {unfused_grads[key].min().item():.6f}, max: {unfused_grads[key].max().item():.6f}, mean: {unfused_grads[key].mean().item():.6f}")
        print(f"  Fused   - min: {fused_grads[key].min().item():.6f}, max: {fused_grads[key].max().item():.6f}, mean: {fused_grads[key].mean().item():.6f}")
        print()
    
    # Timing comparison
    num_runs = 100
    torch.cuda.synchronize()
    
    # Time unfused
    total_time = 0
    for _ in range(num_runs):
        pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
        out_unfused = linear_layer(pconv_out)
        
        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        linear_layer.weight.grad = None
        linear_layer.bias.grad = None
        
        torch.cuda.synchronize()
        start = time.time()
        out_unfused.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        total_time += (time.time() - start)
    print(f"\n(Unfused) Average Time: {(total_time / num_runs) * 1000} ms")
    
    # Time fused
    total_time = 0
    for _ in range(num_runs):
        out_fused = pconv_linear(feats_x, nei_inds, weights, feat_pe)
        
        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        pconv_linear.linear.weight.grad = None
        pconv_linear.linear.bias.grad = None
        
        torch.cuda.synchronize()
        start = time.time()
        out_fused.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()
        total_time += (time.time() - start)
    print(f"(Fused) Average Time: {(total_time / num_runs) * 1000} ms")


def test_pconv_linear_with_memory(num_runs=100):
    path = "/nfs/stak/users/sivakuml/hpc-memory/cutlass/data/500000pts"
    device = torch.device("cuda")

    # Load data
    feats_x = torch.load(f"{path}/feat_x.pt").to(device).contiguous()
    nei_inds = torch.load(f"{path}/nei_inds.pt").to(device).contiguous()
    weights = torch.load(f"{path}/weights.pt").to(device).contiguous()
    feat_pe = torch.load(f"{path}/feat_pe.pt").to(device).contiguous()
    linear_weights = torch.load(f"{path}/linear_weights.pt").to(device).contiguous()
    linear_bias = torch.load(f"{path}/linear_bias.pt").to(device).contiguous()

    # Params
    out_channel = 64
    last_ch = 16
    weightnet_dim = 16

    unfused_stats = {
        'after_pconv': [],
        'after_linear': [],
        'peak_memory': []
    }

    fused_stats = {
        'after_forward': [],
        'peak_memory': []
    }

    # Initialize models
    pconv = PConv()
    linear_layer = torch.nn.Linear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2).to(device)
    linear_layer.weight.data.copy_(linear_weights)
    linear_layer.bias.data.copy_(linear_bias)

    pconv_linear = PConvLinear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2)
    pconv_linear.linear.weight.data.copy_(linear_weights)
    pconv_linear.linear.bias.data.copy_(linear_bias)
    pconv_linear = pconv_linear.cuda()

    # Dummy gradient for backward pass
    grad_output = torch.randn_like(linear_layer(torch.empty(feats_x.shape[0], feats_x.shape[1], 
                                              (out_channel // 4 + last_ch) * weightnet_dim, device=device)))

    print(f"\nRunning memory test for {num_runs} iterations...")

    for i in range(num_runs):
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} runs...")

        # Test Unfused Version
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Forward Pass (Unfused)
        pconv_out = pconv(feats_x, nei_inds, weights, feat_pe)
        torch.cuda.synchronize()
        unfused_stats['after_pconv'].append(torch.cuda.memory_allocated() - mem_before)

        out_unfused = linear_layer(pconv_out)
        torch.cuda.synchronize()
        unfused_stats['after_linear'].append(torch.cuda.memory_allocated() - mem_before)

        # Backward Pass (Unfused)
        out_unfused.backward(grad_output)
        torch.cuda.synchronize()
        unfused_stats['peak_memory'].append(torch.cuda.max_memory_allocated())

        # Clear memory
        del pconv_out, out_unfused
        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        linear_layer.weight.grad = None
        linear_layer.bias.grad = None
        torch.cuda.empty_cache()

        # Test Fused Version
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Forward & Backward (Fused)
        out_fused = pconv_linear(feats_x, nei_inds, weights, feat_pe)
        torch.cuda.synchronize()
        fused_stats['after_forward'].append(torch.cuda.memory_allocated() - mem_before)

        out_fused.backward(grad_output)
        torch.cuda.synchronize()
        fused_stats['peak_memory'].append(torch.cuda.max_memory_allocated())

        # Clear memory
        del out_fused
        feats_x.grad = None
        weights.grad = None
        feat_pe.grad = None
        pconv_linear.linear.weight.grad = None
        pconv_linear.linear.bias.grad = None
        torch.cuda.empty_cache()

    def calculate_stats(values):
        values = torch.tensor(values, dtype=torch.float64)
        return {
            'mean': values.mean().item(),
            'std': values.std().item(),
            'min': values.min().item(),
            'max': values.max().item()
        }

    print("\nUnfused Version Statistics (in MB):")
    for key, values in unfused_stats.items():
        stats = calculate_stats(values)
        print(f"\n{key}:")
        print(f"  Mean: {stats['mean'] / 1024**2:.2f} ± {stats['std'] / 1024**2:.2f}")
        print(f"  Min: {stats['min'] / 1024**2:.2f}")
        print(f"  Max: {stats['max'] / 1024**2:.2f}")

    print("\nFused Version Statistics (in MB):")
    for key, values in fused_stats.items():
        stats = calculate_stats(values)
        print(f"\n{key}:")
        print(f"  Mean: {stats['mean'] / 1024**2:.2f} ± {stats['std'] / 1024**2:.2f}")
        print(f"  Min: {stats['min'] / 1024**2:.2f}")
        print(f"  Max: {stats['max'] / 1024**2:.2f}")

    # Calculate average memory savings
    avg_memory_saved = (torch.tensor(unfused_stats['peak_memory'], dtype=torch.float64).mean() - 
                       torch.tensor(fused_stats['peak_memory'], dtype=torch.float64).mean())
    print(f"\nAverage Memory Saved by Fusion: {avg_memory_saved / 1024**2:.2f} MB")

    print("\nTensor Sizes:")
    print(f"Input features: {feats_x.shape}")
    print(f"Neighbor indices: {nei_inds.shape}")
    print(f"Weights: {weights.shape}")
    print(f"Additional features: {feat_pe.shape}")
    print(f"Linear weights: {linear_weights.shape}")


if __name__ == "__main__":
    test_pconv_linear()
    test_pconv_linear_with_memory()
