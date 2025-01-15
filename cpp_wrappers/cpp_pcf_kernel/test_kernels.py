import time
import torch
import pcf_cuda
import torch.cuda.profiler as profiler


# class PConvFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(
#             ctx,
#             input_feat,
#             neighbor_inds,
#             weightnet,
#             additional_features):
#         neighbor_inds.requires_grad = False
#         output = pcf_cuda.pconv_forward(
#             input_feat, neighbor_inds, weightnet, additional_features)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input, grad_weight, grad_additional = pcf_cuda.pconv_backward(
#             grad_output.contiguous(), *ctx.saved_tensors)
#         return grad_input, None, grad_weight, grad_additional


# class PConv(torch.nn.Module):
#     def __init__(self):
#         super(PConv, self).__init__()

#     @staticmethod
#     def forward(input_features, neighbor_inds, weightnet, additional_features=None):
#         if additional_features is None:
#             additional_features = torch.zeros(input_features.shape[0], input_features.shape[1], neighbor_inds.shape[2], 0)
#         return PConvFunction.apply(
#             input_features,
#             neighbor_inds,
#             weightnet,
#             additional_features)


# class PConvFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(
#             ctx,
#             input_feat,
#             neighbor_inds,
#             weightnet,
#             additional_features,
#             linear_weights,
#             linear_bias):
#         neighbor_inds.requires_grad = False
#         output = pcf_cuda.pconv_forward(
#             input_feat, neighbor_inds, weightnet, additional_features, linear_weights, linear_bias)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input, grad_weight, grad_additional = pcf_cuda.pconv_backward(
#             grad_output.contiguous(), *ctx.saved_tensors)
#         return grad_input, None, grad_weight, grad_additional


# class PConv(torch.nn.Module):
#     def __init__(self):
#         super(PConv, self).__init__()

#     @staticmethod
#     def forward(input_features, neighbor_inds, weightnet, linear_weights, linear_bias, additional_features=None):
#         if additional_features is None:
#             additional_features = torch.zeros(input_features.shape[0], input_features.shape[1], neighbor_inds.shape[2], 0)
#         return PConvFunction.apply(
#             input_features,
#             neighbor_inds,
#             weightnet,
#             additional_features,
#             linear_weights,
#             linear_bias)


# path = "/nfs/stak/users/sivakuml/hpc-memory/cutlass/data/500000pts"
# feats_x = torch.load(f"{path}/feat_x.pt").to(device=torch.device("cuda")).contiguous()
# nei_inds = torch.load(f"{path}/nei_inds.pt").to(device=torch.device("cuda")).contiguous()
# weights = torch.load(f"{path}/weights.pt").to(device=torch.device("cuda")).contiguous()
# feat_pe = torch.load(f"{path}/feat_pe.pt").to(device=torch.device("cuda")).contiguous()
# linear_weights = torch.load(f"{path}/linear_weights.pt").to(device=torch.device("cuda")).contiguous()
# linear_bias = torch.load(f"{path}/linear_bias.pt").to(device=torch.device("cuda")).contiguous()

# out_channel = 64
# last_ch = 16
# weightnet_dim = 16
# linear_layer = torch.nn.Linear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2).to(device=torch.device("cuda"))
# linear_layer.weight.data.copy_(linear_weights)
# linear_layer.bias.data.copy_(linear_bias)

# torch.cuda.synchronize()

# print("input: ", feats_x.shape)
# print("neighbor_inds: ", nei_inds.shape)
# print("weights: ", weights.shape)
# print("additional_features: ", feat_pe.shape)
# print("linear_weights: ", linear_weights.shape)
# print("linear_bias: ", linear_bias.shape)

# num_runs = 100
# total_time = 0
# for num_run in range(num_runs):
#     start = time.time()
#     output = pcf_cuda.pconv_forward(feats_x, nei_inds, weights, feat_pe)
#     output = linear_layer(output)
#     torch.cuda.synchronize()
#     total_time += (time.time() - start)

# average_time = (total_time / num_runs) * 1000
# print(f"(Unfused) Average Time: {average_time} ms")

# torch.cuda.synchronize()

# total_time = 0
# for num_run in range(num_runs):
#     start = time.time()
#     output = pcf_cuda.pconv_linear_forward(feats_x, nei_inds, weights, feat_pe, linear_weights, linear_bias)
#     torch.cuda.synchronize()
#     total_time += (time.time() - start)

# average_time = (total_time / num_runs) * 1000
# print(f"(Fused) PConv + Linear -> Average Time: {average_time} ms")



# # ############
# # # Check Equal
# # ############
# out_1 = pcf_cuda.pconv_forward(feats_x, nei_inds, weights, feat_pe)
# out_1 = linear_layer(out_1)
# torch.cuda.synchronize()
# print(out_1)

# out_2 = pcf_cuda.pconv_linear_forward(feats_x, nei_inds, weights, feat_pe, linear_weights, linear_bias)
# torch.cuda.synchronize()
# print(out_2)

# # print(out_2.shape)
# print(f"Two Tensors are Equal? {torch.allclose(out_1, out_2, atol=1e-5, rtol=1e-5)}")


#############
# Profile
#############
# with torch.autograd.profiler.emit_nvtx():
#     profiler.start()
#     out_1 = pcf_cuda.pconv_forward(feats_x, nei_inds, weights, feat_pe)
#     out_1 = linear_layer(out_1)
#     # out_2 = pcf_cuda.pconv_linear_forward(feats_x, nei_inds, weights, feat_pe, linear_weights, linear_bias)
#     profiler.stop()



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
        # Store pconv output for backward (have to change this so that fused kernel returns the pconv output too)
        pconv_output = pcf_cuda.pconv_forward(input_feat, neighbor_inds, weightnet, additional_features)
        output = pcf_cuda.pconv_linear_forward(input_feat, neighbor_inds, weightnet, additional_features, 
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

    # torch.cuda.synchronize()

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


test_pconv_linear()
