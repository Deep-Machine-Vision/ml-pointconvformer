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



feats_x = torch.load("data/feat_x.pt").to(device=torch.device("cuda")).contiguous()
nei_inds = torch.load("data/nei_inds.pt").to(device=torch.device("cuda")).contiguous()
weights = torch.load("data/weights.pt").to(device=torch.device("cuda")).contiguous()
feat_pe = torch.load("data/feat_pe.pt").to(device=torch.device("cuda")).contiguous()
linear_weights = torch.load("data/linear_weights.pt").to(device=torch.device("cuda"))
linear_bias = torch.load("data/linear_bias.pt").to(device=torch.device("cuda"))

out_channel = 64
last_ch = 16
weightnet_dim = 16
linear_layer = torch.nn.Linear((out_channel // 4 + last_ch) * weightnet_dim, out_channel // 2).to(device=torch.device("cuda"))
linear_layer.weight.data.copy_(linear_weights)
linear_layer.bias.data.copy_(linear_bias)

torch.cuda.synchronize()

num_runs = 100
total_time = 0
for num_run in range(num_runs):
    start = time.time()
    output = pcf_cuda.pconv_forward(feats_x, nei_inds, weights, feat_pe)
    output = linear_layer(output)
    total_time += (time.time() - start)

average_time = (total_time / num_runs) * 1000
print(f"Average Time: {average_time} ms")


total_time = 0
for num_run in range(num_runs):
    start = time.time()
    output = pcf_cuda.pconv_linear_forward(feats_x, nei_inds, weights, feat_pe, linear_weights, linear_bias)
    total_time += (time.time() - start)

average_time = (total_time / num_runs) * 1000
print(f"(Fused) PConv + Linear -> Average Time: {average_time} ms")



#############
# Check Equal
#############
out_1 = pcf_cuda.pconv_forward(feats_x, nei_inds, weights, feat_pe)
out_1 = linear_layer(out_1)

out_2 = pcf_cuda.pconv_linear_forward(feats_x, nei_inds, weights, feat_pe, linear_weights, linear_bias)

print(f"Two Tensors are Equal? {torch.allclose(out_1, out_2, atol=1e-5, rtol=1e-5)}")