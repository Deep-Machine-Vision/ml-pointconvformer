# Example code using faster PConv cuda kernerl.
import torch

import faster_pconv_cuda


class FasterPConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feat, neighbor_inds, guidance, weightnet):
        # Make sure we are not computing gradient on neighbor_inds
        neighbor_inds.requires_grad = False
        output = faster_pconv_cuda.forward(input_feat, neighbor_inds, guidance, weightnet)
        ctx.save_for_backward({input_feat, neighbor_inds, guidance, weightnet})
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_guidance, grad_weight = faster_pconv_cuda.backward(grad_output.contiguous(), *ctx.saved_tensors)
        return grad_input, None, grad_guidance, grad_weight


class FasterPConv(torch.nn.Module):
    def __init__(self):
        super(FasterPConv, self).__init__()

    def forward(self, input_features, neighbor_inds, guidance, weightnet):
        return FasterPConvFunction.apply(input_features, neighbor_inds, guidance, weightnet)