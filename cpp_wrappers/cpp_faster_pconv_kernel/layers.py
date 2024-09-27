import torch
from torch import nn
import torch.nn.functional as F

from util.checkpoint import CheckpointFunction
from layer_utils import PConv, FasterPConv, index_points, VI_coordinate_transform, Linear_BN, UnaryBlock


def _bn_function_factory(mlp_convs):
    # Used for the gradient checkpointing in WeightNet
    def bn_function(*inputs):
        output = inputs[0]
        for conv in mlp_convs:
            output = F.relu(conv(output), inplace=True)
        return output
    return bn_function


class WeightNet(nn.Module):
    '''
    WeightNet for PointConv. This runs a few MLP layers (defined by hidden_unit) on the 
    point coordinates and outputs generated weights for each neighbor of each point. 
    The weights will then be matrix-multiplied with the input to perform convolution

    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        hidden_unit: Number of hidden units, a list which can contain multiple hidden layers
        efficient: If set to True, then gradient checkpointing is used in training to reduce memory cost
    Input: Coordinates for all the kNN neighborhoods
           input shape is B x N x K x in_channel, B is batch size, in_channel is the dimensionality of
            the coordinates (usually 3 for 3D or 2 for 2D, 12 for VI), K is the neighborhood size,
            N is the number of points
    Output: The generated weights B x N x K x C_mid
    '''

    def __init__(
            self,
            in_channel,
            out_channel,
            hidden_unit=[8, 8],
            efficient=False):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.efficient = efficient
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(Linear_BN(in_channel, out_channel))
        else:
            self.mlp_convs.append(Linear_BN(in_channel, hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(
                    Linear_BN(hidden_unit[i - 1], hidden_unit[i]))
            self.mlp_convs.append(Linear_BN(hidden_unit[-1], out_channel))

    def real_forward(self, localized_xyz):
        # xyz : BxNxKxC
        weights = localized_xyz
        for conv in self.mlp_convs:
            weights = conv(weights)
#        if i < len(self.mlp_convs) - 1:
            weights = F.relu(weights, inplace=True)

        return weights

    def forward(self, localized_xyz):
        if self.efficient and self.training:
            # Try this so that weights have gradient
            #            weights = self.mlp_convs[0](localized_xyz)
            conv_bn_relu = _bn_function_factory(self.mlp_convs)
            dummy = torch.zeros(
                1,
                dtype=torch.float32,
                requires_grad=True,
                device=localized_xyz.device)
            args = [localized_xyz + dummy]
            if self.training:
                for conv in self.mlp_convs:
                    args += tuple(conv.bn.parameters())
                    args += tuple(conv.c.parameters())
                weights = CheckpointFunction.apply(conv_bn_relu, 1, *args)
        else:
            weights = self.real_forward(localized_xyz)
        return weights


class FasterPointConv(nn.Module):
    '''
    PointConv layer with a positional embedding concatenated to the features
    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        weightnet: Number of input/output channels for weightnet
    Input:
        dense_xyz: tensor (batch_size, num_points, 3). The coordinates of the points before subsampling (if it is a "strided" convolution wihch simultaneously subsamples the point cloud)
        dense_feats: tensor (batch_size, num_points, num_dims). The features of the points before subsampling.
        nei_inds: tensor (batch_size, num_points2, K). The neighborhood indices of the K nearest neighbors of each point (after subsampling). The indices should index into dense_xyz and dense_feats,
                  as during subsampling features at new coordinates are aggregated from the points before subsampling
        dense_xyz_norm: tensor (batch_size, num_points, 3). The surface normals of the points before subsampling
        sparse_xyz: tensor (batch_size, num_points2, 3). The coordinates of the points after subsampling (if there is no subsampling, just input None for this and the next)
        sparse_xyz_norm: tensor (batch_size, num_points2, 3). The surface normals of the points after subsampling
        vi_features: tensor (batch_size, num_points2, 12). VI features only needs to be computed once per stage. If it has been computed in a previous layer,
                     it can be saved and directly inputted here.
        Note: batch_size is usually 1 since we are using the packed representation packing multiple point clouds into one. However this dimension needs to be there for pyTorch to work properly.
    Output:
        new_feat: output features
        weightNetInput: the input to weightNet, which are relative coordinates
                        or viewpoint-invariance aware transforms of it
    '''

    def __init__(self,
                 in_channel,
                 out_channel,
                 weightnet=[9, 16],
                 use_batch_norm=True,
                 dropout_rate=0.0):
        super(FasterPointConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        # last_ch = min(out_channel // 4, 32)
        last_ch = 0

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(
                in_channel,
                out_channel // 4,
                use_bn=True,
                bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient=True)
        if use_batch_norm:
            self.linear = Linear_BN(
                (out_channel // 4 + last_ch) * weightnet[-1], out_channel // 2, bn_ver='1d')
        else:
            self.linear = nn.Linear(
                (out_channel // 4 + last_ch) * weightnet[-1], out_channel // 2)

        self.dropout = nn.Dropout(
            p=use_batch_norm) if dropout_rate > 0. else nn.Identity()

        # Second upscaling mlp
        self.unary2 = UnaryBlock(
            out_channel // 2,
            out_channel,
            use_bn=True,
            bn_momentum=0.1,
            no_relu=True)

        # Shortcut optional mpl
        if in_channel != out_channel:
            self.unary_shortcut = UnaryBlock(
                in_channel,
                out_channel,
                use_bn=True,
                bn_momentum=0.1,
                no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(
            self,
            dense_xyz,
            dense_feats,
            inv_neighbors,
            inv_k,
            inv_idx,
            nei_inds,
            dense_xyz_norm=None,
            sparse_xyz=None,
            sparse_xyz_norm=None,
            vi_features=None,
            use_vi=False,
            use_cuda_kernel=True):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3), if None, then assume sparse_xyz = dense_xyz
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, N, _ = dense_xyz.shape
        if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
        else:
            M = N
        _, _, K = nei_inds.shape

        # First downscaling mlp
        feats_x = self.unary1(dense_feats)
        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K,
        # D]
        if sparse_xyz is not None:
            localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        else:
            localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)

        if use_vi is True:
            gathered_norm = index_points(dense_xyz_norm, nei_inds)
            if vi_features is None:
                if sparse_xyz is not None:
                    weightNetInput = VI_coordinate_transform(
                        localized_xyz, gathered_norm, sparse_xyz_norm, K)
                else:
                    weightNetInput = VI_coordinate_transform(
                        localized_xyz, gathered_norm, dense_xyz_norm, K)
            else:
                weightNetInput = vi_features
        else:
            weightNetInput = localized_xyz

        # If not using CUDA kernel, then we need to sparse gather the features
        # here
        if not use_cuda_kernel:
            gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
            new_feat = gathered_feat

        weights = self.weightnet(weightNetInput)

        if use_cuda_kernel:
            feats_x = feats_x.contiguous()
            nei_inds = nei_inds.contiguous()
            weights = weights.contiguous()
            new_feat = FasterPConv.forward(
                feats_x, inv_neighbors, inv_k, inv_idx, nei_inds, weights)
        else:
            new_feat = torch.matmul(
                input=new_feat.permute(
                    0, 1, 3, 2), other=weights).view(
                B, M, -1)

        new_feat = self.linear(new_feat)
        new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)
        if sparse_xyz is not None:
            sparse_feats = torch.max(
                index_points(
                    dense_feats,
                    nei_inds),
                dim=2)[0]
        else:
            sparse_feats = dense_feats

        shortcut = self.unary_shortcut(sparse_feats)

        new_feat = self.leaky_relu(new_feat + shortcut)

        return new_feat, weightNetInput


class PointConv(nn.Module):
    '''
    PointConv layer with a positional embedding concatenated to the features
    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        weightnet: Number of input/output channels for weightnet
    Input:
        dense_xyz: tensor (batch_size, num_points, 3). The coordinates of the points before subsampling (if it is a "strided" convolution wihch simultaneously subsamples the point cloud)
        dense_feats: tensor (batch_size, num_points, num_dims). The features of the points before subsampling.
        nei_inds: tensor (batch_size, num_points2, K). The neighborhood indices of the K nearest neighbors of each point (after subsampling). The indices should index into dense_xyz and dense_feats,
                  as during subsampling features at new coordinates are aggregated from the points before subsampling
        dense_xyz_norm: tensor (batch_size, num_points, 3). The surface normals of the points before subsampling
        sparse_xyz: tensor (batch_size, num_points2, 3). The coordinates of the points after subsampling (if there is no subsampling, just input None for this and the next)
        sparse_xyz_norm: tensor (batch_size, num_points2, 3). The surface normals of the points after subsampling
        vi_features: tensor (batch_size, num_points2, 12). VI features only needs to be computed once per stage. If it has been computed in a previous layer,
                     it can be saved and directly inputted here.
        Note: batch_size is usually 1 since we are using the packed representation packing multiple point clouds into one. However this dimension needs to be there for pyTorch to work properly.
    Output:
        new_feat: output features
        weightNetInput: the input to weightNet, which are relative coordinates
                        or viewpoint-invariance aware transforms of it
    '''

    def __init__(self,
                 in_channel,
                 out_channel,
                 weightnet=[9, 16],
                 use_batch_norm=True,
                 dropout_rate=0.0):
        super(PointConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        # last_ch = min(out_channel // 4, 32)
        last_ch = 0

        # First downscaling mlp
        if in_channel != out_channel // 4:
            self.unary1 = UnaryBlock(
                in_channel,
                out_channel // 4,
                use_bn=True,
                bn_momentum=0.1)
        else:
            self.unary1 = nn.Identity()

        self.weightnet = WeightNet(weightnet[0], weightnet[1], efficient=True)
        if use_batch_norm:
            self.linear = Linear_BN(
                (out_channel // 4 + last_ch) * weightnet[-1], out_channel // 2, bn_ver='1d')
        else:
            self.linear = nn.Linear(
                (out_channel // 4 + last_ch) * weightnet[-1], out_channel // 2)

        self.dropout = nn.Dropout(
            p=use_batch_norm) if dropout_rate > 0. else nn.Identity()

        # Second upscaling mlp
        self.unary2 = UnaryBlock(
            out_channel // 2,
            out_channel,
            use_bn=True,
            bn_momentum=0.1,
            no_relu=True)

        # Shortcut optional mpl
        if in_channel != out_channel:
            self.unary_shortcut = UnaryBlock(
                in_channel,
                out_channel,
                use_bn=True,
                bn_momentum=0.1,
                no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

        return

    def forward(
            self,
            dense_xyz,
            dense_feats,
            nei_inds,
            dense_xyz_norm=None,
            sparse_xyz=None,
            sparse_xyz_norm=None,
            vi_features=None,
            use_vi=False,
            use_cuda_kernel=True):
        """
        dense_xyz: tensor (batch_size, num_points, 3)
        sparse_xyz: tensor (batch_size, num_points2, 3), if None, then assume sparse_xyz = dense_xyz
        dense_feats: tensor (batch_size, num_points, num_dims)
        nei_inds: tensor (batch_size, num_points2, K)
        """
        B, N, _ = dense_xyz.shape
        if sparse_xyz is not None:
            _, M, _ = sparse_xyz.shape
        else:
            M = N
        _, _, K = nei_inds.shape

        # First downscaling mlp
        feats_x = self.unary1(dense_feats)
        gathered_xyz = index_points(dense_xyz, nei_inds)
        # localized_xyz = gathered_xyz - sparse_xyz.view(B, M, 1, D) #[B, M, K,
        # D]
        if sparse_xyz is not None:
            localized_xyz = gathered_xyz - sparse_xyz.unsqueeze(dim=2)
        else:
            localized_xyz = gathered_xyz - dense_xyz.unsqueeze(dim=2)

        if use_vi is True:
            gathered_norm = index_points(dense_xyz_norm, nei_inds)
            if vi_features is None:
                if sparse_xyz is not None:
                    weightNetInput = VI_coordinate_transform(
                        localized_xyz, gathered_norm, sparse_xyz_norm, K)
                else:
                    weightNetInput = VI_coordinate_transform(
                        localized_xyz, gathered_norm, dense_xyz_norm, K)
            else:
                weightNetInput = vi_features
        else:
            weightNetInput = localized_xyz

        # If not using CUDA kernel, then we need to sparse gather the features
        # here
        if not use_cuda_kernel:
            gathered_feat = index_points(feats_x, nei_inds)  # [B, M, K, in_ch]
            new_feat = gathered_feat

        weights = self.weightnet(weightNetInput)

        if use_cuda_kernel:
            feats_x = feats_x.contiguous()
            nei_inds = nei_inds.contiguous()
            weights = weights.contiguous()
            new_feat = PConv.forward(feats_x, nei_inds, weights)
        else:
            new_feat = torch.matmul(
                input=new_feat.permute(
                    0, 1, 3, 2), other=weights).view(
                B, M, -1)

        new_feat = self.linear(new_feat)
        new_feat = F.relu(new_feat, inplace=True)

        # Dropout
        new_feat = self.dropout(new_feat)

        # Second upscaling mlp
        new_feat = self.unary2(new_feat)
        if sparse_xyz is not None:
            sparse_feats = torch.max(
                index_points(
                    dense_feats,
                    nei_inds),
                dim=2)[0]
        else:
            sparse_feats = dense_feats

        shortcut = self.unary_shortcut(sparse_feats)

        new_feat = self.leaky_relu(new_feat + shortcut)

        return new_feat, weightNetInput
