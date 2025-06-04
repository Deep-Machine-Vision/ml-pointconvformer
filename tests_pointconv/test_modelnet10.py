"""
This script is used to load the ModelNet10 dataset and train a PointConv encoder on it.
To download the ModelNet10 dataset, open the following link:

http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip

Unzip the file and place the ModelNet10 folder in the ./tests_pointconv directory.

Then run the script with:

python ./tests_pointconv/run_pointconv.py

This will train a PointConv encoder to classify the ModelNet10 dataset.

This file is written by @heleiduanagility and is adopted from ml-pointconvformer.

Further modified by @pvskand.
"""

import os
import numpy as np
import torch
import pcf_cuda
import time
import trimesh
import sys
import glob

import matplotlib.pyplot as plt
import torch.nn as nn

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knn_post_dataloader_utils import compute_knn_packed, prepare
from torch.utils.data import Dataset, DataLoader
from layers import PointConv
from util.voxelize import voxelize
from util.common_util import compute_knn_inverse
from dataclasses import dataclass


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ModelCfg:
    # DATASET CFG
    NUM_FEATURES = 3
    NUM_POINT_CLOUDS = 1024
    # MODEL CFG
    POINT_DIM = 3
    OUT_CHANNEL = 10  # For ModelNet10 classification
    K = 16
    USE_VI = False
    USE_PE = False
    BATCH_NORM = False
    PCONV_OPT = True
    USE_CUDA_KERNEL = True
    dropout_rate = 0.0
    droppath_rate = 0.0
    # TRAINING CFG
    batch_size = 32
    num_epochs = 200
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Downsampling configs
    NUM_DOWNSAMPLING_LAYERS = 2
    DOWNSAMPLE_FACTOR = 4
    VOXEL_SIZE = 0.05
    # KNN configs
    K_forward = [16, 16, 16, 16, 16] # NUM_DOWNSAMPLING_LAYERS + 1
    K_propagate = [16, 16, 16, 16, 16]
    K_self = [16, 16, 16, 16, 16]
    # Encoder layer configs
    ENCODER_LAYERS = [
        (3, 64),    # First layer: XYZ -> 64 features
        (64, OUT_CHANNEL),  # Second layer: 64 -> 10 features
    ]


class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024):
        print(root_dir)
        self.root_dir = os.path.join(root_dir, 'ModelNet10')
        self.classes = sorted(
            [d for d in os.listdir(self.root_dir) 
             if os.path.isdir(os.path.join(self.root_dir, d))
            ]
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_points = num_points
        self.files = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls, split)
            if os.path.exists(cls_dir):
                self.files.extend([
                    (f, cls)
                    for f in glob.glob(os.path.join(cls_dir, '*.off'))
                ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, cls = self.files[idx]
        # Load mesh without processing
        mesh = trimesh.load(file_path, force='mesh', process=False)
        
        # Sample points from the mesh
        points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        points = points.copy()  # Create a copy to avoid reference issues
        
        # Normalize data
        points = points - np.mean(points, axis=0)
        points = points / np.max(np.abs(points))
        
        label = self.class_to_idx[cls]
        return torch.FloatTensor(points), label


class PointConvEncoder(nn.Module):
    def __init__(self, cfg):
        super(PointConvEncoder, self).__init__()
        model, mlp = build_encoder(cfg)
        self.model = model
        self.mlp = mlp
        self._print_model_stats(cfg)

    def _print_model_stats(self, cfg):
        # Count and print model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / (1024 * 1024):.2f} MB (assuming float32)")
        print(f"  Device: {cfg.device}")
        print("")

    def forward(self, xyz, init_feats, nei_inds, inv_neighbors, inv_k, inv_idx):
        """
        Forward pass through PointConv encoder.
        Args:
            xyz: List of point coordinates at different resolutions
            init_feats: Initial features tensor [B, N, 3]
            nei_inds: List of KNN indices for each resolution
            inv_neighbors: List of inverse neighbor indices
            inv_k: List of inverse k values
            inv_idx: List of inverse indices
        Returns:
            Final features tensor [B, N, OUT_CHANNEL]
        """
        curr_feats = init_feats  # Start with initial features
        for i, layer in enumerate(self.model):
            # Downsample features if needed
            if i > 0 and curr_feats.shape[1] != xyz[i].shape[1]:
                # Use the same voxel indices as points
                curr_feats = curr_feats[:, :xyz[i].shape[1], :]
            # print(f"layer {i}, xyz shape: {xyz[i].shape}, feats shape: {curr_feats.shape}, "
            #     f"nei_inds shape: {nei_inds[i].shape}, inv_neighbors shape: {inv_neighbors[i].shape}, "
            #     f"inv_k shape: {inv_k[i].shape}, inv_idx shape: {inv_idx[i].shape}")
            # Process current layer
            if self.cfg.PCONV_OPT:
                next_feats, _ = layer(
                    dense_xyz=xyz[i],
                    dense_feats=curr_feats,
                    nei_inds=nei_inds[i],
                    inv_neighbors=inv_neighbors[i],
                    inv_k=inv_k[i],
                    inv_idx=inv_idx[i]
                )
            else:
                next_feats, _ = layer(
                    dense_xyz=xyz[i],
                    dense_feats=curr_feats,
                    nei_inds=nei_inds[i],
                    sparse_xyz=xyz[i+1],
                )
            # print(f"feats shape after layer {i}: {next_feats.shape}")
            curr_feats = next_feats  # Update for next layer
        return curr_feats  # (B, N, OUT_CHANNEL)


def visualize_point_cloud(points, label, classes, title="Point Cloud"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c='b', marker='.', s=10, alpha=0.8)
    ax.set_box_aspect([1,1,1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(f"{title} - Class: {classes[label]} (Points: {len(points)})", fontsize=12)
    plt.show()


def test_visualization_and_shapes(train_loader, test_loader, train_dataset, test_dataset):
    # Visualize a few train samples
    for batch_idx, (points, labels) in enumerate(train_loader):
        print(f"\nTrain Batch {batch_idx}:")
        print(f"Points shape: {points.shape}")
        print(f"Labels shape: {labels.shape}")
        visualize_point_cloud(points[0].numpy(), labels[0].item(), train_dataset.classes)
        if batch_idx >= 1:
            break

    # Visualize a few test samples
    for batch_idx, (points, labels) in enumerate(test_loader):
        print(f"\nTest Batch {batch_idx}:")
        print(f"Points shape: {points.shape}")
        print(f"Labels shape: {labels.shape}")
        visualize_point_cloud(points[0].numpy(), labels[0].item(), test_dataset.classes)
        if batch_idx >= 1:
            break

    # Simple training loop to verify data shapes
    print("\nSimple training loop (no model):")
    for epoch in range(2):
        print(f"Epoch {epoch}")
        for batch_idx, (points, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: points shape = {points.shape}, labels shape = {labels.shape}, labels  are = {labels}")
            if batch_idx >= 2:
                break


def build_encoder(cfg):
    """
        Build a PointConv encoder with `NUM_DOWNSAMPLING_LAYERS` layers.
    """
    model = nn.ModuleList()
    for i in range(cfg.NUM_DOWNSAMPLING_LAYERS):
        in_channel = cfg.OUT_CHANNEL*(4**i) if i > 0 else cfg.NUM_FEATURES
        out_channel = cfg.OUT_CHANNEL*(4**(i+1))
        model.add_module(f"pconv_{i}", 
                         PointConv(
                            in_channel=in_channel,
                            out_channel=out_channel,
                            cfg=cfg,
                            weightnet=[cfg.POINT_DIM, 16],
                            USE_VI=None
                        ))
    # mlp for classification
    mlp = nn.Sequential(
        nn.Linear(out_channel, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    return model, mlp

def prepare_inputs(cfg, points):
    """
    Prepare inputs for PointConv encoder from batched point clouds.
    Args:
        cfg: Configuration object
        points: Point cloud tensor of shape (B, N, 3)
    Returns:
        xyz: List of point coordinates at different resolutions
        init_feats: Initial features tensor [B, N, 3]
        nei_inds: List of KNN indices for each resolution
        inv_neighbors: List of inverse neighbor indices
        inv_k: List of inverse k values
        inv_idx: List of inverse indices
    """
    
    B, N, _ = points.shape
    xyz_list = [x for x in points] # converting to list of points (N, 3)
    xyz = [xyz_list]  # Original points [B, N, 3]
    init_feats = [x for x in points]  # Start with XYZ as features [B, N, 3]
    
    xyz_downsampled = [[] for _ in range(cfg.NUM_DOWNSAMPLING_LAYERS)]
    for i in range(cfg.NUM_DOWNSAMPLING_LAYERS):
        voxel_size = cfg.VOXEL_SIZE * (cfg.DOWNSAMPLE_FACTOR**i)
        
        for b in range(B):
            points_np = points[b].cpu().numpy()
            voxel_idx = voxelize(points_np, voxel_size=voxel_size, hash_type='fnv', mode='random')
            downsampled_points = torch.from_numpy(points_np[voxel_idx]).to(cfg.device)
            xyz_downsampled[i].append(downsampled_points)

    # merge xyz and xyz_downsampled
    xyz = xyz + xyz_downsampled

    # offset for each point cloud for packed representation
    offset = []
    for i in range(cfg.NUM_DOWNSAMPLING_LAYERS + 1):
        level_offset = [len(xyz[i][j]) for j in range(B)]
        offset.append(level_offset)

    # unsqueeze xyz
    unsqueezed_xyz = []
    for level in xyz:
        unsqueezed_points = [points.unsqueeze(0).to(cfg.device) for points in level]
        unsqueezed_xyz.append(torch.cat(unsqueezed_points, dim=1))
    xyz = unsqueezed_xyz

    # feats
    init_feats = torch.cat(init_feats, dim=0)
    init_feats = init_feats.unsqueeze(0).to(cfg.device)

    # Compute kNN
    nei_self_list, nei_forward_list, nei_propagate_list = compute_knn_packed(xyz, offset, cfg.K_self, cfg.K_forward, cfg.K_propagate)
    nei_self_list, nei_forward_list, nei_propagate_list = prepare(nei_self_list, nei_forward_list, nei_propagate_list)

    return xyz, init_feats, offset, nei_self_list, nei_forward_list, nei_propagate_list

if __name__ == "__main__":
    # Set config directly in Python
    cfg = ModelCfg()
    # Load train and test datasets
    train_dataset = ModelNet10Dataset(
        'tests_pointconv/ModelNet10',
        split='train',
        num_points=cfg.NUM_POINT_CLOUDS
    )
    test_dataset = ModelNet10Dataset(
        'tests_pointconv/ModelNet10',
        split='test',
        num_points=cfg.NUM_POINT_CLOUDS
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Points per sample: {train_dataset.num_points}")

    # NOTE: A test to visualize the data
    # test_visualization_and_shapes(train_loader, test_loader, train_dataset, test_dataset)

    # Initialize model
    model = PointConvEncoder(cfg)
    model = model.to(cfg.device)

    # NOTE: A test to check if the model is working
    # for batch_idx, (points, labels) in enumerate(train_loader):
    #     points = points.to(cfg.device)
    #     labels = labels.to(cfg.device)
    #     # Prepare inputs for the model
    #     xyz, init_feats, nei_inds, inv_neighbors, inv_k, inv_idx = prepare_inputs(cfg, points)
    #     # Forward pass through encoder layers
    #     output = model(xyz, init_feats, nei_inds, inv_neighbors, inv_k, inv_idx)
    #     # Global pooling and classification
    #     pooled = output.mean(dim=1)  # (B, OUT_CHANNEL)
    #     print(f"pooled shape: {pooled.shape}")
    #     break
    # exit()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    training_step = 0
    total_training_time = 0.0
    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.perf_counter()
        total_batch_time = 0.0
        for batch_idx, (points, labels) in enumerate(train_loader):
            batch_start = time.perf_counter()
            points = points.to(cfg.device)
            labels = labels.to(cfg.device)

            # Prepare inputs for the model
            xyz, init_feats, offset, nei_self_list, nei_forward_list, nei_propagate_list = prepare_inputs(cfg, points)
            inv_self, inv_forward, inv_propagate = compute_knn_inverse(xyz, nei_self_list, nei_forward_list, nei_propagate_list)
            
            # Forward pass through encoder layers
            for i, layer in enumerate(model.model):
                output, _ = layer(
                    dense_xyz=xyz[i],
                    dense_feats=init_feats,
                    nei_inds=nei_forward_list[i],
                    sparse_xyz=xyz[i+1],
                    inv_neighbors=inv_forward[0][i].int(),
                    inv_k=inv_forward[1][i].byte(),
                    inv_idx=inv_forward[2][i].int()
                )
                init_feats = output

            # check if the number of points in the output of encoder
            # is same as  the number of points in the last level.
            assert init_feats.shape[1] == sum(offset[-1])
            # Global pooling
            pooled = []
            last_level_offset = torch.cumsum(torch.tensor(offset[-1]), dim=0)
            # append 0 in the beginning to `last_level_offset`
            last_level_offset = torch.cat([torch.tensor([0]), last_level_offset], dim=0)
            for i in range(len(last_level_offset)-1):
                pooled.append(init_feats[:, last_level_offset[i]:last_level_offset[i+1], :].mean(dim=1))
            pooled = torch.cat(pooled, dim=0)  # (B, OUT_CHANNEL)

            # MLP classification
            pooled = model.mlp(pooled)

            optimizer.zero_grad()
            loss = criterion(pooled, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_time = time.perf_counter() - batch_start
            total_batch_time += batch_time
            training_step += 1
            if (batch_idx + 1) % 10 == 0 or (batch_idx == 0):
                print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_time = time.perf_counter() - epoch_start_time
        total_training_time += epoch_time
        avg_loss = running_loss / (batch_idx + 1)  # Per epoch average
        avg_batch_time = total_batch_time / (batch_idx + 1)  # Per epoch average
        overall_avg_batch_time = total_batch_time / training_step  # Overall average
        avg_epoch_time = total_training_time / (epoch + 1)
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] finished:")
        print(f"  - Average Loss: {avg_loss:.4f}")
        print(f"  - Epoch time: {epoch_time:.2f}s")
        print(f"  - Avg batch time (this epoch): {avg_batch_time:.2f}s")
        print(f"  - Overall avg batch time: {overall_avg_batch_time:.2f}s")
        print(f"  - Avg epoch time: {avg_epoch_time:.2f}s")
        print(f"  - Total training time: {total_training_time:.2f}s\n")

        # Test evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        test_start_time = time.perf_counter()
        with torch.no_grad():
            for points, labels in test_loader:
                points = points.to(cfg.device)
                labels = labels.to(cfg.device)
                # Prepare inputs for the model
                xyz, init_feats, nei_inds, inv_neighbors, inv_k, inv_idx = prepare_inputs(cfg, points)
                # Forward pass
                output = model(xyz, init_feats, nei_inds, inv_neighbors, inv_k, inv_idx)
                pooled = output.mean(dim=1)
                loss = criterion(pooled, labels)
                test_loss += loss.item() * points.size(0)
                preds = pooled.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_time = time.perf_counter() - test_start_time
        avg_test_loss = test_loss / total
        accuracy = correct / total
        print(f"Test set:")
        print(f"  - Average loss: {avg_test_loss:.4f}")
        print(f"  - Accuracy: {accuracy*100:.2f}%")
        print(f"  - Test time: {test_time:.2f}s\n")
