import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import numpy as np

# ==================== 1. 设备设置 ====================
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")


def Chamfer_loss(star_mesh, scan_mesh, num_samples=30000):
    """计算Chamfer距离"""
    # 在mesh上采样点云
    star_mesh_sample_points = sample_points_from_meshes(star_mesh, num_samples)
    scan_mesh_sample_points = sample_points_from_meshes(scan_mesh, num_samples)

    # 计算Chamfer距离
    chamfer_dist, _ = chamfer_distance(star_mesh_sample_points, scan_mesh_sample_points)
    return chamfer_dist


def Shape_reg_loss(betas):
    """体型正则化损失"""
    return torch.mean(betas * betas)


def Pose_reg_loss(poses: torch.Tensor,A_pose_tensor):
    # 将姿态约束在你设定的A_pose附近
    # 也就是说：允许微调，但不允许偏离太远

    loss = torch.mean((poses - A_pose_tensor) ** 2)
    return loss
