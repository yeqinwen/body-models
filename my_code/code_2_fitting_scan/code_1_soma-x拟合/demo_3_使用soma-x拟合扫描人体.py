import polyscope as ps
import trimesh
import numpy as np
import torch
# from body_models.soma.numpy import SOMA
from body_models.soma.torch import SOMA
from torch.optim import LBFGS
from scipy import spatial


model = SOMA()
params = model.get_apose(batch_dims=(1,))
vertices = model.forward_vertices(**params)

star_v = vertices[0].cpu().numpy()
star_f = model.faces

scan_path = "dhu_scan_1_sc.obj"
mesh = trimesh.load_mesh(scan_path)
scan_v, scan_f = mesh.vertices, mesh.faces
scan_v = scan_v - scan_v.mean(axis=0) + star_v.mean(axis=0)  # 中心对齐
scan_v[:, 1] = scan_v[:, 1] - np.min(scan_v[:, 1]) + np.min(star_v[:, 1])  # 底部对齐

ps.init()
ps.register_surface_mesh("star", star_v, star_f,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.3,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

ps.register_surface_mesh("scan", scan_v, scan_f,
                         # color=[0, 91 / 255, 255 / 255],
                         edge_width=0.3,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

ps.set_view_projection_mode("perspective")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("y_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('z_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.show()


# 3、拟合扫描人体
# 3.1、定义损失函数
def compute_data_loss0(deformed_v: torch.Tensor, scan_v: np.ndarray, direction: str = 'star_to_scan') -> torch.Tensor:
    """
    计算单向 Chamfer 距离，方向由 direction 参数控制。

    Args:
        deformed_v: [V, 3]，当前 STAR 顶点（GPU tensor）
        scan_v: [N, 3]，扫描点云（CPU numpy array）
        direction: 'star_to_scan' 或 'scan_to_star'，指定距离计算方向

    Returns:
        标量损失（已平均）
    """
    if direction == 'star_to_scan':
        # 源：STAR顶点；目标：扫描点云
        source_points = deformed_v.detach().cpu().numpy()
        target_points = scan_v
        # 为目标点云构建 KD-Tree
        kd_tree = spatial.cKDTree(target_points)
        # 查询每个源点的最近目标点
        dists, idxs = kd_tree.query(source_points, k=1)
        # 获取最近的目标点坐标（转为 GPU tensor）
        nearest_target = torch.from_numpy(target_points[idxs]).float().cuda()
        # 计算 L2 距离平方并平均
        diff = deformed_v - nearest_target
        loss = (diff ** 2).sum(dim=1).mean()

    elif direction == 'scan_to_star':
        # 源：扫描点云；目标：STAR顶点
        source_points = scan_v  # numpy array
        target_points = deformed_v.detach().cpu().numpy()  # numpy array
        # 为目标点云（STAR顶点）构建 KD-Tree
        kd_tree = spatial.cKDTree(target_points)
        # 查询每个扫描点的最近 STAR 顶点
        dists, idxs = kd_tree.query(source_points, k=1)
        # 获取最近的目标点坐标（从 GPU tensor 索引）
        nearest_target = deformed_v[torch.from_numpy(idxs).long().cuda()]
        # 将源点云（scan）转为 GPU tensor 并计算距离
        source_tensor = torch.from_numpy(source_points).float().cuda()
        diff = source_tensor - nearest_target
        loss = (diff ** 2).sum(dim=1).mean()

    else:
        raise ValueError("direction must be 'star_to_scan' or 'scan_to_star'")

    return loss

def compute_data_loss(
        deformed_v: torch.Tensor,
        scan_v: np.ndarray,
        direction: str = 'star_to_scan'
) -> torch.Tensor:
    """
    单向 Chamfer distance

    Args:
        deformed_v:
            当前SOMA人体顶点 [V,3] torch tensor

        scan_v:
            扫描人体顶点 [N,3] numpy

        direction:
            'star_to_scan':
                SOMA顶点 -> scan

            'scan_to_star':
                scan顶点 -> SOMA

    Returns:
        loss
    """

    device = deformed_v.device


    if direction == 'star_to_scan':

        # SOMA顶点作为source
        source_points = deformed_v.detach().cpu().numpy()

        # scan作为target
        target_points = scan_v


        kd_tree = spatial.cKDTree(target_points)

        _, idxs = kd_tree.query(
            source_points,
            k=1
        )


        # 最近scan点
        nearest_target = torch.from_numpy(
            target_points[idxs]
        ).float().to(device)


        diff = deformed_v - nearest_target


        loss = torch.mean(
            torch.sum(diff ** 2, dim=1)
        )


    elif direction == 'scan_to_star':

        # scan作为source
        source_points = scan_v


        # SOMA作为target
        target_points = deformed_v.detach().cpu().numpy()


        kd_tree = spatial.cKDTree(target_points)


        _, idxs = kd_tree.query(
            source_points,
            k=1
        )


        # 最近SOMA点索引
        idxs = torch.from_numpy(idxs).long().to(device)


        nearest_target = deformed_v[idxs]


        source_tensor = torch.from_numpy(
            source_points
        ).float().to(device)


        diff = source_tensor - nearest_target


        loss = torch.mean(
            torch.sum(diff ** 2, dim=1)
        )


    else:
        raise ValueError(
            "direction must be 'star_to_scan' or 'scan_to_star'"
        )


    return loss


def closure() -> torch.Tensor:
    """L-BFGS 优化闭包函数"""
    optimizer.zero_grad()

    # 前向传播生成当前顶点
    deformed_v = model.forward_vertices(**params)[0]

    # 计算损失
    loss = compute_data_loss(deformed_v, scan_v, "scan_to_star")  #
    loss.backward()

    # 实时可视化更新 (不追踪梯度)
    with torch.no_grad():
        updated_vertices = deformed_v.cpu().numpy()
        if 'ps_star_mesh' in globals():
            ps_star_mesh.update_vertex_positions(updated_vertices)
            ps.frame_tick()
    return loss


# 3.2、初始可视化star和scan人体，并更新优化后的star人体
ps.init()
ps_star_mesh = ps.register_surface_mesh(
    "star", star_v, star_f,
    color=np.array([0, 91, 255]) / 255,
    edge_color=[1, 1, 1], edge_width=0.001, smooth_shade=True
)
ps.register_surface_mesh("scan", scan_v, scan_f,
                         color=np.array([255, 164, 0]) / 255,
                         edge_width=0.001,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )
ps.set_up_dir("y_up")
ps.set_view_projection_mode("orthographic")
ps.set_ground_plane_mode("none")
ps.set_SSAA_factor(4)
ps.show()

BATCH_SIZE = 1
LR = 1  # 学习率
MAX_ITER = 500  # 最大迭代次数

# 3.3、优化参数，拟合扫描人体
# 设置可优化参数poses, betas, trans

params["shape"].requires_grad_(True)
params["body_pose"].requires_grad_(True)
params["head_pose"].requires_grad_(True)
params["hand_pose"].requires_grad_(True)
params["global_rotation"].requires_grad_(True)
params["global_translation"].requires_grad_(True)

optimizer = LBFGS([params["shape"],
                  params["body_pose"],
                  params["head_pose"],
                  params["hand_pose"],
                  params["global_rotation"],
                  params["global_translation"]],
                  lr=LR, max_iter=MAX_ITER, history_size=10,
                  line_search_fn="strong_wolfe")

loss = optimizer.step(closure)
print(f">>> 优化完成。最终损失值：{loss.item():.6f}")

# 3.5、可视化拟合结果
with torch.no_grad():
    star_v = model.forward_vertices(**params)[0]
    star_v = star_v.detach().cpu().numpy()

star_mesh = trimesh.Trimesh(star_v,star_f)
scan_mesh = trimesh.Trimesh(scan_v,scan_f)
star_mesh.export("final_star.obj")
scan_mesh.export("final_scan.obj")



ps.remove_all_structures()
ps.init()
ps.register_surface_mesh("star", star_v, star_f,
                         color=np.array([0, 91, 255]) / 255,
                         edge_width=0.001,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )
ps.register_surface_mesh("scan", scan_v, scan_f,
                         color=np.array([255, 164, 0]) / 255,
                         edge_width=0.001,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )
# ps.set_view_projection_mode("orthographic")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 这两行有用
ps.set_shadow_blur_iters(3)  # 模糊的程度，这个数值可以设置很大
ps.set_shadow_darkness(0.5)  # 这个数值也可以设置很大
ps.set_up_dir("y_up")  # 这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉
# ps.set_front_dir('x_front')  # 这两行有用
ps.set_SSAA_factor(4)
# ps.set_user_callback(callback)  # 设置回调函数
ps.show()
