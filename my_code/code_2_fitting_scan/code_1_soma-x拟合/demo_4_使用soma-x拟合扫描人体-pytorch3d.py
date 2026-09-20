import polyscope as ps
import trimesh
import numpy as np
import torch
# from body_models.soma.numpy import SOMA
from body_models.soma.torch import SOMA
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes

from loss import Chamfer_loss, Shape_reg_loss, Pose_reg_loss

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

# ==================== 2. 参数配置 ====================
BATCH_SIZE = 1
MAX_ITER = 500

# 损失函数权重：控制各项损失在总优化目标中的重要性
loss_weights_dict = {
    'chamfer': 1000000,  # 主要数据项：强制STAR顶点贴近扫描点云
    'shape': 10,  # 形状正则化：防止体形参数过度变形
    'pose': 100,  # 姿态正则化：把人体向A姿态拉
}

model = SOMA().to(device)
params = model.get_apose(batch_dims=(1,))
vertices = model.forward_vertices(**params)


for k,v in params.items():
    if torch.is_tensor(v):
        print(k, v.device)



# 初始A姿态
initial_A_pose = params["body_pose"].clone().detach().to(device)

star_v_np = vertices[0].cpu().numpy()
star_f_np = model.faces.cpu().numpy()

# ==================== 4. 加载扫描人体 ====================
path = "rotate_scan.obj"
verts, faces, aux = load_obj(path)
scan_v_np = verts.cpu().numpy()
scan_f_np = faces.verts_idx.cpu().numpy()

# 粗略对齐：重心和底部对齐到STAR位置
scan_v_np = scan_v_np - scan_v_np.mean(axis=0) + star_v_np.mean(axis=0)
scan_v_np[:, 1] = scan_v_np[:, 1] - np.min(scan_v_np[:, 1]) + np.min(star_v_np[:, 1])

scan_v_tensor = torch.tensor(scan_v_np, dtype=torch.float32, device=device)
scan_f_tensor = torch.tensor(scan_f_np, dtype=torch.int64, device=device)

scan_mesh = Meshes(verts=[scan_v_tensor], faces=[scan_f_tensor])

print(f"扫描人体顶点数: {scan_v_tensor.shape[1]}")
print(f"扫描人体面数: {scan_f_tensor.shape[0]}")

# ==================== 5. 初始化可视化 ====================
ps.set_build_gui(False)
ps.set_program_name("Fitting 3D body scan")
ps.init()
ps_star_mesh = ps.register_surface_mesh("star", star_v_np, star_f_np,
                                        color=np.array([0, 91, 255]) / 255,
                                        edge_color=[1, 1, 1],
                                        smooth_shade=True
                                        )
ps_scan_mesh = ps.register_surface_mesh("scan", scan_v_np, scan_f_np,
                                        color=np.array([255, 164, 0]) / 255,
                                        edge_color=[1, 1, 1],
                                        smooth_shade=True
                                        )
ps.set_ground_plane_mode("shadow_only")
ps.set_shadow_blur_iters(3)
ps.set_shadow_darkness(0.5)
ps.set_up_dir("y_up")
ps.set_SSAA_factor(4)
ps.show()

# ==================== 6. 优化器设置 ====================
# 优化姿态、形状、平移三个参数组（约75个参数），而非直接优化每个顶点
params["shape"].requires_grad_(True)
params["body_pose"].requires_grad_(True)
# params["head_pose"].requires_grad_(True)
# params["hand_pose"].requires_grad_(True)
params["global_rotation"].requires_grad_(True)
params["global_translation"].requires_grad_(True)

optim_params = [params["shape"],
                params["body_pose"],
                # params["head_pose"],
                # params["hand_pose"],
                params["global_rotation"],
                params["global_translation"]]

optimizer = torch.optim.Adam(optim_params, lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

star_f_tensor = torch.tensor(star_f_np, dtype=torch.int64, device=device)

# ==================== 7. 训练循环 ====================
for i in range(MAX_ITER):
    optimizer.zero_grad()

    # 前向：根据当前参数生成STAR网格
    deform_vertices = model.forward_vertices(**params)
    star_mesh = Meshes(
        verts=[deform_vertices[0].to(device)],
        faces=[star_f_tensor.to(device)]
    )

    # ==================== 8. 损失函数定义（从外部导入） ====================
    loss_dict = {}
    chamfer_loss = Chamfer_loss(star_mesh, scan_mesh, num_samples=5000)  # 采样5000，速度更快，采样30000，拟合更精确
    shape_reg_loss = Shape_reg_loss(params["shape"])
    pose_reg_loss = Pose_reg_loss(params["body_pose"], initial_A_pose)

    loss_dict["chamfer"] = loss_weights_dict["chamfer"] * chamfer_loss
    loss_dict["shape"] = loss_weights_dict["shape"] * shape_reg_loss
    loss_dict["pose"] = loss_weights_dict["pose"] * pose_reg_loss

    loss = sum(loss_dict.values())

    # 反向传播：计算loss对poses, betas, trans的梯度
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
    optimizer.step()  # 更新参数
    scheduler.step()  # 更新学习率

    # ==================== 9. 实时更新可视化 ====================
    with torch.no_grad():
        deform_vertices = model.forward_vertices(**params)
        updated_vertices = deform_vertices[0].cpu().numpy()
        if 'ps_star_mesh' in globals():
            ps_star_mesh.update_vertex_positions(updated_vertices)
            ps.frame_tick()

# ==================== 10. 保存结果 ====================
with torch.no_grad():
    deform_vertices = model.forward_vertices(**params)
    final_star_v_np = deform_vertices[0].detach().cpu().numpy()

save_obj("final_star.obj", torch.Tensor(final_star_v_np), torch.Tensor(star_f_np))
save_obj("final_scan.obj", scan_v_tensor, scan_f_tensor)
print("结果已保存！")

ps.remove_all_structures()
ps.init()
ps.register_surface_mesh("star_fitted", final_star_v_np, star_f_np,
                         color=np.array([0, 91, 255]) / 255,
                         edge_color=[1, 1, 1],
                         smooth_shade=True
                         )
ps.register_surface_mesh("scan", scan_v_np, scan_f_np,
                         color=np.array([255, 164, 0]) / 255,
                         edge_color=[1, 1, 1],
                         smooth_shade=True
                         )
ps.set_up_dir("y_up")
ps.set_ground_plane_mode("shadow_only")
ps.set_shadow_blur_iters(3)
ps.set_shadow_darkness(0.5)
ps.set_SSAA_factor(4)
ps.show()
