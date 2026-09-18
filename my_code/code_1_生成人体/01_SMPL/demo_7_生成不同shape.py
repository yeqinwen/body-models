# from body_models.smpl.numpy import SMPL
from body_models.smpl.torch import SMPL
import torch
import numpy as np
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPL(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

ps.init()

# params["shape"] 为 torch.zeros((1,10),dtype=torch.int64, device=device)
for i in range(7):
    params["shape"][0, 1] = i - 3

    vertices = model.forward_vertices(**params)
    skeleton = model.forward_skeleton(**params)
    faces = model.faces

    vertices = vertices[0].cpu().numpy()
    faces = faces.cpu().numpy()

    vertices = vertices + np.array([0, 0, i * 1])

    ps.register_surface_mesh("mesh_{}".format(i - 3), vertices, faces,
                             color=[0, 91 / 255, 255 / 255],
                             edge_width=0.003,
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
