import polyscope as ps
import trimesh
import numpy as np
import torch
# from body_models.soma.numpy import SOMA
from body_models.soma.torch import SOMA

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




