import trimesh
import torch

from body_models.garment_measurements.torch import GarmentMeasurements
import polyscope as ps
import numpy as np

model = GarmentMeasurements()

faces = model.faces

values = [-3, -2, -1, 0, 1, 2, 3]

ps.init()
for i in range(15):

    for v in values:
        # 每次重新初始化平均人体
        params = model.get_apose(batch_dims=(1,))

        # 设置第 i 个shape参数
        params["shape"][0, i] = v

        # 生成人体
        vertices = model.forward_vertices(**params)[0]

        vertices = vertices + np.array([i * 1, 0, v * 1])

        ps.register_surface_mesh("mesh_{}_{}".format(i,v), vertices, faces,
                                 color=np.array([0, 91, 255]) / 255,
                                 edge_width=0.0003,
                                 edge_color=[1, 1, 1],
                                 smooth_shade=True,
                                 # material="flat"
                                 )

# ps.set_view_projection_mode("perspective")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("y_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('z_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.show()
