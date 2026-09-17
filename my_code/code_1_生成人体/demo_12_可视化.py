import trimesh
import polyscope as ps
import numpy as np

ps.init()

for i in range(15):
    path = "mesh_{}.obj".format(i)
    mesh = trimesh.load_mesh(path)
    v, f = mesh.vertices, mesh.faces
    v = v  + np.array([0, 0, 0.5 * i]) # 为什么这一行不起作用，mesh并没有间隔距离
    ps.register_surface_mesh("mesh_{}".format(i), v, f,
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
