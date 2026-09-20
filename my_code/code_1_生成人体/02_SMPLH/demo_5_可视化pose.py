# from body_models.smpl.numpy import SMPL
from body_models.smplh.torch import SMPLH
import torch
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPLH(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

# 这里的参数是姿态参数，指的是旋转轴角
print(params["body_pose"].shape)
print(params["hand_pose"].shape)
print(params["pelvis_rotation"].shape)

skeleton = model.forward_skeleton(**params)
vertices = model.forward_vertices(**params)
faces = model.faces

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

J = skeleton[0].cpu().numpy()  # [24, 4, 4]
joints = J[:, :3, 3]  # [24, 3]  每个关节的 (x, y, z)

print(vertices.shape)
print(faces.shape)
print(skeleton.shape)
print(joints.shape)


# SMPL 关节父子连接表 (parent, child)
bones = [
    (0, 3), (3, 6), (6, 9), (9, 15),  # spine to head
    (0, 1), (1, 4), (4, 7), (7, 10),  # 왼쪽 다리
    (0, 2), (2, 5), (5, 8), (8, 11),  # 오른쪽 다리
    (9, 12), (12, 16), (16, 18), (18, 20),  # 왼쪽 팔
    (9, 12), (12, 17), (17, 19), (19, 21),  # 오른쪽 팔
]

ps.init()
ps.set_build_gui(False)
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         transparency=0.5
                         # material="flat"
                         )

ps.register_point_cloud("j", joints, radius=0.005, color=[1, 0, 0])
ps.register_curve_network("bone", joints, bones, radius=0.0015,
                          color=[255 / 255, 0, 218 / 255])

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
