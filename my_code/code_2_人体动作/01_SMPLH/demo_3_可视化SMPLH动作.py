from body_models.smplh.torch import SMPLH
import torch
import polyscope as ps
import numpy as np
import polyscope.imgui as psim

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

###########################################################
# 1、读取动作，提取人体性别和帧数，用于批量生成人体

data = np.load("0015_sitting1_poses.npz")
gender = data["gender"]  # 从动作数据中获得人体性别
num_frame = data["trans"].shape[0]
print("帧数：", num_frame)

# 根据帧数，批量生成所有人体
model = SMPLH(gender=gender).to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(num_frame,))

# 2、设置人体的参数
params["shape"] = torch.tensor(data["betas"], dtype=torch.float32, device=device)

params["pelvis_rotation"] = torch.tensor(data["poses"][:, 0:3], dtype=torch.float32, device=device).reshape(num_frame,
                                                                                                            3)
params["body_pose"] = torch.tensor(data["poses"][:, 3:66], dtype=torch.float32, device=device).reshape(num_frame, 21, 3)
params["hand_pose"] = torch.tensor(data["poses"][:, 66:156], dtype=torch.float32, device=device).reshape(num_frame, 30,
                                                                                                         3)
params["global_translation"] = torch.tensor(data["trans"], dtype=torch.float32, device=device).reshape(num_frame, 3)

all_v = model.forward_vertices(**params)  # torch.Size([807, 6890, 3])
skeleton = model.forward_skeleton(**params)  # torch.Size([807, 52, 4, 4])
faces = model.faces  # torch.Size([13776, 3])

v0 = all_v[0].cpu().numpy()
f = faces.cpu().numpy()

# 设置call_back的参数
curr_frame = 0


def callback():
    # 使用callback回调函数进行交互
    global curr_frame

    slider_updated, curr_frame = psim.SliderInt("Curr Frame", curr_frame, 0, num_frame - 1)

    if slider_updated:
        ps_mesh.update_vertex_positions(all_v[curr_frame].cpu().numpy())


ps.init()
ps_mesh = ps.register_surface_mesh("mesh", v0, f,
                                   color=np.array([0, 91, 255]) / 255,
                                   edge_width=0.001,
                                   edge_color=[1, 1, 1],
                                   smooth_shade=True,
                                   # material="flat"
                                   )

ps.set_window_size(1200, 800)
ps.set_view_projection_mode("orthographic")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("z_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('y_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.set_user_callback(callback)  # 设置回调函数
ps.show()
