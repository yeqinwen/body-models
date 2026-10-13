import numpy as np

data = np.load("0015_sitting1_poses.npz")

for key in data.files:
    value = data[key]

    if isinstance(value, np.ndarray):
        print(f"{key:30s} shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"{key:30s} type={type(value)}")

trans = data["trans"]
gender = data["gender"]
mocap_framerate = data["mocap_framerate"]
betas = data["betas"]
dmpls = data["dmpls"]
poses = data["poses"]

print(trans)
print(gender)
print(mocap_framerate)
print(betas)
print(dmpls)
print(poses)
