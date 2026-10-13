import polyscope as ps

# from body_models.smpl.numpy import SMPL
from body_models.smplx.torch import SMPLX
import torch
import time

model = SMPLX(gender = "female")
params = model.get_rest_pose(batch_dims=(1,))


start = time.time()

for _ in range(100):
    vertices = model.forward_vertices(**params)

torch.cuda.synchronize()

end = time.time()


# 100个人体，需要多少秒
# soma:  180.19419193267822 ms
# smpl:  6.260526180267334 ms
# smplx:  15.473530292510986 ms
print((end - start) / 100 * 1000, "ms")