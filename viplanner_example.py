import torch
import torch.nn.functional as F

from viplanner.viplanner.plannernet.autoencoder import DualAutoEncoder
from viplanner.viplanner.config.learning_cfg import TrainCfg

import numpy as np
from PIL import Image

VI_PLANNER_WEIGHT_PATH = "models/viplanner.pt"
VI_PLANNER_CONFIG_PATH = "models/viplannerconfig.yaml"
DATA_PATH = "/scratch/minh/school/282_project/carla"

DEVICE = 'cuda:4' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def get_models(device):
    vi_model = DualAutoEncoder(train_cfg=TrainCfg.from_yaml(VI_PLANNER_CONFIG_PATH))
    vi_state_dict, _ = torch.load(VI_PLANNER_WEIGHT_PATH)

    # remove all keys that start with encoder_sem because these are preloaded by Mask2Former
    # for key in list(vi_state_dict.keys()):
    #     if key.startswith('encoder_sem'):
    #         del vi_state_dict[key]
    vi_model.load_state_dict(vi_state_dict, strict=False)

    vi_model = vi_model.to(device).eval()

    return vi_model

def example_eval():
    # da_model, vi_model = get_models(DEVICE)

    # rgb_img = torch.rand(1, 3, 644, 392, device=DEVICE)
    # rgb_img_cropped = rgb_img[:, :, 2:-2, 4:-4]
    # goal_pos = torch.rand(1, 3, device=DEVICE)

    # depth = da_model.forward(rgb_img)
    # depth = F.interpolate(depth[:, None], (640, 384), mode="bilinear", align_corners=True)[0, 0]
    # depth = torch.reshape(depth, (1, 1, 640, 384))
    
    # # out = vi_model.forward(depth, rgb_img_cropped, goal_pos)
    # onnx_program = torch.onnx.export(vi_model, (depth, rgb_img_cropped, goal_pos), "viplanner.onnx")
    # # onnx_program.save("viplanner.onnx")
    vi_model = get_models(DEVICE)

    depth_image_file = f"{DATA_PATH}/depth/0008.npy"
    sem_image_file = f"{DATA_PATH}/semantics/0008.png"

    depth_image = np.load(depth_image_file)
    depth_image = np.array(depth_image, dtype=np.float32)
    sem_image = np.array(Image.open(sem_image_file), dtype=np.float32)

    goal_pos = torch.tensor([5.0, 5.0, 0.0], device=DEVICE).unsqueeze(0)
    goal_pos = torch.reshape(goal_pos, (1, 3))

    depth_image = torch.tensor(depth_image, device=DEVICE)
    sem_image = torch.tensor(sem_image, device=DEVICE)
    depth_image = torch.reshape(depth_image, (1, depth_image.shape[0], depth_image.shape[1], depth_image.shape[2]))
    sem_image = torch.reshape(sem_image, (1, sem_image.shape[0], sem_image.shape[1], sem_image.shape[2]))

    depth_image = depth_image.permute(0, 3, 1, 2)
    sem_image = sem_image.permute(0, 3, 1, 2)

    print("before interpolate depth image shape", depth_image.shape)
    print("before interpolate sem image shape", sem_image.shape)

    depth_image = F.interpolate(depth_image, (360, 640), mode="bilinear", align_corners=True)
    sem_image = F.interpolate(sem_image, (360, 640), mode="bilinear", align_corners=True)

    print("after interpolate depth image shape", depth_image.shape)
    print("after interpolate sem image shape", sem_image.shape)

    # depth_image = depth_image.permute(0, 3, 1, 2)
    # sem_image = sem_image.permute(0, 3, 1, 2)

    print("depth image shape", depth_image.shape)
    print("sem image shape", sem_image.shape)

    out = vi_model.forward(depth_image, sem_image, goal_pos)

    print(out)
    

if __name__ == '__main__':
    example_eval()