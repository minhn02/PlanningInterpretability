import torch
import torch.nn.functional as F

from viplanner.viplanner.plannernet.autoencoder import DualAutoEncoder
from viplanner.viplanner.config.learning_cfg import TrainCfg

import numpy as np
from PIL import Image

import utils
import viplanner_wrapper

VI_PLANNER_WEIGHT_PATH = "models/viplanner.pt"
VI_PLANNER_CONFIG_PATH = "models/viplannerconfig.yaml"
DATA_PATH = "/scratch/minh/school/282_project/carla"
CAMERA_CFG_PATH = "/scratch/minh/school/282_project/carla/camera_extrinsic.txt"
DEVICE = 'cuda:4' if torch.cuda.is_available() else 'cpu'

def viplanner_eval():
    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir="models", device=DEVICE)

    # Load and process images from training data. Need to reshape to add batch dimension in front
    depth_image_file = f"{DATA_PATH}/depth/0008.npy"
    sem_image_file = f"{DATA_PATH}/semantics/0008.png"
    depth_image = np.load(depth_image_file)
    depth_image = np.array(depth_image, dtype=np.float32)
    sem_image = np.array(Image.open(sem_image_file), dtype=np.float32)
    # turn into torch tensors
    depth_image = torch.tensor(depth_image, device=DEVICE)
    sem_image = torch.tensor(sem_image, device=DEVICE)
    depth_image = torch.reshape(depth_image, (1, depth_image.shape[0], depth_image.shape[1], depth_image.shape[2]))
    sem_image = torch.reshape(sem_image, (1, sem_image.shape[0], sem_image.shape[1], sem_image.shape[2]))
    depth_image = depth_image.permute(0, 3, 1, 2)
    sem_image = sem_image.permute(0, 3, 1, 2)
    
    # Load camera extrinsics collected during training
    cam_pos, cam_quat = utils.load_camera_extrinsics(CAMERA_CFG_PATH, 8, device=DEVICE)

    # setup goal, also needs to have batch dimension in front
    goals = torch.tensor([137, 111.0, 1.0], device=DEVICE).repeat(1, 1)

    # Use the viplanner_wrapper to load the model and perform planning
    goals = viplanner.goal_transformer(
        goals, cam_pos, cam_quat
    )
    _, paths, fear = viplanner.plan_dual(
        depth_image, sem_image, goals
    )
    print(f"paths {paths}, fear {fear}")

if __name__ == '__main__':
    viplanner_eval()