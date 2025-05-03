import torch
import hydra
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform as tf
import utils
import open3d as o3d
from viplanner.viplanner.config import VIPlannerSemMetaHandler
import viplanner_wrapper
from PIL import Image

def project_points(points_3d, K):
    """
    points_3d: (N, 3) in camera frame
    K: (3, 3) intrinsics matrix
    returns: (N, 2) image coordinates
    """
  
    points_3d = np.array(points_3d)  
    uv = K @ points_3d.T  
    print("after projection: ", uv.T)
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = uv[:2] / uv[2:] 
    return uv.T

def ego_to_camera_frame(pts_ego):
    R = np.array([
        [0, -1,  0],  # y → -x
        [0,  0, -1],  # z → -y
        [1,  0,  0]   # x → z
    ])
    return (R @ pts_ego.T).T


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def visualize_pc(cfg: DictConfig):

    K = np.array([
        [430.69473, 0,        424.0],
        [0,         430.69476, 240.0],
        [0,         0,          1.0]
    ])
        
    # Access configuration parameters
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    device = cfg.viplanner.device
    img_num = 25

    
    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=True)

    # Load and process images from training data. Need to reshape to add batch dimension in front
    depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, img_num, device)

    # setup goal, also needs to have batch dimension in front
    goals = torch.tensor([319, 266, 1.0], device=device).repeat(1, 1)
    goals = viplanner_wrapper.transform_goal(camera_cfg_path, goals, img_num, device=device)
    # goals = torch.tensor([5.0, -3, 0], device=device).repeat(1, 1)

    depth_image = viplanner.input_transformer(depth_image)
    #print(f"depth image {depth_image}")

    # forward/inference
    _, paths, fear = viplanner.plan_dual(depth_image, sem_image, goals, no_grad=True)

    print("output path shape: ", paths.shape)
    print("original paths: ", paths)
    print(f"Generated path with fear: {fear}")
    cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device=device)
    path_world = viplanner.path_transformer(paths, cam_pos, cam_quat)


    path_camera = paths.squeeze(0).cpu().numpy()
    #path_camera = ego_to_camera_frame(path_camera)
    image_path = project_points(path_camera, K) 

    print("image path: ", image_path)
    sem_image_file = f"../carla/semantics/0025.png"

    # Load and resize image
    image = Image.open(sem_image_file).convert("RGB")
    image = image.resize((848, 480))  # (width, height)
    image = np.array(image)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(image)
    plt.plot(image_path[:, 0], image_path[:, 1], 'ro-', markersize=3, linewidth=2, label='Waypoints')
    plt.scatter(image_path[0, 0], image_path[0, 1], c='lime', s=40, label='Start')
    plt.scatter(image_path[-1, 0], image_path[-1, 1], c='blue', s=40, label='Goal')
    plt.axis('off')
    plt.legend()
    plt.title("Projected ViPlanner Waypoints on Semantic Image")
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    visualize_pc()