import torch
import viplanner_wrapper
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from typing import Tuple
from PIL import Image

from viplanner.viplanner.traj_cost_opt.traj_cost import TrajCost
import utils

import open3d as o3d
from viplanner.viplanner.config import VIPlannerSemMetaHandler

def distance_to_class(pcd, goal_position, target_class_color, sem_handler):
    """
    Find the minimum distance from goal to a specific semantic class
    
    Args:
        pcd: The semantically labeled point cloud
        goal_position: 3D coordinates of goal [x, y, z]
        target_class_color: RGB color of the target class
        sem_handler: Instance of VIPlannerSemMetaHandler
    
    Returns:
        min_distance: Minimum distance to the target class
        closest_point: Coordinates of the closest point of that class
    """
    # Get class color from semantic handler if string provided
    if isinstance(target_class_color, str):
        target_class_color = sem_handler.class_color[target_class_color]
    
    # Convert to proper format for comparison
    target_color = np.array(target_class_color) / 255.0
    
    # Extract points and colors
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Find points matching the target class (with small tolerance for floating point)
    class_mask = np.all(np.abs(colors - target_color) < 0.01, axis=1)
    class_points = points[class_mask]
    
    if len(class_points) == 0:
        return float('inf'), None
    
    print(f"Found {len(class_points)} points of class {target_class_color}")
    
    # Calculate distances to all points of this class
    distances = np.linalg.norm(class_points - goal_position, axis=1)
    
    # Find minimum distance and corresponding point
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    closest_point = class_points[min_idx]
    
    return min_distance, closest_point
    
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def visualize_pc(cfg: DictConfig):
    # Access configuration parameters
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    device = cfg.viplanner.device

    img_num = 8

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=False)

    # Load and process images from training data. Need to reshape to add batch dimension in front
    depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, img_num, device)

    # setup goal, also needs to have batch dimension in front
    goals = torch.tensor([137, 111.0, 1.0], device=device).repeat(1, 1)
    goals = viplanner_wrapper.transform_goal(camera_cfg_path, goals, img_num, device)

    depth_image = viplanner.input_transformer(depth_image)
    depth_image.requires_grad = True
    sem_image.requires_grad = True

    # forward/inference
    _, paths, fear = viplanner.plan_dual(depth_image, sem_image, goals, no_grad=False)

    cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device=device)
    odom = torch.cat([cam_pos, cam_quat], dim=1)

    # cfg = viplanner.train_config
    # traj_cost = TrajCost(
    #             6,
    #             log_data=True,
    #             w_obs=cfg.w_obs,
    #             w_height=cfg.w_height,
    #             w_goal=cfg.w_goal,
    #             w_motion=cfg.w_motion,
    #             obstalce_thread=cfg.obstacle_thread)
    # traj_cost.SetMap(data_path, cfg.cost_map_name)

    # cost_map = traj_cost.cost_map
    pc_path = "/scratch/minh/school/282_project/carla/cloud.ply"
    sem_handler = VIPlannerSemMetaHandler()
    point_cloud = o3d.io.read_point_cloud(pc_path)

    vehicle_dist = distance_to_class(
        point_cloud,
        cam_pos.cpu().numpy(),
        "vehicle",
        sem_handler
    )

    sidewalk_dist = distance_to_class(
        point_cloud,
        cam_pos.cpu().numpy(),
        "sidewalk",
        sem_handler
    )

    pole_dist = distance_to_class(
        point_cloud,
        cam_pos.cpu().numpy(),
        "pole",
        sem_handler
    )

    bench_dist = distance_to_class(
        point_cloud,
        cam_pos.cpu().numpy(),
        "bench",
        sem_handler
    )

    print(f"Vehicle distance: {vehicle_dist[0]}")
    print(f"Sidewalk distance: {sidewalk_dist[0]}")
    print(f"Pole distance: {pole_dist[0]}")
    print(f"Bench distance: {bench_dist[0]}")



if __name__ == '__main__':
    visualize_pc()