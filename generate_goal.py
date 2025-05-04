import torch
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform as tf
import utils
import contextlib
import open3d as o3d
from viplanner.viplanner.config import VIPlannerSemMetaHandler
import viplanner_wrapper
from visualize_pc import visualize_semantic_top_down, get_points_in_fov_with_intrinsics

# TODO this is taken from the semantic handler code, not sure if there's another way to import it
TRAVERSABLE_INTENDED_LOSS = 0
TRAVERSABLE_UNINTENDED_LOSS = 0.5
ROAD_LOSS = 1.5

def generate_traversable_goal(fov_point_cloud, cam_pos, sem_handler=None, min_distance=2.0, max_distance=10.0):
    """
    Generate a random goal point from traversable areas in the field of view
    
    Args:
        fov_point_cloud: Point cloud filtered to camera's field of view
        cam_pos: Camera position as numpy array [x, y, z]
        sem_handler: VIPlannerSemMetaHandler instance
        min_distance: Minimum distance from camera (meters)
        max_distance: Maximum distance from camera (meters)
        
    Returns:
        goal_point: 3D numpy array with [x, y, z] coordinates of goal in world frame
               or None if no suitable points found
    """
    if sem_handler is None:
        sem_handler = VIPlannerSemMetaHandler()
        
    # Extract points and colors
    points = np.asarray(fov_point_cloud.points)
    colors = np.asarray(fov_point_cloud.colors)
    
    if len(points) == 0:
        print("No points in field of view")
        return None
    
    # Get traversable intended classes
    traversable_classes = []
    for name, loss in sem_handler.class_loss.items():
        if loss == TRAVERSABLE_INTENDED_LOSS or loss == TRAVERSABLE_UNINTENDED_LOSS or loss == ROAD_LOSS:
            traversable_classes.append(name)
    
    print(f"Identified traversable classes: {traversable_classes}")
    
    # Get normalized colors for traversable classes
    traversable_colors = []
    for class_name in traversable_classes:
        rgb_color = sem_handler.class_color[class_name]
        normalized_color = tuple(np.array(rgb_color) / 255.0)
        traversable_colors.append(normalized_color)
    
    # Find points that match traversable classes
    traversable_mask = np.zeros(len(points), dtype=bool)
    
    for i, point_color in enumerate(colors):
        # Check if point color is close to any traversable color
        for trav_color in traversable_colors:
            color_diff = np.sum(np.abs(point_color - np.array(trav_color)))
            if color_diff < 0.1:  # Allow small tolerance for floating point differences
                traversable_mask[i] = True
                break
    
    traversable_points = points[traversable_mask]
    
    if len(traversable_points) == 0:
        print("No traversable points found in field of view")
        return None
    
    print(f"Found {len(traversable_points)} traversable points")
    
    # Calculate distances from camera
    distances = np.linalg.norm(traversable_points - cam_pos, axis=1)
    
    # Filter points by distance
    valid_mask = (distances >= min_distance) & (distances <= max_distance)
    valid_points = traversable_points[valid_mask]
    valid_distances = distances[valid_mask]
    
    if len(valid_points) == 0:
        print(f"No traversable points found within distance range [{min_distance}, {max_distance}]")
        return None
    
    print(f"Found {len(valid_points)} valid traversable points within distance range")
    
    # Randomly select a point, with probability weighted by distance
    # This encourages picking points that are farther away
    weights = valid_distances / np.sum(valid_distances)
    selected_idx = np.random.choice(len(valid_points), p=weights)
    goal_point = valid_points[selected_idx]
    
    print(f"Selected goal at {goal_point} (distance: {valid_distances[selected_idx]:.2f}m)")
    
    # Return the goal point in world frame
    return goal_point

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def generate_goal(cfg: DictConfig):
    # Access configuration parameters
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    point_cloud_path = cfg.viplanner.point_cloud_path
    device = cfg.viplanner.device

    # Define camera intrinsics from the camera_intrinsics.txt file in carla folder
    # TODO are these right intrinsics?
    K = np.array([
        [430.69473, 0,        424.0],
        [0,         430.69476, 240.0],
        [0,         0,          1.0]
    ])
    img_width, img_height = 848, 480

    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    sem_handler = VIPlannerSemMetaHandler()

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=True)

    # Get camera parameters
    for img_num in range(20, 29):
        cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device="cpu")
        cam_pos = cam_pos.cpu().numpy().squeeze(0)
        cam_quat = cam_quat.cpu().numpy().squeeze(0)

        # Get only the points in the camera's field of view
        fov_point_cloud = get_points_in_fov_with_intrinsics(
            point_cloud, 
            cam_pos, 
            cam_quat, 
            K,
            img_width, 
            img_height,
            forward_axis="X+",  # Use the detected best axis
            max_distance=15
        )

        goal_point = generate_traversable_goal(
        fov_point_cloud,
            cam_pos,
            sem_handler=sem_handler,
            min_distance=2.0,  # At least 2 meters from camera
            max_distance=10.0  # No more than 10 meters away
        )
        print(goal_point)

        # Load and process images from training data. Need to reshape to add batch dimension in front
        depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, img_num, device)

        # setup goal, also needs to have batch dimension in front
        goals = torch.tensor(goal_point, device=device, dtype=torch.float32).repeat(1, 1)
        goals = viplanner_wrapper.transform_goal(camera_cfg_path, goals, img_num, device=device)

        # forward/inference
        depth_image = viplanner.input_transformer(depth_image)
        _, paths, fear = viplanner.plan_dual(depth_image, sem_image, goals, no_grad=True)
        print(f"Generated path with fear: {fear}")
        cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device=device)
        path = viplanner.path_transformer(paths, cam_pos, cam_quat)

        cam_pos = cam_pos.cpu().numpy().squeeze(0)
        cam_quat = cam_quat.cpu().numpy().squeeze(0)

        # Visualize the results
        fig, ax = visualize_semantic_top_down(
            fov_point_cloud,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            resolution=0.1,
            height_range=[-1.5, 2.0],
            sem_handler=sem_handler,
            forward_axis="X+",
            path=path.cpu().numpy()[0],
            fig_name=f"goal_{img_num} with fear {fear.item():.2f}",
            file_name=f"plots/goal_{img_num}.png",
        )


def generate_all_goals_tensor(cfg: DictConfig, image_count=1000):
    # Access configuration parameters
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    point_cloud_path = cfg.viplanner.point_cloud_path
    device = cfg.viplanner.device

    K = np.array([
        [430.69473, 0,        424.0],
        [0,         430.69476, 240.0],
        [0,         0,          1.0]
    ])
    img_width, img_height = 848, 480

    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    sem_handler = VIPlannerSemMetaHandler()

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=True)

    # Get camera parameters
    goals = []
    for img_num in range(image_count):
        if img_num == 42:
            img_num = 100 # TEMP: Image 42 is low quality, so swap it for a better image
        cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device="cpu")
        cam_pos = cam_pos.cpu().numpy().squeeze(0)
        cam_quat = cam_quat.cpu().numpy().squeeze(0)

        with contextlib.redirect_stdout(None):
            # Get only the points in the camera's field of view
            fov_point_cloud = get_points_in_fov_with_intrinsics(
                point_cloud, 
                cam_pos, 
                cam_quat, 
                K,
                img_width, 
                img_height,
                forward_axis="X+",  # Use the detected best axis
                max_distance=15
            )

            goal_point = generate_traversable_goal(
                fov_point_cloud,
                cam_pos,
                sem_handler=sem_handler,
                min_distance=2.0,  # At least 2 meters from camera
                max_distance=10.0  # No more than 10 meters away
            )

        # setup goal, also needs to have batch dimension in front
        if goal_point is None:
            goal_point = [5.3700, 0.2616, 0.1474] # Hardcoded average of images 1-100
            print(f"No traversable terrain in image {img_num}, skipping...")
        goal = torch.tensor(goal_point, device=device, dtype=torch.float32).repeat(1, 1)
        goal = viplanner_wrapper.transform_goal(camera_cfg_path, goal, img_num, device=device)
        goals.append(goal)

    goals = torch.cat(goals, axis=0)
    return goals


if __name__ == '__main__':
    generate_goal()
