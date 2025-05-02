import torch
import viplanner_wrapper
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from typing import Tuple
from PIL import Image
import scipy.spatial.transform as tf


from viplanner.viplanner.traj_cost_opt.traj_cost import TrajCost
import utils

import open3d as o3d
from viplanner.viplanner.config import VIPlannerSemMetaHandler

def visualize_fov_point_cloud(fov_point_cloud, cam_pos, cam_quat=None, goal_pos=None):
    """
    Visualize FOV point cloud using matplotlib
    
    Args:
        fov_point_cloud: open3d point cloud containing filtered points in FOV
        cam_pos: Camera position [x, y, z]
        cam_quat: Camera quaternion (optional) for orientation display
        goal_pos: Goal position (optional)
    """
    # Extract points and colors
    points = np.asarray(fov_point_cloud.points)
    colors = np.asarray(fov_point_cloud.colors)
    
    if len(points) == 0:
        print("No points to visualize")
        return
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors,  # Use semantic colors
        s=1,       # Point size
        alpha=0.5  # Transparency
    )
    
    # Plot camera position
    ax.scatter(
        [cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
        color='blue',
        s=100,
        marker='^',
        label='Camera'
    )
    
    # If we have camera orientation, draw a line showing the viewing direction
    if cam_quat is not None:
        # Convert quaternion to rotation matrix
        rot = tf.Rotation.from_quat(cam_quat).as_matrix()
        # Camera looks along positive Z-axis
        # Create a viewing direction vector of length 5 units
        view_dir = cam_pos + 5 * rot[:, 2]  # Use third column for forward direction
        # Draw line from camera to view direction
        ax.plot(
            [cam_pos[0], view_dir[0]],
            [cam_pos[1], view_dir[1]],
            [cam_pos[2], view_dir[2]],
            color='blue',
            linewidth=2
        )
    
    # If goal position provided, plot it
    if goal_pos is not None:
        ax.scatter(
            [goal_pos[0]], [goal_pos[1]], [goal_pos[2]],
            color='red',
            s=100,
            marker='*',
            label='Goal'
        )
    
    # Get point cloud bounds with some padding
    all_points = [points]
    if cam_pos is not None:
        all_points.append(np.array([cam_pos]))
    if goal_pos is not None:
        all_points.append(np.array([goal_pos]))
    
    all_points = np.vstack(all_points)
    min_bounds = all_points.min(axis=0) - 1
    max_bounds = all_points.max(axis=0) + 1
    
    # Set axis limits
    ax.set_xlim([min_bounds[0], max_bounds[0]])
    ax.set_ylim([min_bounds[1], max_bounds[1]])
    ax.set_zlim([min_bounds[2], max_bounds[2]])
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Camera FOV Point Cloud ({len(points)} points)')
    
    # Add legend
    ax.legend()
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    return fig, ax

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

def get_points_in_view_cone(point_cloud, cam_pos, cam_quat, angle_deg=60, max_distance=15.0):
    """
    Get points in a cone-shaped field of view from the camera
    
    Args:
        point_cloud: Point cloud as o3d.geometry.PointCloud
        cam_pos: Camera position [x, y, z]
        cam_quat: Camera orientation quaternion [qx, qy, qz, qw]
        angle_deg: Half-angle of the viewing cone in degrees
        max_distance: Maximum distance to consider
        
    Returns:
        Points in FOV as a new point cloud with semantic colors
    """
    # Get points and colors as numpy arrays
    points_world = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    
    print(f"Original point cloud has {len(points_world)} points")
    
    # Convert camera position to numpy
    if isinstance(cam_pos, torch.Tensor):
        cam_pos = cam_pos.cpu().numpy()
    if isinstance(cam_quat, torch.Tensor):
        cam_quat = cam_quat.cpu().numpy()
    
    # Convert to camera frame
    rot = tf.Rotation.from_quat(cam_quat).as_matrix()
    if rot.ndim == 3 and rot.shape[0] == 1:
        rot = rot.squeeze(0)
        
    # Calculate vectors from camera to each point
    points_rel = points_world - cam_pos
    
    # Calculate distances
    distances = np.linalg.norm(points_rel, axis=1)
    
    # Filter by distance
    dist_mask = distances <= max_distance
    points_rel = points_rel[dist_mask]
    points_world_filtered = points_world[dist_mask]
    colors_filtered = colors[dist_mask]
    distances = distances[dist_mask]
    
    print(f"Filtered to {len(points_rel)} points within {max_distance} meters")
    
    # Normalize direction vectors
    with np.errstate(divide='ignore', invalid='ignore'):
        dir_vecs = points_rel / distances[:, np.newaxis]
    
    # Replace NaNs with zeros
    dir_vecs = np.nan_to_num(dir_vecs)
    
    # Try all possible axes as forward direction
    axes = {
        "Z+": rot[:, 2],      # Standard assumption
        "X+": rot[:, 0],      # Alternative axes
        "Y+": rot[:, 1],      
        "Z-": -rot[:, 2],     # Inverted axes
        "X-": -rot[:, 0],
        "Y-": -rot[:, 1]
    }
    
    # Calculate minimum cosine for the FOV angle
    min_cos = np.cos(np.radians(angle_deg))
    
    # Test all axes to find the one with most points
    results = {}
    for name, forward_vec in axes.items():
        # Calculate dot product with this forward vector (cosine of angle)
        cos_angles = np.dot(dir_vecs, forward_vec)
        
        # Find points within the cone for this axis
        in_fov = cos_angles >= min_cos
        count = np.sum(in_fov)
        results[name] = (count, in_fov)
        print(f"Using {name} as forward: {count} points in view cone")
    
    # Find best axis
    best_axis = max(results.keys(), key=lambda k: results[k][0])
    best_count, best_in_fov = results[best_axis]
    
    print(f"Best forward direction is {best_axis} with {best_count} points")
    print(f"Using {best_axis} as forward direction")
    
    # Create resulting point cloud
    pcd_fov = o3d.geometry.PointCloud()
    if best_count > 0:
        pcd_fov.points = o3d.utility.Vector3dVector(points_world_filtered[best_in_fov])
        pcd_fov.colors = o3d.utility.Vector3dVector(colors_filtered[best_in_fov])
    
    # As a fallback, if we found very few points, try with a wider angle
    if best_count < 100 and angle_deg < 85:
        print(f"Few points found ({best_count}), trying with wider angle")
        wider_angle = min(angle_deg + 30, 85)
        wider_pcd = get_points_in_view_cone(point_cloud, cam_pos, cam_quat, wider_angle, max_distance)
        
        if len(wider_pcd.points) > best_count:
            print(f"Wider angle {wider_angle}° gave better results: {len(wider_pcd.points)} points")
            return wider_pcd
    
    return pcd_fov

def find_distances_in_fov(point_cloud, cam_pos, cam_quat, angle_deg, sem_handler, class_name=None, max_distance=15.0):
    """
    Find distances to objects of a specific class within the camera's FOV
    
    Args:
        point_cloud: The semantic point cloud
        cam_pos: Camera position in world frame
        cam_quat: Camera orientation as quaternion
        K: Camera intrinsic matrix
        img_width, img_height: Image dimensions
        sem_handler: VIPlannerSemMetaHandler instance
        class_name: Target class name (optional)
        max_distance: Maximum distance to consider
        
    Returns:
        If class_name provided: (min_distance, closest_point)
        If class_name not provided: dict of class names to (distance, point) tuples
    """
    # Get points in FOV
    pcd_fov = get_points_in_view_cone(point_cloud, cam_pos, cam_quat, angle_deg, max_distance)
    
    # If no points in FOV
    if len(pcd_fov.points) == 0:
        if class_name:
            return float('inf'), None
        else:
            return {}
    
    # If looking for specific class
    if class_name:
        return distance_to_class(pcd_fov, cam_pos, class_name, sem_handler)
    
    # If analyzing all classes
    results = {}
    for name in sem_handler.names:
        distance, point = distance_to_class(pcd_fov, cam_pos, name, sem_handler)
        if point is not None:  # Only include classes that were found
            results[name] = (distance, point)
    
    # Sort by distance
    results = dict(sorted(results.items(), key=lambda item: item[1][0]))
    return results
    
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
    # Load point cloud and camera parameters
    pc_path = "/scratch/minh/school/282_project/carla/cloud.ply"
    point_cloud = o3d.io.read_point_cloud(pc_path)
    sem_handler = VIPlannerSemMetaHandler()

    # Get camera parameters
    cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device="cpu")
    cam_pos = cam_pos.cpu().numpy().squeeze(0)
    cam_quat = cam_quat.cpu().numpy().squeeze(0)

    # Define camera intrinsics
    K = np.array([
        [430.69473, 0,        424.0],
        [0,         430.69476, 240.0],
        [0,         0,          1.0]
    ])
    img_width, img_height = 848, 480

    # Get only the points in the camera's field of view
    fov_point_cloud = get_points_in_view_cone(
        point_cloud, 
        cam_pos, 
        cam_quat, 
        angle_deg=60,  # Half-angle of the viewing cone
        max_distance=15  # Optional distance limit
    )

    fig, ax = visualize_fov_point_cloud(fov_point_cloud, cam_pos, cam_quat)
    fig.savefig("fov_point_cloud.png", dpi=300)

    # Find distances to specific class
    vehicle_dist, vehicle_point = distance_to_class(
        fov_point_cloud,
        cam_pos,
        "vehicle",
        sem_handler
    )
    print(f"Distance to nearest visible vehicle: {vehicle_dist:.2f}")

    # Or analyze all visible classes
    visible_objects = find_distances_in_fov(
        point_cloud, 
        cam_pos, 
        cam_quat,
        60,
        sem_handler
    )

    # Print results
    print("\nVisible Objects Analysis:")
    print("-" * 50)
    print(f"{'Class':<15} {'Distance (m)':<15} {'Position'}")
    print("-" * 50)
    for class_name, (distance, point) in visible_objects.items():
        print(f"{class_name:<15} {distance:<15.2f} {point}")



if __name__ == '__main__':
    visualize_pc()