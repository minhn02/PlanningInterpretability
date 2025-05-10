import torch
import viplanner_wrapper
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from PIL import Image
import open3d as o3d
import scipy.spatial.transform as tf
import os

from viplanner.viplanner.traj_cost_opt.traj_cost import TrajCost
from viplanner.viplanner.config import VIPlannerSemMetaHandler
import utils
from visualize_pc import visualize_semantic_top_down, get_points_in_fov_with_intrinsics
from generate_goal import generate_traversable_goal

def loss(
    waypoints: torch.Tensor,
    fear: torch.Tensor,
    traj_cost: TrajCost,
    odom: torch.Tensor,
    goal: torch.Tensor,
    step: float = 0.1,
    dataset: str = "train",
) -> Tuple[torch.Tensor, torch.Tensor]:
    loss = traj_cost.CostofTraj(
        waypoints,
        odom,
        goal,
        fear,
        0,
        ahead_dist=2.5,
        dataset=dataset,
    )

    return loss, waypoints

def relu_hook_function(module, grad_in, grad_out):
    if isinstance(module, torch.nn.ReLU):
        return (torch.clamp(grad_in[0], min=0.),)

def plot_saliency_map(grad, title, filename):
    """
    Plots the saliency map for the given gradient.

    Args:
        grad (torch.Tensor): Gradient tensor to visualize.
        title (str): Title of the plot.
        filename (str): File name to save the plot.
    """
    # Convert gradient to numpy and normalize
    grad = grad.cpu().detach().numpy()
    grad = np.abs(grad).sum(axis=1)  # Sum over the channel dimension
    grad = (grad - grad.min()) / (grad.max() - grad.min())  # Normalize to [0, 1]

    # Plot the saliency map
    plt.figure(figsize=(6, 6))
    plt.imshow(grad[0], cmap="viridis")  # Use the first batch
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    
    # Save individual image
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Return the figure for combining later
    fig = plt.gcf()
    plt.close()
    return fig

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def guided_backprop_multi(cfg: DictConfig):
    # Access configuration parameters
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    point_cloud_path = cfg.viplanner.point_cloud_path
    device = cfg.viplanner.device

    os.makedirs("plots", exist_ok=True)

    # Define the image numbers to process
    img_numbers = [25]  # Example numbers - change as needed
    
    # Load necessary shared resources
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    sem_handler = VIPlannerSemMetaHandler()
    
    # Define camera intrinsics
    K = np.array([
        [430.69473, 0,        424.0],
        [0,         430.69476, 240.0],
        [0,         0,          1.0]
    ])
    img_width, img_height = 848, 480
    
    # Create VIPlanner instance and add ReLU hooks
    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=False)
    for module in viplanner.net.modules():
        if isinstance(module, torch.nn.ReLU):
            module.register_backward_hook(relu_hook_function)
    
    # Prepare to collect all figures
    all_figures = []
    
    # Process each image
    for img_num in img_numbers:
        print(f"\n\n===== Processing Image {img_num} =====\n")
        
        # Get camera parameters
        cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device=device)
        cam_pos_np = cam_pos.cpu().numpy().squeeze(0)
        cam_quat_np = cam_quat.cpu().numpy().squeeze(0)
        
        # Get the points in the camera's field of view
        fov_point_cloud = get_points_in_fov_with_intrinsics(
            point_cloud, 
            cam_pos_np, 
            cam_quat_np, 
            K,
            img_width, 
            img_height,
            forward_axis="X+",
            max_distance=15
        )
        
        # Generate a traversable goal
        # goal_point = generate_traversable_goal(
        #     fov_point_cloud,
        #     cam_pos_np,
        #     sem_handler=sem_handler,
        #     min_distance=2.0,
        #     max_distance=10.0
        # )
        goal_point = [313.5, 266, 0.0]
        # goal_point = [319.3, 266.0, 0.0]
        
        if goal_point is None:
            print(f"No valid goal could be found for image {img_num}, skipping")
            continue
        
        # Load and process images
        depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, img_num, device)
        
        # Set up goal
        goals = torch.tensor(goal_point, device=device, dtype=torch.float32).unsqueeze(0)
        goals = viplanner_wrapper.transform_goal(camera_cfg_path, goals, img_num, device=device)
        
        # Prepare for gradient calculation
        depth_image_transformed = viplanner.input_transformer(depth_image)
        depth_image_transformed.requires_grad = True
        sem_image.requires_grad = True
        
        # Forward pass
        _, paths, fear = viplanner.plan_dual(depth_image_transformed, sem_image, goals, no_grad=False)
        print(f"Image {img_num}: Generated path with fear {fear.item():.4f}")
        
        # Transform path to world frame
        path_world = viplanner.path_transformer(paths, cam_pos, cam_quat)
        
        # Set up trajectory cost for loss calculation
        odom = torch.cat([cam_pos, cam_quat], dim=1)
        cfg_model = viplanner.train_config
        traj_cost = TrajCost(
            6,
            log_data=True,
            w_obs=cfg_model.w_obs,
            w_height=cfg_model.w_height,
            w_goal=cfg_model.w_goal,
            w_motion=cfg_model.w_motion,
            obstalce_thread=cfg_model.obstacle_thread
        )
        traj_cost.SetMap(data_path, cfg_model.cost_map_name)
        
        # Calculate loss and backpropagate
        calc_loss, _ = loss(paths, fear, traj_cost, odom, goals, dataset="train")
        calc_loss.backward()
        
        # Get gradients
        depth_grad = depth_image_transformed.grad
        sem_grad = sem_image.grad
        
        # Create the three visualizations for this image
        # 1. Top-down semantic view with path
        top_down_fig, _ = visualize_semantic_top_down(
            fov_point_cloud,
            cam_pos=cam_pos_np,
            cam_quat=cam_quat_np,
            resolution=0.1,
            height_range=[-1.5, 2.0],
            sem_handler=sem_handler,
            forward_axis="X+",
            path=path_world.detach().cpu().numpy()[0],
            fig_name=f"Image {img_num}: {fear.item():.4f} fear and goal {goal_point}",
            file_name=f"plots/top_down_{img_num}_{goal_point}.png",
        )
        
        # 2. Depth gradient saliency map
        depth_fig = plot_saliency_map(
            depth_grad, 
            f"Depth Saliency: Image {img_num}: goal {goal_point}", 
            f"plots/depth_saliency_{img_num}_{goal_point}.png"
        )
        
        # 3. Semantic gradient saliency map
        sem_fig = plot_saliency_map(
            sem_grad, 
            f"Semantic Saliency: Image {img_num}: goal {goal_point}", 
            f"plots/sem_saliency_{img_num}_{goal_point}.png"
        )

if __name__ == '__main__':
    guided_backprop_multi()