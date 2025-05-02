import torch
import viplanner_wrapper
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from typing import Tuple
from PIL import Image

import open3d as o3d
import numpy as np

from viplanner.viplanner.traj_cost_opt.traj_cost import TrajCost
import utils

import cv2
import numpy as np

def visualize_path_on_image(paths, sem_image, original_sem_image=None):
    # Process image as before
    if original_sem_image is not None:
        sem_np = original_sem_image
    else:
        sem_np = sem_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if sem_np.max() <= 1.0:
            sem_np = sem_np * 255
        sem_np = sem_np.astype(np.uint8)
    
    # Create a copy to draw on
    vis_img = sem_np.copy()
    
    # Get path points
    path_points = paths[0].detach().cpu().numpy()
    
    # Extract x and y from path points
    xs = path_points[:, 0]
    ys = path_points[:, 2]
    
    # Center of the image
    img_center_x = vis_img.shape[1] // 2
    img_center_y = vis_img.shape[0] // 2
    
    # Scale factor - use 1/3 of the image width/height
    scale_x = vis_img.shape[1] // 3 / max(abs(xs.max()), abs(xs.min()))
    scale_y = vis_img.shape[0] // 3 / max(abs(ys.max()), abs(ys.min()))
    scale = min(scale_x, scale_y)
    
    # Project points to image
    points_2d = []
    for x, y in zip(xs, ys):
        u = int(img_center_x + x * scale)
        v = int(img_center_y + y * scale)
        if 0 <= u < vis_img.shape[1] and 0 <= v < vis_img.shape[0]:
            points_2d.append((u, v))
    
    print(f"Generated {len(points_2d)} points on image")
    
    # Use colors that will show up well on the semantic image
    path_color = (0, 0, 255)  # Red
    start_color = (0, 255, 0)  # Green
    goal_color = (255, 0, 0)   # Blue

    print(f"Path points in 2D: {points_2d}")
    
    # Draw path as line with gradient color
    if len(points_2d) > 1:
        for i in range(len(points_2d) - 1):
            # Create color gradient from green (start) to red (end)
            ratio = i / max(1, len(points_2d) - 2)
            b = int(255 * (1 - ratio))
            g = int(255 * (1 - ratio))
            r = int(255 * ratio)
            cv2.line(vis_img, points_2d[i], points_2d[i+1], (b, g, r), 2)
    
    # Draw waypoints
    for i, point in enumerate(points_2d):
        if i == 0:  # Start point
            cv2.circle(vis_img, point, 7, start_color, -1)
            cv2.putText(vis_img, "Start", (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, start_color, 2)
        elif i == len(points_2d) - 1:  # End point
            cv2.circle(vis_img, point, 7, goal_color, -1)
            cv2.putText(vis_img, "Goal", (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, goal_color, 2)
        else:
            cv2.circle(vis_img, point, 3, (0, 255, 255), -1)
    
    # Add a title
    cv2.putText(vis_img, "Planned Path", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save the image
    cv2.imwrite("path_on_semantic.png", vis_img)
    print("Saved visualization to path_on_semantic.png")
    
    # Display image if running in an environment with GUI
    try:
        cv2.imshow("Path on Semantic Image", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass
    
    return vis_img

def loss(
    waypoints: torch.Tensor,
    fear: torch.Tensor,
    traj_cost: TrajCost,
    odom: torch.Tensor,
    goal: torch.Tensor,
    step: float = 0.1,
    dataset: str = "train",
) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f"traj cost {traj_cost}")
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

def save_grad_as_image(grad, filename):
    """
    Saves the gradient as an image without using a matplotlib figure.

    Args:
        grad (torch.Tensor): Gradient tensor to save as an image.
        filename (str): File name to save the image.
    """
    # Convert gradient to numpy and normalize
    grad = grad.cpu().detach().numpy()
    grad = np.abs(grad).sum(axis=1)  # Sum over the channel dimension
    grad_min, grad_max = grad.min(), grad.max()
    if grad_max > grad_min:
        grad = (grad - grad_min) / (grad_max - grad_min)  # Normalize to [0, 1]
    else:
        grad = np.zeros_like(grad)  # Handle case where grad is constant

    # Convert to 8-bit (0-255) for saving as an image
    grad = (grad[0] * 255).astype(np.uint8)  # Use the first batch

    # Save as an image using PIL
    img = Image.fromarray(grad)
    img.save(filename)

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
    plt.figure(figsize=(10, 10))
    plt.imshow(grad[0], cmap="viridis")  # Use the first batch
    # plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename, dpi=300)
    plt.close()
    
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def visualize_path(cfg: DictConfig):
    # Access configuration parameters
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    device = cfg.viplanner.device

    img_num = 8

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=False)
    for i, module in enumerate(viplanner.net.modules()):
        if isinstance(module, torch.nn.ReLU):
            module.register_backward_hook(relu_hook_function)

    # Load and process images from training data. Need to reshape to add batch dimension in front
    depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, img_num, device)

    # setup goal, also needs to have batch dimension in front
    goals = torch.tensor([0.0, 50.0, 1.0], device=device).repeat(1, 1)
    goals = viplanner_wrapper.transform_goal(camera_cfg_path, goals, img_num, device)

    depth_image = viplanner.input_transformer(depth_image)

    # forward/inference
    _, paths, fear = viplanner.plan_dual(depth_image, sem_image, goals, no_grad=True)

    cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device=device)
    odom = torch.cat([cam_pos, cam_quat], dim=1)

    cfg = viplanner.train_config
    traj_cost = TrajCost(
                6,
                log_data=True,
                w_obs=cfg.w_obs,
                w_height=cfg.w_height,
                w_goal=cfg.w_goal,
                w_motion=cfg.w_motion,
                obstalce_thread=cfg.obstacle_thread)
    traj_cost.SetMap(data_path, cfg.cost_map_name)

    calc_loss, _ = loss(
        paths,
        fear,
        traj_cost,
        odom,
        goals,
        dataset="train",
    )

    visualize_path_on_image(paths, sem_image)

    

if __name__ == '__main__':
    visualize_path()