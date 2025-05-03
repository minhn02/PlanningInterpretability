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
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def guided_backprop(cfg: DictConfig):
    # Access configuration parameters
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    device = cfg.viplanner.device

    img_num = 32

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=False)
    for i, module in enumerate(viplanner.net.modules()):
        if isinstance(module, torch.nn.ReLU):
            module.register_backward_hook(relu_hook_function)

    # Load and process images from training data. Need to reshape to add batch dimension in front
    depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, img_num, device)

    # setup goal, also needs to have batch dimension in front
    goals = torch.tensor([49, 152.5, 1.0], device=device).repeat(1, 1)
    goals = viplanner_wrapper.transform_goal(camera_cfg_path, goals, img_num, device)

    depth_image = viplanner.input_transformer(depth_image)
    depth_image.requires_grad = True
    sem_image.requires_grad = True

    # forward/inference
    _, paths, fear = viplanner.plan_dual(depth_image, sem_image, goals, no_grad=False)
    print(f"generated path with fear {fear}")

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

    print("loss: ", calc_loss.item())
    calc_loss.backward()
    
    depth_grad = depth_image.grad
    sem_grad = sem_image.grad

    print("depth grad: ", depth_grad)
    print("sem grad: ", sem_grad)

    plot_saliency_map(depth_grad, "Depth Gradient Saliency Map", "plots/depth_saliency_map.png")
    plot_saliency_map(sem_grad, "Semantic Gradient Saliency Map", "plots/sem_saliency_map.png")

    # Save gradients as images
    # save_grad_as_image(depth_grad, "depth_saliency_map.png")
    # save_grad_as_image(sem_grad, "plots/sem_saliency_map.png")

if __name__ == '__main__':
    guided_backprop()