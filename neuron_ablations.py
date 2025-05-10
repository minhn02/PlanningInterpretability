import torch
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform as tf
import utils
import open3d as o3d
from viplanner.viplanner.config import VIPlannerSemMetaHandler
import viplanner_wrapper
import pickle
from visualize_pc import visualize_semantic_top_down, get_points_in_fov_with_intrinsics
from statistical_analysis import find_top_k_weights_indices

"""
Layers:
encoder_depth (PlannerNet):
    A ResNet-style CNN with 4 stages:
        - conv1:     [N, 64, H/2,  W/2 ]
        - maxpool:   [N, 64, H/4,  W/4 ]
        - layer1:    [N, 64, H/4,  W/4 ]
        - layer2:    [N, 128, H/8,  W/8 ]
        - layer3:    [N, 256, H/16, W/16]
        - layer4:    [N, 512, H/32, W/32] ← Output for depth path

encoder_sem:
    - If `train_cfg.rgb` and `train_cfg.pre_train_sem`:
        Uses pre-trained RGBEncoder
            - Output: [N, 512, H/32, W/32]
    - Else:
        Uses PlannerNet with same architecture as above
            - Output: [N, 512, H/32, W/32]

fusion:
    Concatenates encoder outputs:
        - [N, 1024, H/32, W/32] ← [depth 512 + sem 512]

decoder:
    - Accepts fused feature map and a broadcasted goal vector
    - Applies 2 conv layers (or 4 if using DecoderS), followed by fully connected layers:
        - conv1:      [N, 512, ...]
        - conv2:      [N, 256, ...]
        - flatten     → [N, 256 * H', W']
        - fc1         → [N, 1024]
        - fc2         → [N, 512]
        - fc3         → [N, k × 3] → reshaped to [N, k, 3] (3D waypoints)
        - frc1, frc2  → [N, 1] (fear/confidence)

Notes:
------
- The exact number of trajectory keypoints (`k`) is set via `train_cfg.knodes`
- Semantic encoder can be frozen if pre-trained (controlled by `train_cfg.pre_train_freeze`)
- All spatial dimensions are based on input resolution (typically [192, 320] → final [12, 20])

Recommended Hook Points for Activation Access:
----------------------------------------------
- encoder_depth.layer4        → [N, 512, 12, 20]
- encoder_sem.layer4 / encoder_sem.resnet.layer4 → [N, 512, 12, 20]
- decoder.fc1 / fc2           → [N, 1024], [N, 512]
"""
    
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def neuron_ablations(cfg: DictConfig):
    # Access configuration parameters
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    device = cfg.viplanner.device
    pc_path = cfg.viplanner.point_cloud_path
    img_num = 25
    # interesting imgs: 46 and 16

    indices_file = "data/ground prop_encoder_top_100_weights.pkl"
    with open(indices_file, "rb") as f:
        ground_prop_indices = pickle.load(f)

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=True)

    # Load and process images from training data. Need to reshape to add batch dimension in front
    depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, img_num, device)

    # setup goal, also needs to have batch dimension in front
    # goals = torch.tensor([89.0, 212.6, 0.0], device=device).repeat(1, 1)
    goals = torch.tensor([317.5, 268.0, 0], device=device).repeat(1, 1)
    goals = viplanner_wrapper.transform_goal(camera_cfg_path, goals, img_num, device=device)
    # goals = torch.tensor([5.0, -3, 0], device=device).repeat(1, 1)

    depth_image = viplanner.input_transformer(depth_image)

    # forward/inference run 1
    keypoints, paths, fear = viplanner.plan_dual(depth_image, sem_image, goals, no_grad=True)
    print(f"Generated path with fear: {fear}")
    print(f"Keypoints: {keypoints}")
    cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device=device)
    path = viplanner.path_transformer(paths, cam_pos, cam_quat)

    # forward/inference run 2
    keypoints_ab, paths_ab, fear_ab = viplanner.plan_dual(depth_image, sem_image, goals, no_grad=True, ablate=ground_prop_indices)
    print(f"Generated path with fear: {fear_ab}")
    print(f"Keypoints after ablate: {keypoints_ab}")
    path_ab = viplanner.path_transformer(paths_ab, cam_pos, cam_quat)

    diff = torch.norm(path - path_ab).item()
    print(f"Path L2 difference: {diff:.6f}")
    print(f"Fear delta: {fear_ab.item() - fear.item():.6f}")

    point_cloud = o3d.io.read_point_cloud(pc_path)
    sem_handler = VIPlannerSemMetaHandler()

    # Get camera parameters
    cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device="cpu")
    cam_pos = cam_pos.cpu().numpy().squeeze(0)
    cam_quat = cam_quat.cpu().numpy().squeeze(0)

    # Define camera intrinsics from the camera_intrinsics.txt file in carla folder
    K = np.array([
        [430.69473, 0,        424.0],
        [0,         430.69476, 240.0],
        [0,         0,          1.0]
    ])
    img_width, img_height = 848, 480

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
        fig_name="Top-Down View Before Activation Patching",
        file_name=f"plotsf/top_down_view_before_patching_{img_num}.png"
    )

    fig, ax = visualize_semantic_top_down(
        fov_point_cloud,
        cam_pos=cam_pos,
        cam_quat=cam_quat,
        resolution=0.1,
        height_range=[-1.5, 2.0],
        sem_handler=sem_handler,
        forward_axis="X+",
        path=path_ab.cpu().numpy()[0],
        fig_name=f"Top-Down View After Activation Patching",
        file_name=f"plotsf/top_down_view_after_{len(ground_prop_indices)}_patching_{img_num}.png"
    )

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def sweep_ablations(cfg: DictConfig):
# Access configuration parameters
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    device = cfg.viplanner.device
    pc_path = cfg.viplanner.point_cloud_path
    img_num = 16

    weights_file = "data/ground prop_encoder_weights.pkl"
    with open(weights_file, "rb") as f:
        weights = pickle.load(f)

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=True)

    # Load and process images from training data. Need to reshape to add batch dimension in front
    depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, img_num, device)

    # setup goal, also needs to have batch dimension in front
    # goals = torch.tensor([89.0, 212.6, 0.0], device=device).repeat(1, 1)
    goals = torch.tensor([270, 129, 0], device=device).repeat(1, 1)
    goals = viplanner_wrapper.transform_goal(camera_cfg_path, goals, img_num, device=device)
    # goals = torch.tensor([5.0, -3, 0], device=device).repeat(1, 1)

    depth_image = viplanner.input_transformer(depth_image)

    fears = []
    keypoint_diff = []

    keypoints, paths, fear = viplanner.plan_dual(depth_image, sem_image, goals, no_grad=True)
    fears.append(fear.item())
    keypoint_diff.append(0.0)

    for k in range(25, 1025, 50):
        # Get the top k weights
        top_k_indices = np.array(find_top_k_weights_indices(weights, k))

        # forward/inference run 2
        keypoints_ab, paths_ab, fear_ab = viplanner.plan_dual(depth_image, sem_image, goals, no_grad=True, ablate=top_k_indices)
        print(f"Generated path with fear: {fear_ab} by removing top {k} weights")
        fears.append(fear_ab.item())
        keypoint_diff.append(torch.norm(keypoints - keypoints_ab).item())

    # Plot the results
    plt.figure()
    plt.plot(range(0, 1001, 50), fears)
    plt.xlabel("Number of Activations Patched")
    plt.ylabel("Fear Value")
    plt.title("Number of Activations Patched vs Fear Value")
    plt.savefig("plotsf/fear_vs_weights.png")

    plt.figure()
    plt.plot(range(0, 1001, 50), keypoint_diff)
    plt.xlabel("Number of Activations Patched")
    plt.ylabel("Keypoint Difference")
    plt.title("Number of Activations Patched vs Keypoint L2 Difference")
    plt.savefig("plotsf/keypoint_diff_vs_weights.png")

if __name__ == '__main__':
    neuron_ablations()
    # sweep_ablations()