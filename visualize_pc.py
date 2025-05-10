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

def visualize_semantic_top_down(fov_point_cloud, cam_pos=None, cam_quat=None, resolution=0.1, 
                               height_range=None, sem_handler=None, forward_axis="X+", path=None,
                               fig_name="Top-Down Semantic View", file_name="plots/top_down_semantic_view.png"):
    """
    Create a top-down semantic map from the point cloud with class legend and camera position
    
    Args:
        fov_point_cloud: Point cloud with semantic colors
        cam_pos: Optional camera position [x,y,z]
        cam_quat: Optional camera orientation quaternion
        resolution: Cell size in meters
        height_range: Optional [min_height, max_height] to filter by height
        sem_handler: Optional VIPlannerSemMetaHandler instance for class names
        forward_axis: Which axis represents forward direction
        path: Optional path waypoints as array of shape (N, 3) for [x, y, z] coordinates
    
    Returns:
        fig, ax: The matplotlib figure and axes objects
    """
    points = np.asarray(fov_point_cloud.points)
    colors = np.asarray(fov_point_cloud.colors)
    
    if len(points) == 0:
        print("No points to visualize")
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        plt.title("Top-Down Semantic View (No Points)")
        return fig, ax
        
    # Filter by height if specified
    if height_range is not None:
        min_h, max_h = height_range
        height_mask = (points[:, 2] >= min_h) & (points[:, 2] <= max_h)
        points = points[height_mask]
        colors = colors[height_mask]
        
        if len(points) == 0:
            print(f"No points in height range [{min_h}, {max_h}]")
            fig = plt.figure(figsize=(10, 10))
            ax = plt.gca()
            plt.title(f"Top-Down Semantic View (No Points in Height Range [{min_h}, {max_h}])")
            return fig, ax
    
    # Determine grid size based on point cloud bounds
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    
    # Add padding to bounds
    padding = max(2.0, resolution * 10)  # At least 2m or 10 cells
    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding
    
    print(f"X range: {x_min:.2f} to {x_max:.2f}, Y range: {y_min:.2f} to {y_max:.2f}")
    
    # Create grid
    grid_size_x = int((x_max - x_min) / resolution) + 1
    grid_size_y = int((y_max - y_min) / resolution) + 1
    
    print(f"Creating top-down view with {grid_size_x} x {grid_size_y} cells")
    
    # Create empty semantic image
    semantic_img = np.zeros((grid_size_y, grid_size_x, 3), dtype=np.uint8)
    
    # Track unique colors for legend
    unique_colors_dict = {}
    
    # Project points to grid
    for i, (point, color) in enumerate(zip(points, colors)):
        grid_x = int((point[0] - x_min) / resolution)
        # Flip y-axis for correct top-down orientation
        grid_y = grid_size_y - 1 - int((point[1] - y_min) / resolution)
        
        # Ensure within bounds
        if 0 <= grid_x < grid_size_x and 0 <= grid_y < grid_size_y:
            # Convert from 0-1 range to 0-255
            pixel_color = (color * 255).astype(np.uint8)
            semantic_img[grid_y, grid_x] = pixel_color
            
            # Store color as tuple for legend lookup
            color_key = tuple(pixel_color)
            unique_colors_dict[color_key] = True
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Display the image
    ax.imshow(semantic_img)
    ax.set_title(fig_name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # =============================================
    # Add camera position and direction if provided
    # =============================================
    if cam_pos is not None:
        # Convert camera position to grid coordinates
        print(f"cam pos {cam_pos}")
        cam_grid_x = int((cam_pos[0] - x_min) / resolution)
        # Flip y-axis for camera position too
        cam_grid_y = grid_size_y - 1 - int((cam_pos[1] - y_min) / resolution)
        
        # Check if camera is within grid bounds
        if 0 <= cam_grid_x < grid_size_x and 0 <= cam_grid_y < grid_size_y:
            # Draw camera as a marker
            camera = ax.plot(cam_grid_x, cam_grid_y, 'bo', markersize=10, label='Camera Position')[0]
            
            # If we have orientation, add direction indicator
            if cam_quat is not None:
                # Get rotation matrix
                rot = tf.Rotation.from_quat(cam_quat).as_matrix()
                
                # Get forward direction based on specified axis
                if forward_axis == "X+":
                    forward = rot[:, 0]
                elif forward_axis == "Y+":
                    forward = rot[:, 1]
                elif forward_axis == "Z+":
                    forward = rot[:, 2]
                elif forward_axis == "X-":
                    forward = -rot[:, 0]
                elif forward_axis == "Y-":
                    forward = -rot[:, 1]
                else:  # Z-
                    forward = -rot[:, 2]
                
                # Scale for visualization (5 meters)
                arrow_length = int(2 / resolution)
                
                # Project to 2D (ignore z component)
                dx = forward[0] * arrow_length
                # Flip dy for top-down view
                dy = -forward[1] * arrow_length
                
                # Draw arrow
                ax.arrow(cam_grid_x, cam_grid_y, dx, dy, 
                       head_width=arrow_length/10, 
                       head_length=arrow_length/6, 
                       fc='blue', 
                       ec='blue')
        else:
            print(f"Camera position ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}) is outside the grid bounds")

    # ===========================
    # Add path overlay if provided
    # ===========================
    path_grid_x = []
    path_grid_y = []
    
    if path is not None:
        # Convert path to numpy if it's a tensor
        if isinstance(path, torch.Tensor):
            path = path.cpu().numpy()
            
        # Convert path points to grid coordinates
        for point in path:
            # Check if point is within bounds
            if (x_min <= point[0] <= x_max) and (y_min <= point[1] <= y_max):
                grid_x = int((point[0] - x_min) / resolution)
                # Flip y-axis for path points too
                grid_y = grid_size_y - 1 - int((point[1] - y_min) / resolution)
                path_grid_x.append(grid_x)
                path_grid_y.append(grid_y)
        
        if path_grid_x:  # Only plot if we have valid path points
            # Plot path as line with markers
            ax.plot(path_grid_x, path_grid_y, 'r-', linewidth=2.5, marker='o', 
                   markersize=6, markerfacecolor='yellow', zorder=5)
            
            # Add path start and end markers
            ax.plot(path_grid_x[0], path_grid_y[0], 'go', markersize=10)
            ax.plot(path_grid_x[-1], path_grid_y[-1], 'mo', markersize=10)
    
    # ==================================
    # Create consolidated legend contents
    # ==================================
    # Create legend items
    legend_elements = []
    legend_names = []
    
    # Process semantic classes with deduplication
    if sem_handler is not None:
        # Create mapping between normalized colors and class names
        color_to_class = {}
        for class_name, rgb_color in sem_handler.class_color.items():
            normalized_color = tuple(np.array(rgb_color) / 255.0)
            color_to_class[normalized_color] = class_name
        
        # Map detected colors to classes with deduplication
        class_to_color = {}  # Maps class names to their canonical colors
        
        for color_tuple in unique_colors_dict.keys():
            normalized_tuple = tuple(np.array(color_tuple) / 255.0)
            
            # Find closest match in semantic handler colors
            best_match = None
            best_diff = float('inf')
            
            for known_color in color_to_class.keys():
                diff = np.sum(np.abs(np.array(normalized_tuple) - np.array(known_color)))
                if diff < best_diff:
                    best_diff = diff
                    best_match = known_color
            
            # If we found a close enough match
            if best_diff < 0.1 and best_match in color_to_class:
                class_name = color_to_class[best_match]
                # Store canonical color for this class (if not already present)
                if class_name not in class_to_color:
                    class_to_color[class_name] = np.array(sem_handler.class_color[class_name]) / 255.0
    
        # Create legend items for semantic classes
        for class_name, color in sorted(class_to_color.items()):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color))
            legend_names.append(class_name)
    
    # If no semantic classes found, use color-based legend
    if not legend_elements and unique_colors_dict:
        for i, color_tuple in enumerate(unique_colors_dict.keys()):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=np.array(color_tuple) / 255.0))
            legend_names.append(f"Class {i+1}")
    
    # Add camera elements
    if cam_pos is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10))
        legend_names.append('Camera Position')
        
        if cam_quat is not None:
            legend_elements.append(plt.Line2D([0], [0], color='blue', lw=2))
            legend_names.append(f'View Direction ({forward_axis})')
    
    # Add path elements
    if path is not None and path_grid_x:
        legend_elements.append(plt.Line2D([0], [0], color='r', marker='o', 
                              markersize=6, markerfacecolor='yellow', lw=2))
        legend_names.append('Path')
        
        legend_elements.append(plt.Line2D([0], [0], marker='o', 
                              markersize=8, markerfacecolor='g', color='w'))
        legend_names.append('Path Start')
        
        legend_elements.append(plt.Line2D([0], [0], marker='o', 
                              markersize=8, markerfacecolor='m', color='w'))
        legend_names.append('Path End/Goal')
    
    # Create the legend
    if legend_elements:
        ax.legend(legend_elements, legend_names, loc="upper right", bbox_to_anchor=(1.15, 1))
    
    # Add grid lines and axis labels
    grid_interval = max(1, int(grid_size_x / 10))
    ax.set_xticks(np.arange(0, grid_size_x, grid_interval))
    ax.set_yticks(np.arange(0, grid_size_y, grid_interval))
    
    # Add real-world coordinates
    real_x_ticks = np.arange(0, grid_size_x, grid_interval) * resolution + x_min
    real_y_ticks = np.arange(0, grid_size_y, grid_interval) * resolution + y_min
    # Reverse y-tick labels to match flipped orientation
    real_y_ticks = real_y_ticks[::-1]
    ax.set_xticklabels([f"{x:.1f}" for x in real_x_ticks])
    ax.set_yticklabels([f"{y:.1f}" for y in real_y_ticks])
    
    plt.tight_layout()
    if file_name:
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
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

def get_points_in_fov_with_intrinsics(point_cloud, cam_pos, cam_quat, K, img_width, img_height, forward_axis="X+", max_distance=15.0):
    """
    Get points in the field of view using camera intrinsics and the detected forward direction
    
    Args:
        point_cloud: Point cloud as o3d.geometry.PointCloud
        cam_pos: Camera position [x, y, z]
        cam_quat: Camera orientation quaternion [qx, qy, qz, qw]
        K: Camera intrinsic matrix
        img_width: Image width in pixels
        img_height: Image height in pixels
        forward_axis: Which axis to use as forward direction (default "X+")
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
    
    # Create camera-aligned coordinate system based on the specified forward axis
    # We need to map our detected forward direction to the Z-axis for standard projection
    if forward_axis == "X+":
        # For X+ forward, we need to swap axes: X->Z, Y->X, Z->Y
        # This creates a transformation from world to camera-aligned frame
        cam_to_standard = np.array([
            [0, 1, 0],  # World Y becomes camera X
            [0, 0, 1],  # World Z becomes camera Y
            [1, 0, 0]   # World X becomes camera Z (forward)
        ])
    elif forward_axis == "X-":
        cam_to_standard = np.array([
            [0, -1, 0],  # -Y
            [0, 0, 1],   # Z
            [-1, 0, 0]   # -X -> Z
        ])
    # Add other axis mappings similarly
    else:
        # Default to standard Z-forward
        cam_to_standard = np.eye(3)
    
    # Transform points to camera frame first
    points_camera = (rot.T @ points_rel.T).T
    
    # Apply the camera-to-standard transformation
    points_camera = (cam_to_standard @ points_camera.T).T
    
    # Filter points in front (positive Z after transformation)
    front_indices = points_camera[:, 2] > 0
    
    print(f"Found {np.sum(front_indices)} points in front of camera")
    
    if not np.any(front_indices):
        return o3d.geometry.PointCloud()
        
    points_camera = points_camera[front_indices]
    points_world_filtered = points_world_filtered[front_indices]
    colors_filtered = colors_filtered[front_indices]
    
    # Project to image plane (vectorized)
    # Convert to homogeneous coordinates and normalize by z
    points_normalized = points_camera / points_camera[:, 2:]
    
    # Apply camera intrinsics
    pixel_coords = np.zeros((points_normalized.shape[0], 2))
    pixel_coords[:, 0] = K[0, 0] * points_normalized[:, 0] + K[0, 2]
    pixel_coords[:, 1] = K[1, 1] * points_normalized[:, 1] + K[1, 2]
    
    # Print pixel coordinate ranges for debugging
    print(f"Pixel coordinates range: X [{pixel_coords[:, 0].min():.1f}, {pixel_coords[:, 0].max():.1f}], " +
          f"Y [{pixel_coords[:, 1].min():.1f}, {pixel_coords[:, 1].max():.1f}]")
    print(f"Image dimensions: {img_width} x {img_height}")
    
    # Filter points within image bounds (add small margin for near-edge points)
    margin = 5  # 5-pixel margin
    in_fov = (
        (pixel_coords[:, 0] >= -margin) & 
        (pixel_coords[:, 0] < img_width + margin) & 
        (pixel_coords[:, 1] >= -margin) & 
        (pixel_coords[:, 1] < img_height + margin)
    )
    
    print(f"Found {np.sum(in_fov)} points in image bounds")
    
    # Create new point cloud with filtered points
    pcd_fov = o3d.geometry.PointCloud()
    if np.sum(in_fov) > 0:
        pcd_fov.points = o3d.utility.Vector3dVector(points_world_filtered[in_fov])
        pcd_fov.colors = o3d.utility.Vector3dVector(colors_filtered[in_fov])
    
    return pcd_fov

def find_distances_in_fov(pcd_fov, cam_pos, sem_handler, class_name=None, max_distance=15.0):
    """
    Find distances to objects of a specific class within the camera's FOV
    
    Args:
        pcd_fov: The semantic point cloud in the camera's FOV
        cam_pos: Camera position in world frame
        sem_handler: VIPlannerSemMetaHandler instance
        class_name: Target class name (optional)
        max_distance: Maximum distance to consider
        
    Returns:
        If class_name provided: (min_distance, closest_point)
        If class_name not provided: dict of class names to (distance, point) tuples
    """
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
    pc_path = cfg.viplanner.point_cloud_path
    img_num = 32

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=True)

    # Load and process images from training data. Need to reshape to add batch dimension in front
    depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, img_num, device)

    # setup goal, also needs to have batch dimension in front
    goals = torch.tensor([268, 129, 0], device=device).repeat(1, 1)
    goals = viplanner_wrapper.transform_goal(camera_cfg_path, goals, img_num, device=device)
    # goals = torch.tensor([5.0, -3, 0], device=device).repeat(1, 1)

    depth_image = viplanner.input_transformer(depth_image)
    print(f"depth image {depth_image}")

    def ablate_input_class(semantic_img, target_rgb, replacement_rgb=(0, 0, 0)):
        """
        Replace all pixels with a specific RGB color in a semantic image.

        Args:
            semantic_img (Tensor): (1, 3, H, W) float tensor with RGB values
            target_rgb (tuple): RGB color to ablate, e.g. (127, 0, 255)
            replacement_rgb (tuple): RGB color to replace with, default = (0, 0, 0)

        Returns:
            Tensor: Modified semantic image
        """
        sem = semantic_img.clone()
        r, g, b = sem[0]

        # Create mask where all 3 channels match the target RGB
        mask = (r == target_rgb[0]) & (g == target_rgb[1]) & (b == target_rgb[2])

        # Apply replacement color where mask is true
        r[mask] = replacement_rgb[0]
        g[mask] = replacement_rgb[1]
        b[mask] = replacement_rgb[2]

        return torch.stack([r, g, b], dim=0).unsqueeze(0)  # shape: (1, 3, H, W)

    ablated = ablate_input_class(sem_image, [127, 0, 255], [255, 128, 0])

    keypoints, paths, fear = viplanner.plan_dual(depth_image, sem_image, goals, no_grad=True)
    cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device=device)
    path = viplanner.path_transformer(paths, cam_pos, cam_quat)

    plot_path(path, camera_cfg_path, img_num)

    # Finds the largest activation
    def inspect(module, input, output):
        fc2_activations = output.detach().cpu().numpy()
        # Get absolute values
        abs_acts = np.abs(fc2_activations)

        # Find max value and its index
        max_idx_flat = np.argmax(abs_acts)
        max_idx = np.unravel_index(max_idx_flat, abs_acts.shape)
        max_val = abs_acts[max_idx]

        print(f"Max magnitude value: {max_val}")
        print(f"Index: {max_idx}")

    # Ablates a specfic neuron
    def ablate(module, input, output):
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
                - fc3         → [N, k x 3] → reshaped to [N, k, 3] (3D waypoints)
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
        output = output.clone()
        # output[0, 20, 5, 0] = 0.0
        output[:, :] = 0
        return output

    # Zero out a neuron in a layer (here it's layer4 of depth encoder)
    # hook = viplanner.net.decoder.fc1.register_forward_hook(ablate)

    # # # forward/inference run 2
    # _, paths_ab, fear_ab = viplanner.plan_dual(depth_image, sem_image, goals, no_grad=True)
    # # print(f"Generated path with fear: {fear_ab}")
    # path_ab = viplanner.path_transformer(paths_ab, cam_pos, cam_quat)

    # plot_path(path_ab, camera_cfg_path, img_num)

    # hook.remove()
    # diff = torch.norm(path - path_ab).item()
    # print(f"Path L2 difference: {diff:.6f}")
    # print(f"Fear delta: {fear_ab.item() - fear.item():.6f}")

def plot_path(path, camera_cfg_path, img_num):
    # TODO put this in the config
    pc_path = "/Users/ryan/Desktop/python/182/carla/cloud.ply"
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
        path=path.cpu().numpy()[0]
    )

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
        fov_point_cloud,
        cam_pos, 
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