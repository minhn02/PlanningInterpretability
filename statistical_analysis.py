import torch
import viplanner_wrapper
import hydra
import os
import contextlib
import utils
from omegaconf import DictConfig

from typing import List, Optional, Tuple
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from scipy.stats import skew, kurtosis, entropy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pickle

from generate_goal import generate_all_goals_tensor, generate_traversable_goal

from viplanner.viplanner.config.viplanner_sem_meta import VIPLANNER_SEM_META, OBSTACLE_LOSS
from viplanner.viplanner.config import VIPlannerSemMetaHandler
from visualize_pc import get_points_in_fov_with_intrinsics, find_distances_in_fov


IMAGE_COUNT = 100
BATCH_SIZE = 50


def get_image_batches(cfg: DictConfig) -> Tuple[torch.tensor, torch.tensor]:
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    device = cfg.viplanner.device

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device)

    print("Loading images...")

    images = range(IMAGE_COUNT)
    depth_images = []
    sem_images = []
    for n in images:
        if n == 42:
            n = 100 # TEMP: Image 42 is of low quality, so swap out for a better image
        depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, n, device)
        sem_image = viplanner.transform(sem_image) / 255
        depth_image = viplanner.input_transformer(depth_image)
        depth_images.append(depth_image)
        sem_images.append(sem_image)
    depth_img = torch.cat(depth_images, axis=0)
    sem_img = torch.cat(sem_images, axis=0)
    return depth_img, sem_img


def get_goals(cfg: DictConfig) -> torch.tensor:
    """
    Get goals from config file.

    Args:
        cfg: config file

    Returns:
        torch.Tensor of shape [N, 3]
    """
    if os.path.exists(f"checkpoints/goals.pt"):
        goals = torch.load(f"checkpoints/goals.pt")
    else:
        goals = generate_all_goals_tensor(cfg, IMAGE_COUNT)
        torch.save(goals, f"checkpoints/goals.pt")
    return goals


def compute_activations(cfg: DictConfig, layer: str) -> torch.tensor:
    """
    Compute the activations of the provided images at a particular layer of the model.

    Args:
        cfg (DictConfig): config file
        layer (str): A string representing the layer, must be one of
            "encoder" or "decoder-n" for an int n in [1, 5].

    Returns:
        torch.tensor: the activations at the layer.
    """
    depth_img, sem_img = get_image_batches(cfg)
    goal = get_goals(cfg)

    print("processed depth size", depth_img.shape)
    print("processed sem size", sem_img.shape)
    print("goal size", goal.shape)
    print(f"Computing activations for layer {layer}...")

    # List of layer operations in planning network
    model_path = cfg.viplanner.model_path
    device = cfg.viplanner.device
    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device)
    decoder = viplanner.net.decoder
    layers = [
        lambda x: decoder.relu(decoder.conv1(x)),
        lambda x: decoder.relu(decoder.conv2(x)),
        lambda x: decoder.relu(decoder.fc1(torch.flatten(x, 1))),
        lambda x: decoder.relu(decoder.fc2(x)),
        lambda x: decoder.fc3(x).reshape(-1, decoder.k, 3),
    ]

    activation_batches = []
    for i in range(IMAGE_COUNT // BATCH_SIZE):
        with torch.no_grad():
            print(f"Image {i * BATCH_SIZE} / {IMAGE_COUNT}")
            batch_start = i * BATCH_SIZE
            batch_end = (i+1) * BATCH_SIZE

            # encode depth
            x_depth = depth_img[batch_start:batch_end, :, :, :]
            x_sem = sem_img[batch_start:batch_end, :, :, :]
            x_depth = x_depth.expand(-1, 3, -1, -1)
            x_depth = viplanner.net.encoder_depth(x_depth)

            # encode sem
            x_sem = viplanner.net.encoder_sem(x_sem)

            # concat
            x = torch.cat((x_depth, x_sem), dim=1)  # x.size = (N, 1024, 12, 20)

            if layer == "encoder":
                activation_batches.append(x)
                continue
            else:
                # encode goal
                goal_batch = decoder.fg(goal[batch_start:batch_end, 0:3])
                goal_batch = goal_batch[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
                x = torch.cat((x, goal_batch), dim=1)
                layer_num = int(layer[-1])
                for i in range(layer_num):
                    x = layers[i](x)
                activation_batches.append(x)

    activations = torch.cat(activation_batches, axis=0)
    return activations


def get_activations(cfg: DictConfig, layer: str) -> torch.tensor:
    """
    Compute potentially cached activations at the given layer.

    Args:
        cfg (DictConfig): config file
        layer (str): A string representing the layer, must be one of
            "encoder" or "decoder-n" for an int n in [1, 6].

    Returns:
        torch.tensor: the activations at the layer.
    """
    if os.path.exists(f"checkpoints/{layer}.pt"):
        activations = torch.load(f"checkpoints/{layer}.pt")
    else:
        activations = compute_activations(cfg, layer)
        torch.save(activations, f"checkpoints/{layer}.pt")

    return activations

def compute_distance_features(cfg: DictConfig) -> torch.tensor:
    """
    Compute min distance from goal to obstacles.

    Args:
        cfg (DictConfig): the config file.

    Returns:
        torch.tensor of shape [N, 1]
    """
    sem_handler = VIPlannerSemMetaHandler()
    obstacle_names = []
    for name, loss in sem_handler.class_loss.items():
        if loss == OBSTACLE_LOSS:
            obstacle_names.append(name)

    features = []
    for img_num in range(IMAGE_COUNT):
        print(img_num)
        if img_num == 42:
            img_num = 100 # TEMP: Image 42 is low quality, so swap it for a better image
        # Access configuration parameters
        model_path = cfg.viplanner.model_path
        data_path = cfg.viplanner.data_path
        camera_cfg_path = cfg.viplanner.camera_cfg_path
        point_cloud_path = cfg.viplanner.point_cloud_path
        device = cfg.viplanner.device

        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        sem_handler = VIPlannerSemMetaHandler()

        # Get camera parameters
        cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device=device)
        cam_pos = cam_pos.cpu().numpy().squeeze(0)
        cam_quat = cam_quat.cpu().numpy().squeeze(0)

        # Define camera intrinsics from the camera_intrinsics.txt file in carla folder
        K = np.array([
            [430.69473, 0,        424.0],
            [0,         430.69476, 240.0],
            [0,         0,          1.0]
        ])
        img_width, img_height = 848, 480

        with contextlib.redirect_stdout(None):
            fov_point_cloud = get_points_in_fov_with_intrinsics(
                point_cloud, 
                cam_pos, 
                cam_quat, 
                K,
                img_width, 
                img_height,
                forward_axis="X+",  # Use the detected best axis
                max_distance=100
            )
            goal_point = generate_traversable_goal(
                fov_point_cloud,
                cam_pos,
                sem_handler=sem_handler,
                min_distance=2.0,  # At least 2 meters from camera
                max_distance=10.0  # No more than 10 meters away
            )
            goal_dist_dict = find_distances_in_fov(fov_point_cloud, goal_point, sem_handler)
            cam_dist_dict = find_distances_in_fov(fov_point_cloud, cam_pos, sem_handler)

        goal_obstacle_distances = [goal_dist_dict[name][0] for name in goal_dist_dict if name in obstacle_names]
        cam_obstacle_distances = [cam_dist_dict[name][0] for name in cam_dist_dict if name in obstacle_names]
        goal_min_dist = min(goal_obstacle_distances)
        cam_min_dist = min(cam_obstacle_distances)
        feature = [goal_min_dist, cam_min_dist, np.linalg.norm(goal_point - cam_pos)]
        features.append(torch.tensor(feature).repeat(1, 1))
    features = torch.cat(features, axis=0)
    return features


# This function was generated, in part, using ChatGPT
def compute_semantic_features(rgb_sem: torch.tensor) -> torch.tensor:
    """
    Compute interpretable features from semantic RGB images and metadata.

    Args:
        rgb_sem (torch.Tensor): [N, 3, H, W] RGB-labeled semantic maps

    Returns:
        torch.Tensor of shape [N, F]
    """
    N, _, H, W = rgb_sem.shape
    device = rgb_sem.device

    # Preprocess metadata
    color_to_loss = {}
    color_to_ground = {}
    loss_to_index = {}
    class_colors = []

    for i, meta in enumerate(VIPLANNER_SEM_META):
        color = tuple(meta['color'])
        color_to_loss[color] = meta['loss']
        color_to_ground[color] = meta['ground']
        if meta['loss'] not in loss_to_index:
            loss_to_index[meta['loss']] = len(loss_to_index)
        class_colors.append(color)

    # Convert RGB images to class index maps
    rgb_np = (rgb_sem.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()  # [N, H, W, 3]

    # Init storage
    features = []

    for i in range(N):
        img = rgb_np[i]
        H, W, _ = img.shape
        img_flat = img.reshape(-1, 3)

        total_pixels = H * W
        loss_counts = defaultdict(int)
        ground_count = 0
        class_counts = defaultdict(int)

        for pixel in img_flat:
            pixel_tuple = tuple(pixel.tolist())
            if pixel_tuple in color_to_loss:
                loss_val = color_to_loss[pixel_tuple]
                ground_val = color_to_ground[pixel_tuple]
                loss_counts[loss_val] += 1
                if ground_val:
                    ground_count += 1
                class_counts[pixel_tuple] += 1
            else:
                # Unrecognized pixel class (e.g. black or undefined)
                continue

        # Convert counts to normalized proportions
        loss_props = [loss_counts.get(loss, 0) / total_pixels for loss in sorted(loss_to_index)]
        ground_prop = ground_count / total_pixels
        nonground_prop = 1.0 - ground_prop

        # Ratios
        ratios = []
        if loss_counts.get(0, 0) > 0:  # avoid divide-by-zero
            ratios.append(loss_counts.get(2.0, 0) / loss_counts[0])  # obstacle / traversable_intended
            ratios.append(loss_counts.get(1.5, 0) / loss_counts[0])  # road / traversable_intended
        else:
            ratios += [0.0, 0.0]

        if nonground_prop > 0:
            ratios.append(ground_prop / nonground_prop)
        else:
            ratios.append(0.0)

        # Entropy over class counts
        class_freqs = np.array(list(class_counts.values()), dtype=np.float32)
        class_probs = class_freqs / class_freqs.sum() if class_freqs.sum() > 0 else class_freqs
        class_entropy = entropy(class_probs + 1e-8)

        # Dominant class loss
        if class_counts:
            dominant_class = max(class_counts, key=class_counts.get)
            dominant_loss = color_to_loss[dominant_class]
        else:
            dominant_loss = 0.0

        # Final feature vector for image i
        features.append(loss_props + [ground_prop, nonground_prop] + ratios + [class_entropy, dominant_loss])

    features_tensor = torch.tensor(features, dtype=torch.float32, device=device)  # [N, F]
    return features_tensor


# This function was generated, in part, using ChatGPT
def compute_depth_features(depth_batch: torch.tensor, num_hist_bins: int = 20) -> torch.tensor:
    """
    Compute rich scalar features from a batch of depth images.
    
    Args:
        depth_batch (torch.Tensor): shape [N, 1, H, W]
        num_hist_bins (int): number of bins for histogram entropy
    
    Returns:
        torch.Tensor of shape [N, F]
    """
    N, _, H, W = depth_batch.shape
    device = depth_batch.device

    # Flatten per image for stat ops
    flat = depth_batch.view(N, -1)

    # Convert to numpy for some stats
    flat_np = flat.detach().cpu().numpy()

    # 1. Global stats
    mean = flat.mean(dim=1)
    std = flat.std(dim=1)
    min_ = flat.min(dim=1)[0]
    max_ = flat.max(dim=1)[0]
    
    # skew/kurtosis
    skewness = torch.tensor([skew(d) for d in flat_np], dtype=torch.float32, device=device)
    kurt = torch.tensor([kurtosis(d) for d in flat_np], dtype=torch.float32, device=device)

    # 2. Quantiles
    quantiles = torch.tensor(np.quantile(flat_np, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1),
                             dtype=torch.float32, device=device).T  # shape [N, 5]

    # 3. Histogram entropy
    hist_entropy = []
    for d in flat_np:
        hist, _ = np.histogram(d, bins=num_hist_bins, density=True)
        hist_entropy.append(entropy(hist + 1e-8))  # add epsilon for stability
    hist_entropy = torch.tensor(hist_entropy, dtype=torch.float32, device=device)

    # 4. Gradient features (Sobel filter)
    sobel_x = torch.tensor([[[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]], dtype=torch.float32, device=device).unsqueeze(0)
    sobel_y = torch.tensor([[[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]]], dtype=torch.float32, device=device).unsqueeze(0)

    grad_x = F.conv2d(depth_batch, sobel_x, padding=1)
    grad_y = F.conv2d(depth_batch, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)

    grad_x_mean = grad_x.view(N, -1).mean(dim=1)
    grad_x_std = grad_x.view(N, -1).std(dim=1)
    grad_y_mean = grad_y.view(N, -1).mean(dim=1)
    grad_y_std = grad_y.view(N, -1).std(dim=1)
    grad_mag_mean = grad_mag.view(N, -1).mean(dim=1)
    grad_mag_std = grad_mag.view(N, -1).std(dim=1)

    # Combine all features
    features = torch.stack([
        mean, std, min_, max_,
        skewness, kurt,
        *quantiles.T,
        hist_entropy,
        grad_x_mean, grad_x_std,
        grad_y_mean, grad_y_std,
        grad_mag_mean, grad_mag_std
    ], dim=1)  # shape [N, F]

    return features


def get_features(cfg: DictConfig, feature_set: str) -> Tuple[torch.tensor, List[str]]:
    """
    Compute potentially cached features of a given type.

    Args:
        cfg (DictConfig): config file
        feature_set (str): the type of feature to load, can currently be either
            "simple_depth", "simple_semantic", or "distance".

    Returns:
        torch.tensor: the features in question.
        List[str]: the names of the features (has length equal to dimension 1 of above tensor).
    """
    feature_functions = {
        "simple_depth": lambda cfg: compute_depth_features(get_image_batches(cfg)[0]),
        "simple_semantic": lambda cfg: compute_semantic_features(get_image_batches(cfg)[1]),
        "distance": compute_distance_features,
    }

    feature_names = {
        "simple_depth": [
            "mean", "std", "min", "max", "skewness", "kurt", "0.1 quantile", 
            "0.25 quantile", "0.5 quantile", "0.75 quantile", "0.9 quantile", 
            "entropy", "grad x mean", "grad x std", "grad y mean", 
            "grad y std", "grad mag mean", "grad mag std",
        ],
        "simple_semantic": [
            "traversable intended prop", "traversable unintended prop", "terrain prop",
            "road prop", "obstacle prop", "ground prop", "nonground prop", "obstacle vs traversable ratio",
            "road vs traversable ratio", "ground vs nonground ratio", "semantic entropy", "dominant loss",
        ],
        "distance": [
            "goal distance from nearest obstacle", "camera distance from nearest obstacle", 
            "distance from camera to goal"
        ],
    }

    if os.path.exists(f"checkpoints/{feature_set}.pt"):
        features = torch.load(f"checkpoints/{feature_set}.pt")
    else:
        features = feature_functions[feature_set](cfg)
        torch.save(features, f"checkpoints/{feature_set}.pt")

    return features, feature_names[feature_set]


# This function was generated, in part, using ChatGPT
def train_test_split_tensors(
    pooled: torch.Tensor, 
    hand_features: torch.Tensor, 
    test_size: float = 0.2, 
    seed: int = 42,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    pooled_np = pooled.cpu().numpy()
    hand_np = hand_features.cpu().numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        pooled_np, hand_np, test_size=test_size, random_state=seed
    )

    X_train = torch.tensor(X_train, dtype=pooled.dtype)
    X_test = torch.tensor(X_test, dtype=pooled.dtype)
    y_train = torch.tensor(y_train, dtype=hand_features.dtype)
    y_test = torch.tensor(y_test, dtype=hand_features.dtype)

    return X_train, X_test, y_train, y_test


# This function was generated, in part, using ChatGPT
def visualize_lin_reg_weights(reg: LinearRegression, feature_name: str, layer=None):
    reshapings = {
        1024: (32, 32),
        512: (16, 32),
        256: (16, 16),
        15: (5, 3),
    }
    weights = np.asarray(reg.coef_)
    heatmap = weights.reshape(reshapings[weights.shape[0]])
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="viridis")
    plt.colorbar(label="Weight Magnitude")
    plt.title(f"Linear Regression weights by channel\nfor {feature_name} in {layer} layer")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"plots/{feature_name}_{layer}_weights.png", dpi=300)


def visualize_pca_and_tsne(
    pooled: torch.tensor, 
    gen_features: torch.tensor, 
    feature_names: List[str],
    n_components: int = 2,
    perplexity: int = 30,
):
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(pooled.cpu().numpy())

    tsne = TSNE(n_components=2, perplexity=30)
    tsne_proj = tsne.fit_transform(pooled.cpu().numpy())
    
    for gen_feature in range(gen_features.shape[1]):
        feature_name = feature_names[gen_feature]
        plt.scatter(pca_proj[:, 0], pca_proj[:, 1], c=gen_features[:, gen_feature])
        plt.title(f"PCA of encoder features, labeled by {feature_name}")
        plt.show()

        plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=gen_features[:, gen_feature])
        plt.title(f"TSNE of encoder features, labeled by {feature_name}")
        plt.show()


def find_top_k_weights_indices(weights, k=10):
    """
    Find the indices of the top k weights in a 1D array.

    Args:
        weights (np.ndarray): 1D array of weights
        k (int): number of top weights to find

    Returns:
        np.ndarray: indices of the top k weights
    """
    return np.argsort(weights)[-k:][::-1]

def run_linear_probing(
    pooled: torch.tensor,
    gen_features: torch.tensor,
    feature_names: List[str],
    skip_features: Optional[List[str]] = None,
    test_size: float = 0.2,
    seed: int = 42,
    visualize_weights: bool = False,
    find_top_k_weights: bool = True,
    layer=None,
):
    X_train, X_test, y_train, y_test = train_test_split_tensors(pooled, gen_features, test_size=test_size, seed=seed)

    for gen_feature in range(gen_features.shape[1]):
        # Skip NaN features
        feature_name = feature_names[gen_feature]
        if skip_features and feature_name in skip_features:
            continue
        reg = LinearRegression()
        reg.fit(X_train.cpu().numpy(), y_train[:, gen_feature].cpu().numpy())
        score = reg.score(X_test.cpu().numpy(), y_test[:, gen_feature].cpu().numpy())
        print(f"R² score for {feature_name}:", score)
        if visualize_weights:
            visualize_lin_reg_weights(reg, feature_name, layer=layer)
        if find_top_k_weights:
            weights = np.abs(reg.coef_)
            weights_file = f"data/{feature_name}_{layer}_weights.pkl"
            with open(weights_file, "wb") as f:
                pickle.dump(weights, f)
            top_k_indices = find_top_k_weights_indices(weights, k=800)
            file_name = f"data/{feature_name}_{layer}_top_{len(top_k_indices)}_weights.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(top_k_indices, f)
            print(f"Top {len(top_k_indices)} weights for {feature_name} saved to {file_name}")

def run_analysis(
    cfg: DictConfig, 
    layer: str, 
    feature_set: str,
    analysis_types: List[str]
):
    """
    Performs analysis on a given layer and feature set

    Args:
        cfg (DictConfig): the config file.
        layer (str): name of the layer to analyze.
        feature_set (str): name of the feature set to analyze.
        analysis_types (List[str]): list of types of analysis to perform, can include
            "pca_tsne" and "linear_probing".
    """
    activations = get_activations(cfg, layer)
    features, feature_names = get_features(cfg, feature_set)

    print("activations shape", activations.shape)
    print("features shape", features.shape)

    if activations.ndim == 2:
        pooled = activations
    elif activations.ndim == 3:
        pooled = activations.view(activations.size(0), -1)
    else:
        pooled = activations.mean(dim=(-2, -1))

    print("pooled shape", pooled.shape)

    if "pca_tsne" in analysis_types:
        visualize_pca_and_tsne(pooled, features, feature_names)

    if "linear_probing" in analysis_types:
        run_linear_probing(
            pooled, 
            features, 
            feature_names, 
            skip_features=["skewness", "kurt"], 
            visualize_weights=True,
            layer=layer
        )


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def analyze(cfg: DictConfig):
    layers = ["encoder", "decoder-1", "decoder-2", "decoder-3", "decoder-4", "decoder-5"]
    features = ["simple_depth", "simple_semantic", "distance"]
    for layer in layers:
        for feature in features:
            print(f"Layer: {layer}, feature: {feature}")
            run_analysis(
                cfg,
                layer=layer,
                feature_set=feature,
                analysis_types=["pca_tsne", "linear_probing"],
            )
    

if __name__ == '__main__':
    analyze()
