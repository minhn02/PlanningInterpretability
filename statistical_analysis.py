import torch
import viplanner_wrapper
import hydra
import os
from omegaconf import DictConfig

from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F
from scipy.stats import skew, kurtosis, entropy
import numpy as np
import matplotlib.pyplot as plt

from viplanner.viplanner.config.viplanner_sem_meta import VIPLANNER_SEM_META


IMAGE_COUNT = 1000
BATCH_SIZE = 25


def add_featurewise_noise(X, scale=0.1):
    """
    X: [N, D] torch.Tensor
    scale: scalar float, how much of each feature's std to use as noise std
    """
    stds = X.std(dim=0, keepdim=True)  # [1, D]
    noise = torch.randn_like(X) * stds * scale
    return X + noise


def get_image_batches(cfg: DictConfig):
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    device = cfg.viplanner.device

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device)

    images = range(IMAGE_COUNT)
    depth_images = []
    sem_images = []
    for n in images:
        depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, n, device)
        sem_image = viplanner.transform(sem_image) / 255
        depth_image = viplanner.input_transformer(depth_image)
        depth_images.append(depth_image)
        sem_images.append(sem_image)
    depth_img = torch.cat(depth_images, axis=0)
    sem_img = torch.cat(sem_images, axis=0)
    return depth_img, sem_img


# This function was generated, in part, using ChatGPT
def compute_semantic_features(rgb_sem):
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
def compute_depth_features(depth_batch, num_hist_bins=20):
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


def get_features(cfg: DictConfig):
    depth_img, sem_img = get_image_batches(cfg)
    print("processed depth size", depth_img.shape)
    print("processed sem size", sem_img.shape)

    features_batches = []
    for i in range(IMAGE_COUNT // BATCH_SIZE):
        with torch.no_grad():
            # encode depth
            batch_start = i * BATCH_SIZE
            batch_end = (i+1) * BATCH_SIZE
            x_depth = depth_img[batch_start:batch_end, :, :, :]
            x_sem = sem_img[batch_start:batch_end, :, :, :]
            x_depth = x_depth.expand(-1, 3, -1, -1)
            x_depth = viplanner.net.encoder_depth(x_depth)
            # encode sem
            x_sem = viplanner.net.encoder_sem(x_sem)
            # concat
            features = torch.cat((x_depth, x_sem), dim=1)  # x.size = (N, 1024, 12, 20)
            features_batches.append(features)
    features = torch.cat(features_batches, axis=0)
    return features


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def analyze(cfg: DictConfig):
    # Compute features from embeddings and from hand analysis
    if os.path.exists("checkpoints/features.pt"):
        features = torch.load("checkpoints/features.pt")
    else:
        features = get_features(cfg)
        torch.save(features, "checkpoints/features.pt")

    if os.path.exists("checkpoints/depth_features.pt"):
        depth_features = torch.load("checkpoints/depth_features.pt")
    else:
        depth_batch, sem_batch = get_image_batches(cfg)
        depth_features = compute_depth_features(depth_batch)
        torch.save(depth_features, "checkpoints/depth_features.pt")

    if os.path.exists("checkpoints/sem_features.pt"):
        sem_features = torch.load("checkpoints/sem_features.pt")
    else:
        depth_batch, sem_batch = get_image_batches(cfg)
        sem_features = compute_semantic_features(sem_batch)
        torch.save(sem_features, "checkpoints/sem_features.pt")

    print("depth features shape", depth_features.shape)
    print("sem features shape", sem_features.shape)
    print("features shape", features.shape)
    pooled = features.mean(dim=(-2, -1))
    print("pooled shape", pooled.shape)

    # PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(pooled.cpu().numpy())

    # TSNE
    tsne = TSNE(n_components=2, perplexity=30)
    tsne_proj = tsne.fit_transform(pooled.cpu().numpy())

    # Plotting over generated features
    generated_features = torch.cat((depth_features, sem_features), axis=1)
    _, gen_feature_count = generated_features.shape
    depth_feature_names = [
        "mean", "std", "min", "max", "skewness", "kurt", "0.1 quantile", 
        "0.25 quantile", "0.5 quantile", "0.75 quantile", "0.9 quantile", 
        "entropy", "grad x mean", "grad x std", "grad y mean", 
        "grad y std", "grad mag mean", "grad mag std",
    ]
    sem_feature_names = [
        "traversable intended prop", "traversable unintended prop", "terrain prop",
        "road prop", "obstacle prop", "ground prop", "nonground prop", "obstacle vs traversable ratio",
        "road vs traversable ratio", "ground vs nonground ratio", "semantic entropy", "dominant loss"
    ]
    gen_feature_names = ["depth " + dfn for dfn in depth_feature_names] + sem_feature_names
    """
    noisy_features_predict = add_featurewise_noise(generated_features, scale=0.25)
    noisy_features_score = add_featurewise_noise(generated_features, scale=0.25)
    """
    for gen_feature in range(gen_feature_count):
        plt.scatter(projected[:, 0], projected[:, 1], c=generated_features[:, gen_feature])
        plt.title(f"PCA of encoder features, labeled by {gen_feature_names[gen_feature]}")
        plt.show()

        plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=generated_features[:, gen_feature])
        plt.title(f"TSNE of encoder features, labeled by {gen_feature_names[gen_feature]}")
        plt.show()

        """
        # Note: Linear Probing currently overfitting, more samples needed.
        # Skip NaN features
        if gen_feature_names[gen_feature] in ["depth skewness", "depth kurt"]:
            continue
        reg = LinearRegression()
        reg.fit(pooled.cpu().numpy(), noisy_features_predict[:, gen_feature].cpu().numpy())
        score = reg.score(pooled.cpu().numpy(), noisy_features_score[:, gen_feature].cpu().numpy())
        print(f"R² score for {gen_feature_names[gen_feature]}:", score)
        """


if __name__ == '__main__':
    analyze()
