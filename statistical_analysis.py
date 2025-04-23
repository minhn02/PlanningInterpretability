import torch
import viplanner_wrapper
import hydra
import os
from omegaconf import DictConfig

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import skew, kurtosis, entropy
import numpy as np


IMAGE_COUNT = 1000
BATCH_SIZE = 25


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

    print("depth features shape", features.shape)
    print("features shape", features.shape)
    pooled = features.mean(dim=(-2, -1))
    print("pooled shape", pooled.shape)

    # PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(pooled.cpu().numpy())

    # TSNE
    tsne = TSNE(n_components=2, perplexity=30)
    tsne_proj = tsne.fit_transform(pooled.cpu().numpy())

    # Plotting over depth features
    _, depth_feature_count = depth_features.shape
    depth_feature_names = [
        "mean", "std", "min", "max", "skewness", "kurt", "0.1 quantile", 
        "0.25 quantile", "0.5 quantile", "0.75 quantile", "0.9 quantile", 
        "entropy", "grad x mean", "grad x std", "grad y mean", 
        "grad y std", "grad mag mean", "grad mag std",
    ]
    for depth_feature in range(depth_feature_count):
        plt.scatter(projected[:, 0], projected[:, 1], c=depth_features[:, depth_feature])
        plt.title(f"PCA of encoder features, labeled by {depth_feature_names[depth_feature]}")
        plt.show()

        plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=depth_features[:, depth_feature])
        plt.title(f"TSNE of encoder features, labeled by {depth_feature_names[depth_feature]}")
        plt.show()


if __name__ == '__main__':
    analyze()
