from PIL import Image
import numpy as np
import os
import hydra
from omegaconf import DictConfig

# Target class to filter by
TARGET_CLASS = "building"
MIN_PERCENTAGE = 0.1  # i.e. 40%

# Class mappings
OBSTACLE_LOSS = 2.0
TRAVERSABLE_INTENDED_LOSS = 0
TRAVERSABLE_UNINTENDED_LOSS = 0.5
ROAD_LOSS = 1.5
TERRAIN_LOSS = 1.0
# NOTE: only obstacle loss should be over obscale_loss defined in costmap_cfg.py

# original coco meta
VIPLANNER_SEM_META = [
    # TRAVERSABLE SPACE ###
    # traversable intended
    {
        "name": "sidewalk",
        "loss": TRAVERSABLE_INTENDED_LOSS,
        "color": [0, 255, 0],
        "ground": True,
    },
    {
        "name": "crosswalk",
        "loss": TRAVERSABLE_INTENDED_LOSS,
        "color": [0, 102, 0],
        "ground": True,
    },
    {
        "name": "floor",
        "loss": TRAVERSABLE_INTENDED_LOSS,
        "color": [0, 204, 0],
        "ground": True,
    },
    {
        "name": "stairs",
        "loss": TRAVERSABLE_INTENDED_LOSS,
        "color": [0, 153, 0],
        "ground": True,
    },
    # traversable not intended
    {
        "name": "gravel",
        "loss": TRAVERSABLE_UNINTENDED_LOSS,
        "color": [204, 255, 0],
        "ground": True,
    },
    {
        "name": "sand",
        "loss": TRAVERSABLE_UNINTENDED_LOSS,
        "color": [153, 204, 0],
        "ground": True,
    },
    {
        "name": "snow",
        "loss": TRAVERSABLE_UNINTENDED_LOSS,
        "color": [204, 102, 0],
        "ground": True,
    },
    {
        "name": "indoor_soft",  # human made thing, can be walked on
        "color": [102, 153, 0],
        "loss": TERRAIN_LOSS,
        "ground": False,
    },
    {
        "name": "terrain",
        "color": [255, 255, 0],
        "loss": TERRAIN_LOSS,
        "ground": True,
    },
    {
        "name": "road",
        "loss": ROAD_LOSS,
        "color": [255, 128, 0],
        "ground": True,
    },
    # OBSTACLES ###
    # human
    {
        "name": "person",
        "color": [255, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "anymal",
        "color": [204, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # vehicle
    {
        "name": "vehicle",
        "color": [153, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "on_rails",
        "color": [51, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "motorcycle",
        "color": [102, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "bicycle",
        "color": [102, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # construction
    {
        "name": "building",
        "loss": OBSTACLE_LOSS,
        "color": [127, 0, 255],
        "ground": False,
    },
    {
        "name": "wall",
        "color": [102, 0, 204],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "fence",
        "color": [76, 0, 153],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "bridge",
        "color": [51, 0, 102],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "tunnel",
        "color": [51, 0, 102],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # object
    {
        "name": "pole",
        "color": [0, 0, 255],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "traffic_sign",
        "color": [0, 0, 153],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "traffic_light",
        "color": [0, 0, 204],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "bench",
        "color": [0, 0, 102],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # nature
    {
        "name": "vegetation",
        "color": [153, 0, 153],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "water_surface",
        "color": [204, 0, 204],
        "loss": OBSTACLE_LOSS,
        "ground": True,
    },
    # sky
    {
        "name": "sky",
        "color": [102, 0, 51],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "background",
        "color": [102, 0, 51],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # void outdoor
    {
        "name": "dynamic",
        "color": [32, 0, 32],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "static",  # also everything unknown
        "color": [0, 0, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    # indoor
    {
        "name": "furniture",
        "color": [0, 0, 51],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "door",
        "color": [153, 153, 0],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
    {
        "name": "ceiling",
        "color": [25, 0, 51],
        "loss": OBSTACLE_LOSS,
        "ground": False,
    },
]

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def filter_images(cfg: DictConfig):
    
    DATA_PATH = cfg.viplanner.data_path

    # Build color-to-class mapping
    COLOR_TO_CLASS = {tuple(meta["color"]): meta["name"] for meta in VIPLANNER_SEM_META}

    SEMANTIC_DIR = f"{DATA_PATH}/semantics"
    selected_images = []

    for fname in os.listdir(SEMANTIC_DIR):
        if not fname.endswith(".png"):
            continue

        path = os.path.join(SEMANTIC_DIR, fname)
        img = np.array(Image.open(path).convert("RGB"))
        H, W, _ = img.shape
        flat_img = img.reshape(-1, 3)
        num_pixels = flat_img.shape[0]
        sample_size = min(5000, num_pixels)  # sample up to 5,000 pixels
        indices = np.random.choice(num_pixels, sample_size, replace=False)
        sampled_pixels = flat_img[indices]

        # Count how many pixels belong to the target class
        total_pixels = sampled_pixels.shape[0]
        target_count = 0

        for rgb in sampled_pixels:
            class_name = COLOR_TO_CLASS.get(tuple(rgb), "unknown")
            if class_name == TARGET_CLASS:
                target_count += 1

        percent = target_count / total_pixels
        if percent >= MIN_PERCENTAGE:
            selected_images.append(fname)

    print(f"Found {len(selected_images)} images with mostly '{TARGET_CLASS}'")


    with open(f"mostly_{TARGET_CLASS}.txt", "w") as f:
        for name in selected_images:
            f.write(name + "\n")

if __name__ == '__main__':
    filter_images()
