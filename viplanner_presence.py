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

color_to_idx = {}
for idx, entry in enumerate(VIPLANNER_SEM_META):
    color = tuple(entry['color'])  # [R, G, B] -> (R, G, B) tuple
    color_to_idx[color] = idx

# Step 2: Load the segmented image
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    return img_array

# Step 3: Create the one-hot presence vector
def create_presence_vector(img_array, color_to_idx, num_classes):
    presence_vector = np.zeros(num_classes, dtype=np.uint8)
    
    # Reshape to list of pixels
    pixels = img_array.reshape(-1, 3)
    
    # For faster check, build a set of unique colors
    unique_colors = np.unique(pixels, axis=0)
    
    for color in unique_colors:
        color_tuple = tuple(color)
        if color_tuple in color_to_idx:
            idx = color_to_idx[color_tuple]
            presence_vector[idx] = 1
    
    return presence_vector

if __name__ == '__main__':
    # Step 4: Example usage
    image_path = "./samples/semantics/0002.png"
    img_array = load_image(image_path)
    presence_vector = create_presence_vector(img_array, color_to_idx, num_classes=len(VIPLANNER_SEM_META))

    # Optional: print the classes present
    print(presence_vector)
    for idx, present in enumerate(presence_vector):
        if present:
            print(f"Class present: {VIPLANNER_SEM_META[idx]['name']}")

