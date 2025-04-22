import numpy as np
import torch

def load_camera_extrinsics(filepath, picture_number, device="cpu"):
    """
    Returns the camera position and quaternion for a given picture number as PyTorch tensors.

    Parameters:
        filepath (str): Path to the text file.
        picture_number (int): Line number (0-indexed) of the image.
        device (str): The device to which the tensors should be moved (e.g., 'cpu', 'cuda').

    Returns:
        pos (torch.Tensor): Camera position (3,) as a PyTorch tensor.
        quat (torch.Tensor): Camera orientation quaternion (x, y, z, w) (4,) as a PyTorch tensor.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if picture_number < 0 or picture_number >= len(lines):
        raise IndexError("Picture number out of range.")

    line = lines[picture_number].strip()
    values = list(map(float, line.split(',')))

    if len(values) != 7:
        raise ValueError(f"Expected 7 values per line, got {len(values)}.")

    pos = torch.tensor(values[:3], device=device, dtype=torch.float32)  # x, y, z
    quat = torch.tensor(values[3:], device=device, dtype=torch.float32)  # x, y, z, w
    pos = pos.reshape(1, 3)
    quat = quat.reshape(1, 4)

    return pos, quat