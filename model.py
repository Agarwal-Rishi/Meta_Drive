import numpy as np
import torch
import torch.nn as nn


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DrivingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),   # 112 → 56
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56 → 28
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 28 → 14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 14 → 7
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.features(x)


def extract_image(observation):
    """Pull the latest RGB frame from a MetaDrive observation."""
    if isinstance(observation, dict):
        image = observation["image"]
    else:
        image = observation

    # MetaDrive stacks frames on the last axis: (H, W, C, stack)
    if image.ndim == 4:
        image = image[..., -1]
    return np.asarray(image)


def image_to_tensor(image, device=None):
    """Convert one HxWxC image to a batch tensor (1, C, H, W)."""
    tensor = torch.as_tensor(image, dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape {tuple(tensor.shape)}")
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def images_to_tensor(images):
    """Convert a list/array of HxWxC images to (N, C, H, W)."""
    array = np.asarray(images, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"Expected NxHxWxC images, got shape {tuple(array.shape)}")
    return torch.from_numpy(array).permute(0, 3, 1, 2)
