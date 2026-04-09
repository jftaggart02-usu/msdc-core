"""This module implements a PyTorch Dataset for the steering network.

The dataset is of the format:
```
dataset/
    rgb/
        00000000.png
        00000001.png
        ...
    intrinsics.json
    labels.csv
```
"""

from pathlib import Path

from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as transforms
from msdc_core.steering_net.dataset_utils import validate_dataset_structure, normalize_index_column


class SteeringDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str) -> None:

        # Ensure dataset_dir exists and has the expected structure (must have labels.csv, intrinsics.json, and rgb/ subdirectory)
        validate_dataset_structure(dataset_dir, validate_depth=False)
        self.dataset_dir = Path(dataset_dir)

        # Load labels.csv into a DataFrame
        self.labels = pd.read_csv(Path(dataset_dir) / "labels.csv")
        normalize_index_column(self.labels)

        # Define transforms (Downsample by a factor of 2 and convert to tensor)
        self.transforms = transforms.Compose([transforms.Resize((240, 320)), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the RGB image and steering angle for the given index."""
        # Get the row corresponding to the index
        row = self.labels.iloc[idx]

        # Load the RGB image
        rgb_path = self.dataset_dir / "rgb" / f"{row['index']}.png"
        if not rgb_path.exists():
            raise ValueError(f"RGB image not found at path: {rgb_path}")
        image_tensor = self.image_to_tensor(rgb_path)

        # Get the steering angle
        steering_angle = row["steering_angle_rad"]

        return image_tensor, torch.Tensor([steering_angle])

    def image_to_tensor(self, image_path: Path) -> torch.Tensor:
        """Loads an image from the given path and converts it to a normalized PyTorch tensor of shape (C, H, W),
        where C is the number of channels (3 for RGB), H is the height, and W is the width.
        The pixel values are normalized to be in the range [0, 1].
        """
        img = Image.open(image_path).convert("RGB")

        img_tensor: torch.Tensor = self.transforms(img)
        return img_tensor.float() / 255.0
