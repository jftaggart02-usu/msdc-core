"""This module defines the SteeringDataset class, a PyTorch Dataset for loading images and steering angles for training the SteeringNet neural network."""

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SteeringDataset(Dataset):
    """PyTorch Dataset for steering angle prediction from images.

    Loads images from a folder and their corresponding steering angles from a CSV file.

    Dataset directory structure:
    ```
    dataset_dir/
        images/
            000000.png
            000001.png
            ...
        labels.csv
    ```

    labels.csv format:
    ```
    index,timestamp,steering
    1,1697051234.123456,0.034
    2,1697051234.156789,0.041
    ...
    ```

    Args:
        dataset_dir: Path to the dataset directory containing 'images/' and 'labels.csv'.
        transform: Optional torchvision transforms to apply to images.
        image_size: Target image size (height, width). If None, no resizing is applied.

    Attributes:
        labels: DataFrame containing the steering labels.
        image_dir: Path to the images directory.

    Examples:
        >>> dataset = SteeringDataset('/path/to/dataset/')
        >>> image, steering = dataset[0]
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    """

    def __init__(
        self,
        dataset_dir: str,
        transform: Callable | None = None,
        image_size: tuple | None = None,
        color_space: str = "RGB",
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset_dir: Path to dataset directory.
            transform: Custom transforms to apply.
            image_size: Resize images to (height, width).
            color_space: Color space to use ("RGB" or "YCbCr"). YCbCr is PIL's YUV equivalent.
        """
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        self.labels_path = self.dataset_dir / "labels.csv"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.image_dir}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels CSV not found: {self.labels_path}")

        # Load labels
        self.labels = pd.read_csv(self.labels_path)
        self.image_size = image_size
        self.color_space = color_space

        # Build transform pipeline
        if transform is None:
            transform_list = [transforms.ToTensor()]
            if image_size:
                transform_list.insert(0, transforms.Resize(image_size))
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        """Get a sample by index.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (image after transforms are applied, steering angle tensor).
        """
        row = self.labels.iloc[idx]
        image_idx = int(row["index"])
        steering = float(row["steering"])

        # Load image
        image_path = self.image_dir / f"{image_idx:06d}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert(self.color_space)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(steering, dtype=torch.float32)
