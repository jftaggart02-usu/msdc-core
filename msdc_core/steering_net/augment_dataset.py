"""This module contains code to augment a dataset for training the steering net.

Note: negative steering angle is a left turn, positive steering angle is a right turn.
"""

import math
from typing import Any
from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
from msdc_core.steering_net.dataset_utils import copy_dataset, normalize_index_column
from msdc_core.steering_net.perspective_transform import create_transform_matrix, perspective_transform, load_intrinsic_json


@dataclass
class AugmentationConfig:
    """Parameters for dataset augmentation."""

    max_rotation: float = 5.0  # degrees
    max_translation: float = 0.1  # meters
    steering_translation_factor: float = (
        0.5  # how much to adjust the steering angle based on the translation (steering angle change (degrees) per meter of translation)
    )
    steering_rotation_factor: float = (
        1.0  # how much to adjust the steering angle based on the rotation (steering angle change (degrees) per degree of rotation)
    )
    percent_samples_to_augment: float = 1.0  # percentage of samples to augment [0-1]
    do_rotation: bool = True  # whether to apply random rotations
    do_translation: bool = True  # whether to apply random translations

    def __post_init__(self):
        if not (0.0 <= self.percent_samples_to_augment <= 1.0):
            raise ValueError("percent_samples_to_augment must be between 0 and 1.")


def augment_dataset(dataset_dir: str, target_dir: str, config: AugmentationConfig) -> None:
    """Augment the dataset by applying random transformations to a percentage of samples.

    Args:
        dataset_dir: Path to the original dataset directory.
        target_dir: Path to the directory where the augmented dataset will be saved.
        config: AugmentationConfig object containing augmentation parameters.
    """
    # First, copy the original dataset to the target directory
    copy_dataset(dataset_dir, target_dir, copy_depth=False)
    df = pd.read_csv(f"{target_dir}/labels.csv")
    normalize_index_column(df, width=8)

    # Randomly choose samples to augment based on the specified percentage
    num_samples_to_augment = int(len(df) * config.percent_samples_to_augment)
    if num_samples_to_augment < 1:
        print("No samples to augment based on the specified percentage. Exiting augmentation.")
        return
    print(f"Augmenting {num_samples_to_augment} samples out of {len(df)} total samples.")
    samples_to_augment = df.sample(n=num_samples_to_augment, random_state=42)

    # Determine the starting index for new augmented samples by finding the maximum existing index and adding 1
    int_indices = [int(index) for index in df["index"]]
    start_index = max(int_indices) + 1
    current_index = start_index

    # Loop through the samples to augment, apply random transformations, and save the augmented samples
    intrinsic = load_intrinsic_json(f"{dataset_dir}/intrinsics.json")
    new_rows: list[dict[str, Any]] = []
    for _, row in samples_to_augment.iterrows():
        timestamp = row["timestamp"]
        index = row["index"]
        steering_angle_rad = row["steering_angle_rad"]
        current_index_str = str(current_index).zfill(8)

        # Generate random rotation and translation values within the specified limits
        if config.do_rotation:
            rotation_y = random.uniform(-config.max_rotation, config.max_rotation)
            rotation_vector = np.array([0.0, rotation_y, 0.0])
        else:
            rotation_vector = np.array([0.0, 0.0, 0.0])
        if config.do_translation:
            translation_x = random.uniform(-config.max_translation, config.max_translation)
            translation_vector = np.array([translation_x, 0.0, 0.0])
        else:
            translation_vector = np.array([0.0, 0.0, 0.0])

        # Transform the images using the random rotation and translation, and save the augmented images
        transform_matrix = create_transform_matrix(rotation_vector, translation_vector)
        perspective_transform(
            rgb_path=f"{dataset_dir}/rgb/{index}.png",
            depth_path=f"{dataset_dir}/depth/{index}.png",
            dest_path=f"{target_dir}/rgb/{current_index_str}.png",
            intrinsic=intrinsic,
            transformation_matrix=transform_matrix,
            inpaint=True,
        )

        # Adjust the steering angle based on the applied transformation
        steering_angle_deg = math.degrees(steering_angle_rad)
        if config.do_translation:
            steering_angle_deg += translation_x * config.steering_translation_factor
        if config.do_rotation:
            steering_angle_deg += rotation_y * config.steering_rotation_factor

        # Save the new sample information to add to the augmented dataframe later
        new_rows.append(
            {
                "timestamp": timestamp,
                "index": current_index_str,
                "steering_angle_rad": math.radians(steering_angle_deg),
            }
        )

        current_index += 1

        print(f"Augmented sample {current_index - start_index} of {num_samples_to_augment}")

    # Add the new rows to the augmented dataframe and save the updated labels.csv
    new_rows_df = pd.DataFrame(new_rows)
    new_rows_df.to_csv(f"{target_dir}/labels.csv", mode="a", index=False, header=False)


def main() -> None:
    """Example usage of augment_dataset function."""
    dataset_dir = "/home/jftaggart02/datasets/trial_03_balanced"
    target_dir = "/home/jftaggart02/datasets/trial_03_augmented"
    config = AugmentationConfig(
        percent_samples_to_augment=0.50, max_translation=0.15, steering_translation_factor=0.5, do_rotation=False
    )
    augment_dataset(dataset_dir, target_dir, config)


if __name__ == "__main__":
    main()
