"""This module contains code that balances a dataset by removing samples with certain steering angles.

The dataset is of the format:
```
dataset/
    rgb/
        00000000.png
        00000001.png
        ...
    depth/
        00000000.png
        00000001.png
        ...
    intrinsics.json
    labels.csv
```

Where labels.csv has columns: `timestamp`, `index`, and `steering_angle_rad`.
"""

from pathlib import Path
import shutil

import pandas as pd
import matplotlib.pyplot as plt


def plot_histogram_of_steering_angles(df: pd.DataFrame, bins: int, show_plot: bool) -> None:
    """Plots a histogram of the steering angles in the dataset.

    Args:
        df: The DataFrame containing the steering angle data.
        bins: The number of bins for the histogram.
        show_plot: Whether to display the plot.
    """
    # Plot the histogram of steering angles
    df["steering_angle_rad"].hist(bins=bins)
    plt.xlabel("Steering Angle (radians)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Steering Angles")
    if show_plot:
        plt.show()


def validate_dataset_structure(dataset_dir: str) -> None:
    """Validates that the dataset directory has the expected structure.

    Args:
        dataset_dir: The directory containing the dataset.

    Raises:
        ValueError: If the dataset directory does not have the expected structure.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise ValueError("Dataset directory does not exist.")
    if not (dataset_path / "labels.csv").exists():
        raise ValueError("Dataset must contain labels.csv.")
    if not (dataset_path / "intrinsics.json").exists():
        raise ValueError("Dataset must contain intrinsics.json.")
    if not (dataset_path / "depth").is_dir():
        raise ValueError("Dataset must contain depth/ subdirectory.")
    if not (dataset_path / "rgb").is_dir():
        raise ValueError("Dataset must contain rgb/ subdirectory.")


def normalize_index(index: object, width: int = 8) -> str:
    """Normalizes the index value to a zero-padded string of the specified width."""

    if pd.isna(index):
        raise ValueError("Index cannot be NaN.")
    s = str(index).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if not s.isdigit():
        raise ValueError(f"Invalid index value: '{index}'.")
    return str(int(s)).zfill(width)


def copy_dataset(source_dir: str, target_dir: str, samples_to_keep: pd.DataFrame) -> None:
    """Copies the dataset from the source directory to the target directory, keeping only the samples specified in the DataFrame.

    Args:
        source_dir: The directory containing the original dataset.
        target_dir: The directory where the modified dataset will be saved. This directory must not already exist.
        samples_to_keep: A DataFrame containing the samples to keep, with the same format as labels.csv.
    """
    validate_dataset_structure(source_dir)

    # Create target dir structure
    Path(target_dir).mkdir(parents=True, exist_ok=False)
    (Path(target_dir) / "rgb").mkdir(parents=True, exist_ok=False)
    (Path(target_dir) / "depth").mkdir(parents=True, exist_ok=False)

    # Copy over rgb and depth images for the samples to keep
    for _, row in samples_to_keep.iterrows():
        index = normalize_index(row["index"], width=8)
        shutil.copy(f"{source_dir}/rgb/{index}.png", f"{target_dir}/rgb/{index}.png")
        shutil.copy(f"{source_dir}/depth/{index}.png", f"{target_dir}/depth/{index}.png")

    # Copy over intrinsics.json from source to target
    shutil.copy(f"{source_dir}/intrinsics.json", f"{target_dir}/intrinsics.json")

    # Normalize index column of labels.csv
    samples_to_keep["index"] = samples_to_keep["index"].apply(lambda x: normalize_index(x, width=8))

    # Write the modified labels.csv file to the target directory
    labels_path = f"{target_dir}/labels.csv"
    samples_to_keep.to_csv(labels_path, index=False)


def balance_dataset(
    dataset_dir: str,
    target_dir: str,
    steering_angle_ranges: list[tuple[float, float]],
    samples_to_remove: list[int],
    hist_bins: int,
) -> None:
    """Randomly removes the specified number of samples from the dataset that have steering angles within the specified ranges.

    Args:
        dataset_dir: The directory containing the dataset.
        target_dir: The directory where the balanced dataset will be saved. This directory must not already exist.
        steering_angle_ranges: A list of tuples representing the ranges of steering angles to consider for removal (min, max).
        samples_to_remove: A list of integers representing the number of samples to remove from each specified range.
        hist_bins: The number of bins for the histogram.
    """

    # Validate dataset directory structure (must have labels.csv, intrinsics.json, and depth/ and rgb/ subdirectories)
    validate_dataset_structure(dataset_dir)

    # Validate list lengths
    if len(steering_angle_ranges) == 0:
        raise ValueError("steering_angle_ranges must contain at least one range.")
    if len(steering_angle_ranges) != len(samples_to_remove):
        raise ValueError("steering_angle_ranges and samples_to_remove must have the same length.")

    # Ensure min_angle is less than max_angle for each range
    for min_angle, max_angle in steering_angle_ranges:
        if min_angle >= max_angle:
            raise ValueError(f"Invalid steering angle range: ({min_angle}, {max_angle}). min_angle must be less than max_angle.")

    # Warn if the steering angle ranges overlap, as this may lead to unintended consequences when removing samples
    overlapping_ranges = []
    for i in range(len(steering_angle_ranges)):
        for j in range(i + 1, len(steering_angle_ranges)):
            range_i = steering_angle_ranges[i]
            range_j = steering_angle_ranges[j]
            if range_i[0] < range_j[1] and range_i[1] > range_j[0]:
                overlapping_ranges.append((range_i, range_j))

    # prompt user to continue or not
    if overlapping_ranges:
        print(
            "Warning: The following steering angle ranges overlap, which may lead to unintended consequences when removing samples:"
        )
        for range_i, range_j in overlapping_ranges:
            print(f"Range 1: {range_i}, Range 2: {range_j}")
        response = input("Do you wish to continue? (y/n): ")
        if response.lower() != "y":
            print("Aborting.")
            return

    # Ensure that samples_to_remove is a nonnegative integer for each range
    for num_samples in samples_to_remove:
        if not isinstance(num_samples, int):
            raise ValueError(f"Invalid number of samples to remove: {num_samples}. Must be an integer.")
        if num_samples < 0:
            raise ValueError(f"Invalid number of samples to remove: {num_samples}. Must be nonnegative.")

    # Load the labels.csv file into a DataFrame
    df = pd.read_csv(f"{dataset_dir}/labels.csv")

    # Filter the DataFrame to get samples with steering angles within the specified ranges
    for i, (min_angle, max_angle) in enumerate(steering_angle_ranges):
        filtered_df = df[(df["steering_angle_rad"] >= min_angle) & (df["steering_angle_rad"] <= max_angle)]

        # Ensure the number of samples to remove does not exceed the number of available samples in the filtered DataFrame
        if samples_to_remove[i] > len(filtered_df):
            raise ValueError(
                f"Cannot remove {samples_to_remove[i]} samples from range ({min_angle}, {max_angle}) because it only contains {len(filtered_df)} samples."
            )

        # Randomly sample the specified number of samples to remove
        samples_to_remove_df = filtered_df.sample(n=samples_to_remove[i], random_state=42)

        # Remove the sampled rows from the original DataFrame
        df = df.drop(samples_to_remove_df.index)

    # Visualize the new distribution of steering angles after removal
    plot_histogram_of_steering_angles(df, bins=hist_bins, show_plot=True)
    response = input("Do you wish to create a new dataset with the balanced steering angles? (y/n): ")
    if response.lower() == "y":
        # Create target dir if it doesn't exist
        copy_dataset(dataset_dir, target_dir, df)
        print(f"New dataset created at {target_dir}.")
    else:
        print("Aborting. No changes were made to the dataset.")


def main() -> None:
    """Example usage of the balance_dataset function."""

    dataset_dir = "/home/jftaggart02/datasets/trial_03_clean"
    target_dir = "/home/jftaggart02/datasets/trial_03_balanced"
    steering_angle_ranges = [(-0.05, 0.05)]
    samples_to_remove = [772]
    hist_bins = 15

    balance_dataset(dataset_dir, target_dir, steering_angle_ranges, samples_to_remove, hist_bins)


if __name__ == "__main__":
    # plot_histogram_of_steering_angles(
    #     pd.read_csv("/home/jftaggart02/datasets/trial_03_clean/labels.csv"), bins=15, show_plot=True
    # )
    # main()
    plot_histogram_of_steering_angles(
        pd.read_csv("/home/jftaggart02/datasets/trial_03_balanced/labels.csv"), bins=15, show_plot=True
    )
