"""This module contains code that performs preprocessing on the dataset, such as normalization and augmentation.

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

import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram_of_steering_angles(labels_csv_file: str, bins: int) -> None:
    """Plots a histogram of the steering angles in the dataset.

    Args:
        labels_csv_file (str): The path to the labels CSV file.
        bins (int): The number of bins for the histogram.
    """
    # Load the labels.csv file
    df = pd.read_csv(labels_csv_file)

    # Plot the histogram of steering angles
    df['steering_angle_rad'].hist(bins=bins)
    plt.xlabel('Steering Angle (radians)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Steering Angles')
    plt.show()


def remove_samples_with_small_steering_angles(labels_csv_file: str, steering_angle_range: tuple[float, float], samples_to_remove: int) -> None:
    """Randomly removes the specified number of samples from the CSV file that have steering angles within the specified range.
    
    Args:
        labels_csv_file (str): The path to the labels CSV file.
        steering_angle_range (tuple[float, float]): The range of steering angles to consider for removal (min, max).
        samples_to_remove (int): The number of samples to remove from the specified range.
    """

    # Load the labels.csv file
    df = pd.read_csv(labels_csv_file)

    # Filter the DataFrame to get samples with steering angles within the specified range
    min_angle, max_angle = steering_angle_range
    filtered_df = df[(df['steering_angle_rad'] >= min_angle) & (df['steering_angle_rad'] <= max_angle)]

    # Randomly sample the specified number of samples to remove
    samples_to_remove_df = filtered_df.sample(n=samples_to_remove, random_state=42)

    # Remove the sampled rows from the original DataFrame
    df = df.drop(samples_to_remove_df.index)

    # Save the modified DataFrame back to a new CSV file
    df.to_csv(labels_csv_file.replace('.csv', '_balanced.csv'), index=False)


if __name__ == "__main__":
    # plot_histogram_of_steering_angles('/home/jftaggart02/datasets/trial_03_clean/labels.csv', bins=15)
    plot_histogram_of_steering_angles('/home/jftaggart02/datasets/trial_03_clean/labels_balanced.csv', bins=15)
    # remove_samples_with_small_steering_angles('/home/jftaggart02/datasets/trial_03_clean/labels.csv', (-0.05, 0.05), 772)

