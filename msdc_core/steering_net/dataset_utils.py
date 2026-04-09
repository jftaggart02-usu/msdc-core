from pathlib import Path
import shutil

import pandas as pd


def validate_dataset_structure(dataset_dir: str, validate_depth: bool = True) -> None:
    """Validates that the dataset directory has the expected structure.

    Args:
        dataset_dir: The directory containing the dataset.
        validate_depth: Whether to check for the presence of depth images as well.

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
    if validate_depth:
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


def normalize_index_column(df: pd.DataFrame, width: int = 8) -> None:
    """Normalizes the index column of the DataFrame to be zero-padded strings of a given width."""
    df["index"] = df["index"].apply(lambda x: normalize_index(x, width=width))


def copy_dataset(source_dir: str, target_dir: str, samples_to_keep: pd.DataFrame | None = None, copy_depth: bool = True) -> None:
    """Copies the dataset from the source directory to the target directory, keeping only the samples specified in the DataFrame.

    Args:
        source_dir: The directory containing the original dataset.
        target_dir: The directory where the modified dataset will be saved. This directory must not already exist.
        samples_to_keep: A DataFrame containing the samples to keep, with the same format as labels.csv.
        copy_depth: Whether to copy the depth images as well.
    """
    validate_dataset_structure(source_dir, validate_depth=copy_depth)

    # Create target dir structure
    Path(target_dir).mkdir(parents=True, exist_ok=False)

    # Copy over rgb and depth images for the samples to keep
    if samples_to_keep is not None:
        (Path(target_dir) / "rgb").mkdir(parents=True, exist_ok=False)
        if copy_depth:
            (Path(target_dir) / "depth").mkdir(parents=True, exist_ok=False)
        normalize_index_column(samples_to_keep)
        for _, row in samples_to_keep.iterrows():
            index = row["index"]
            shutil.copy(f"{source_dir}/rgb/{index}.png", f"{target_dir}/rgb/{index}.png")
            if copy_depth:
                shutil.copy(f"{source_dir}/depth/{index}.png", f"{target_dir}/depth/{index}.png")
    else:
        # If no samples_to_keep is provided, copy all images
        shutil.copytree(f"{source_dir}/rgb", f"{target_dir}/rgb")
        if copy_depth:
            shutil.copytree(f"{source_dir}/depth", f"{target_dir}/depth")

    # Copy over intrinsics.json from source to target
    shutil.copy(f"{source_dir}/intrinsics.json", f"{target_dir}/intrinsics.json")

    # Write the modified labels.csv file to the target directory
    labels_path = f"{target_dir}/labels.csv"
    if samples_to_keep is not None:
        samples_to_keep.to_csv(labels_path, index=False)
    else:
        shutil.copy(f"{source_dir}/labels.csv", labels_path)

    # Sanity check the target directory structure after copying
    validate_dataset_structure(target_dir, validate_depth=copy_depth)
