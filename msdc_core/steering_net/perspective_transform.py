"""This module provides functionality to perform a perspective transform on an RGB image using depth data.
It includes functions to create a transformation matrix, load camera intrinsics from a JSON file, and reproject a point cloud using a z-buffer.
The transformed image can optionally be inpainted to fill in gaps caused by the perspective change.

Note on coordinate frames:
- The input RGB and depth images are assumed to be in the same camera frame, with the camera looking along the positive z-axis, x-axis to the right, and y-axis down.
- The transformation matrix is applied to the point cloud in this camera frame, and the output image is generated from the transformed point cloud using the same camera intrinsics.
- A positive translation along the x-axis will shift the camera left.
- A positive translation along the y-axis will shift the camera up.
- A positive translation along the z-axis will shift the camera backward.
- A positive rotation around the x-axis will pitch the camera down.
- A positive rotation around the y-axis will yaw the camera left.
- A positive rotation around the z-axis will roll the camera counter-clockwise (left).
"""

import json
import os

import open3d as o3d
import numpy as np
import cv2


def create_transform_matrix(rotation_angles: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    """Create a homogeneous transformation matrix from rotation angles and translation vector.

    Args:
        rotation_angles: A 3-element array containing rotation angles (in degrees) around the x, y, and z axes.
        translation_vector: A 3-element array containing translation along the x, y, and z axes.
    """
    # Create the rotation matrix from the rotation angles using Open3D's built-in function
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(rotation_angles))

    # Create the homogeneous transformation matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = translation_vector
    return T


def load_intrinsic_json(filepath: str) -> o3d.camera.PinholeCameraIntrinsic:
    """Load intrinsic data from a JSON file and store in an Open3D class for easy use.

    Args:
        filepath: The path to the JSON file containing intrinsics data.

    Returns:
        A class containing camera intrinsics data.
    """
    with open(filepath, "r") as f:
        camera_info = json.load(f)
        rgb_info = camera_info["rgb"]
        depth_info = camera_info["depth"]

        # Extract intrinsics from RGB camera info
        width = rgb_info["width"]
        height = rgb_info["height"]
        intrinsic_matrix = np.array(rgb_info["k"])

        # Sanity check: make sure the depth and RGB intrinsics line up
        if width != depth_info["width"]:
            raise ValueError("Image widths for RGB and Depth are not equivalent!")
        if height != depth_info["height"]:
            raise ValueError("Image heights for RGB and Depth are not equivalent!")
        if not np.isclose(intrinsic_matrix, np.array(depth_info["k"])).all():
            raise ValueError("Intrinsic matrices for RGB and Depth are not equivalent!")
        if rgb_info["frame_id"] != depth_info["frame_id"]:
            raise ValueError("Frame IDs for RGB and Depth are not equivalent!")

        # Reshape the intrinsic matrix to be 3x3 and create an Open3D PinholeCameraIntrinsic object
        intrinsic_matrix = intrinsic_matrix.reshape(3, 3)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, intrinsic_matrix=intrinsic_matrix)
        return intrinsic


def zbuffer_reproject(
    pcd: o3d.geometry.PointCloud, intrinsic: o3d.camera.PinholeCameraIntrinsic
) -> tuple[np.ndarray, np.ndarray]:

    height, width = intrinsic.height, intrinsic.width
    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()

    # points: (N,3) in target camera frame, meters
    # colors: (N,3) float in [0,1]
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    out_rgb = np.zeros((height, width, 3), dtype=np.uint8)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Only points in front of camera and finite
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 1e-6)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    c = colors[valid]

    u = np.round((fx * (x / z)) + cx).astype(np.int32)
    v = np.round((fy * (y / z)) + cy).astype(np.int32)

    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    c = c[in_bounds]

    # Classic z-buffer update
    for i in range(z.shape[0]):
        ui = u[i]
        vi = v[i]
        zi = z[i]
        if zi < z_buffer[vi, ui]:
            z_buffer[vi, ui] = zi
            out_rgb[vi, ui] = np.clip(c[i] * 255.0, 0, 255).astype(np.uint8)

    return out_rgb, z_buffer


def perspective_transform(
    rgb_path: str,
    depth_path: str,
    dest_path: str,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    transformation_matrix: np.ndarray,
    inpaint: bool = True,
) -> None:
    """Perform a perspective transform on an RGB image using the depth data."""

    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"RGB path does not exist: {rgb_path}")
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"Depth path does not exist: {depth_path}")

    # Step 1: Create RGBD image from color and depth images
    rgb_image = o3d.io.read_image(rgb_path)
    depth_image = o3d.io.read_image(depth_path)
    if np.asarray(depth_image).dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth image, got {np.asarray(depth_image).dtype}")

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, convert_rgb_to_intensity=False, depth_scale=1000.0, depth_trunc=10.0
    )

    # Step 2: Create point cloud using RGBD image and camera intrinsics
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Step 3: Perform transform on the point cloud using transformation matrix
    pcd.transform(transformation_matrix)

    # Step 4: Convert point cloud back to RGB image using z-buffer
    transformed_rgb, zbuffer = zbuffer_reproject(pcd, intrinsic)

    # Step 5: Optionally fill in gaps
    if inpaint:
        mask = (zbuffer == np.inf).astype(np.uint8) * 255  # Create mask where missing pixels are 255 and valid pixels are 0
        inpainted_rgb = cv2.inpaint(transformed_rgb, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        transformed_image = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
    else:
        transformed_image = cv2.cvtColor(transformed_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving

    # Step 6: Save the transformed image
    ok = cv2.imwrite(dest_path, transformed_image)
    if not ok:
        raise IOError(f"Failed to write transformed image to: {dest_path}")


if __name__ == "__main__":
    intrinsic = load_intrinsic_json("/home/jftaggart02/datasets/trial_03_clean/intrinsics.json")
    transformation_matrix = create_transform_matrix(
        rotation_angles=np.array([0.0, 0.0, 0.0]), translation_vector=np.array([0.0, 0.0, 0.0])
    )
    perspective_transform(
        rgb_path="/home/jftaggart02/datasets/trial_03_clean/rgb/00000080.png",
        depth_path="/home/jftaggart02/datasets/trial_03_clean/depth/00000080.png",
        dest_path="/home/jftaggart02/datasets/transformed_3.png",
        intrinsic=intrinsic,
        transformation_matrix=transformation_matrix,
        inpaint=True,
    )
