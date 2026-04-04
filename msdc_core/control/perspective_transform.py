import json
from dataclasses import dataclass
import argparse
import numpy as np
import cv2
import open3d as o3d


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def load_intrinsics_from_json(json_path: str, stream: str = "rgb") -> Intrinsics:
    """
    Load camera intrinsics from your saved ROS2 camera_info JSON.

    Args:
        json_path: path to JSON file
        stream: "rgb" or "depth"

    Returns:
        Intrinsics(fx, fy, cx, cy)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if stream not in data:
        raise ValueError(f"Stream '{stream}' not found in JSON. Options: {list(data.keys())}")

    K = data[stream]["k"]  # 3x3 matrix in row-major

    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]

    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def load_rgb_png(path: str) -> np.ndarray:
    """Load RGB PNG as HxWx3 RGB uint8."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def load_depth_png(path: str) -> np.ndarray:
    """
    Load depth PNG as HxW uint16.
    Assumes pass-through from ROS2 / Realsense depth image.
    """
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    if depth.ndim != 2:
        raise ValueError("Depth image must be single-channel.")
    if depth.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth PNG, got {depth.dtype}.")
    return depth


def shift_view_rgbd(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    intr: Intrinsics,
    depth_scale: float = 1000.0,
    shift_x_m: float = 0.05,
    shift_y_m: float = 0.0,
    shift_z_m: float = 0.0,
    depth_trunc_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift viewpoint using RGB-D reprojection.

    Parameters
    ----------
    rgb : HxWx3 uint8
        RGB image.
    depth_raw : HxW uint16
        Depth image stored in PNG.
    intr : Intrinsics
        Camera intrinsics for the image frame you are rendering in.
        For aligned depth-to-color, this should be COLOR intrinsics.
    depth_scale : float
        Raw depth units per meter. For many RealSense PNG/ROS depth images,
        1000.0 means raw units are millimeters.
    shift_x_m : float
        Positive = simulate camera moving right.
    shift_y_m : float
        Positive = simulate camera moving up.
    shift_z_m : float
        Positive = simulate camera moving forward.
    depth_trunc_m : float
        Ignore depth beyond this value.

    Returns
    -------
    shifted_rgb : HxWx3 uint8
    shifted_depth_m : HxW float32
    """
    h, w = depth_raw.shape
    if rgb.shape[:2] != (h, w):
        raise ValueError("RGB and depth must have the same resolution.")

    # Convert depth to meters
    depth_m = depth_raw.astype(np.float32) / float(depth_scale)

    # Remove invalid depth
    valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m < depth_trunc_m)

    # Create Open3D RGBD image
    rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_raw))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )

    # Use the intrinsics for the image frame you are rendering in.
    # Since depth is aligned to color, use COLOR intrinsics here.
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w,
        height=h,
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.cx,
        cy=intr.cy,
    )

    # Back-project to point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # Simulate camera translation:
    # moving camera right means world points appear shifted left,
    # so we transform the point cloud by the negative translation.
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = -shift_x_m
    T[1, 3] = -shift_y_m
    T[2, 3] = -shift_z_m
    pcd.transform(T)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if points.shape[0] == 0:
        raise RuntimeError("No valid 3D points were generated from the depth image.")

    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    valid2 = np.isfinite(Z) & (Z > 1e-6)
    X = X[valid2]
    Y = Y[valid2]
    Z = Z[valid2]
    colors = colors[valid2]

    # Reproject to new image
    u = intr.fx * (X / Z) + intr.cx
    v = intr.fy * (Y / Z) + intr.cy

    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)

    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[inside]
    v = v[inside]
    Z = Z[inside]
    colors = colors[inside]

    # Z-buffer rendering
    shifted_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    shifted_depth_m = np.full((h, w), np.inf, dtype=np.float32)

    # far -> near so nearer points overwrite farther ones
    order = np.argsort(Z)[::-1]

    for i in order:
        uu = u[i]
        vv = v[i]
        zz = Z[i]
        if zz < shifted_depth_m[vv, uu]:
            shifted_depth_m[vv, uu] = zz
            shifted_rgb[vv, uu] = np.clip(colors[i] * 255.0, 0, 255).astype(np.uint8)

    # Optional hole filling
    hole_mask = np.isinf(shifted_depth_m).astype(np.uint8) * 255
    if np.any(hole_mask):
        shifted_rgb = cv2.inpaint(shifted_rgb, hole_mask, 3, cv2.INPAINT_TELEA)
        shifted_depth_m[hole_mask.astype(bool)] = 0.0

    return shifted_rgb, shifted_depth_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True, help="Path to RGB PNG")
    parser.add_argument("--depth", required=True, help="Path to depth PNG (uint16)")
    parser.add_argument("--out_rgb", default="shifted_rgb.png")
    parser.add_argument("--out_depth", default="shifted_depth.png")

    # Intrinsics for the aligned image frame.
    # If depth is aligned to color, use COLOR intrinsics here.
    parser.add_argument("--intrinsics_json", required=True, help="Path to JSON with camera intrinsics")

    parser.add_argument("--depth_scale", type=float, default=1000.0,
                        help="Raw depth units per meter. Often 1000.0 for mm.")
    parser.add_argument("--shift_x", type=float, default=0.05,
                        help="Camera shift in meters. Positive = move camera right.")
    parser.add_argument("--shift_y", type=float, default=0.0)
    parser.add_argument("--shift_z", type=float, default=0.0)
    parser.add_argument("--depth_trunc", type=float, default=10.0)

    args = parser.parse_args()

    rgb = load_rgb_png(args.rgb)
    depth_raw = load_depth_png(args.depth)

    intr = load_intrinsics_from_json(args.intrinsics_json, stream="rgb")

    shifted_rgb, shifted_depth_m = shift_view_rgbd(
        rgb=rgb,
        depth_raw=depth_raw,
        intr=intr,
        depth_scale=args.depth_scale,
        shift_x_m=args.shift_x,
        shift_y_m=args.shift_y,
        shift_z_m=args.shift_z,
        depth_trunc_m=args.depth_trunc,
    )

    # Save RGB
    shifted_bgr = cv2.cvtColor(shifted_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out_rgb, shifted_bgr)

    # Save depth as 16-bit PNG in millimeters
    shifted_depth_mm = np.clip(shifted_depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(args.out_depth, shifted_depth_mm)

    print(f"Saved shifted RGB to:   {args.out_rgb}")
    print(f"Saved shifted depth to: {args.out_depth}")


if __name__ == "__main__":
    main()
