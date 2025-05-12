import numpy as np
import cv2
import argparse


def quaternion_to_matrix(q):
    """Convert a (w, x, y, z) quaternion to a 33603 rotation matrix."""
    w, x, y, z = q
    # Normalize quaternion
    n = np.linalg.norm([w, x, y, z])
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    # Rotation matrix components (from quaternion algebra)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def stabilize_360_equirect(input_path, rotations, output_path="stabilized_output.mp4"):
    """
    Stabilize a 360�� equirectangular video using per-frame camera rotations.

    Args:
        input_path (str): Path to input equirectangular video.
        rotations (list or ndarray): List of camera rotations, one per frame.
            Each rotation can be a 3��3 matrix or a quaternion (w, x, y, z).
        output_path (str): Path for output stabilized video.

    The function reads each frame, applies the inverse camera rotation to
    its spherical coordinates, and warps the frame with cv2.remap().
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")
    # Video parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Prepare output video writer (MP4 with same size and fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Precompute spherical direction vectors for each pixel (constant)
    xs = np.linspace(0, width - 1, width, dtype=np.float32)
    ys = np.linspace(0, height - 1, height, dtype=np.float32)
    lon = (xs / width) * 2 * np.pi - np.pi  # shape (width,)
    lat = np.pi / 2 - (ys / height) * np.pi  # shape (height,)
    lon_map, lat_map = np.meshgrid(lon, lat)  # shape (height, width)
    # Convert (lat, lon) to 3D unit sphere (x,y,z)
    X = np.cos(lat_map) * np.sin(lon_map)
    Y = np.sin(lat_map)
    Z = np.cos(lat_map) * np.cos(lon_map)
    dirs = np.stack((X, Y, Z), axis=2)  # shape (height, width, 3)

    frame_idx = 0
    for R in rotations:
        ret, frame = cap.read()
        if not ret:
            break  # end of video or error
        # Convert rotation to 3x3 matrix if needed
        if isinstance(R, np.ndarray) and R.shape == (3, 3):
            Rmat = R.astype(np.float64)
        else:
            # Assume quaternion (w, x, y, z)
            q = np.array(R, dtype=np.float64)
            Rmat = quaternion_to_matrix(q)
        # Inverse rotation (world->camera)
        R_inv = Rmat.T  # transpose of orthonormal rotation matrix

        # Apply R_inv to all direction vectors (vectorized)
        # Flatten directions to (N,3), apply R_inv, then reshape
        H, W = height, width
        dirs_flat = dirs.reshape(-1, 3)  # (H*W, 3)
        dirs_rot = dirs_flat.dot(R_inv)  # (H*W, 3) rotated directions
        Xr = dirs_rot[:, 0].reshape(H, W)
        Yr = dirs_rot[:, 1].reshape(H, W)
        Zr = dirs_rot[:, 2].reshape(H, W)

        # Convert rotated vectors back to spherical angles
        # ��' = asin(y), ��' = atan2(x, z)
        # Clamp y for asin to [-1,1] to avoid numerical issues
        Yr_clamped = np.clip(Yr, -1.0, 1.0)
        lat_p = np.arcsin(Yr_clamped)  # ��' in [-��/2, ��/2]
        lon_p = np.arctan2(Xr, Zr)  # ��' in [-��, ��]

        # Map spherical angles to equirectangular pixel coordinates
        map_x = (lon_p / (2 * np.pi) + 0.5) * W  # u = (��'/2�� + 0.5)*width
        map_y = (0.5 - lat_p / np.pi) * H  # v = (0.5 - ��'/��)*height

        # Ensure type float32 for cv2.remap
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # Warp the input frame using remap
        # BORDER_WRAP ensures horizontal wrap-around for 360�� images.
        stabilized = cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )

        out.write(stabilized)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Stabilized video saved to {output_path}")

    if __name__ == "__main__":

        parser = argparse.ArgumentParser(
            description="Stabilize a 360° equirectangular video."
        )
        parser.add_argument("input_path", type=str, help="Path to the input video.")
        parser.add_argument(
            "rotations_path",
            type=str,
            help="Path to the file containing rotations (as a numpy array).",
        )
        parser.add_argument(
            "--output_path",
            type=str,
            default="stabilized_output.mp4",
            help="Path to save the stabilized video.",
        )
        args = parser.parse_args()

        # Load rotations from file
        rotations = np.load(args.rotations_path)

        # Call the stabilization function
        stabilize_360_equirect(args.input_path, rotations, args.output_path)
