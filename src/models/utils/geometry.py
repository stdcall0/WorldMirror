import torch
import numpy as np


def _is_spherical_intrinsics(camera_intrinsics: torch.Tensor, height: int, width: int, rel_tol: float = 0.05) -> torch.Tensor:
    """Detect ERP pseudo intrinsics from focal lengths."""
    fx = camera_intrinsics[:, 0, 0]
    fy = camera_intrinsics[:, 1, 1]

    expected_fx = float(width) / (2.0 * np.pi)
    expected_fy = float(height) / np.pi
    tol_fx = max(expected_fx * rel_tol, 1e-6)
    tol_fy = max(expected_fy * rel_tol, 1e-6)

    full_erp = ((fx - expected_fx).abs() <= tol_fx) & ((fy - expected_fy).abs() <= tol_fy)
    # Allow horizontal FoV-cropped ERP intrinsics: fy follows ERP rule while fx can be larger.
    partial_erp = ((fy - expected_fy).abs() <= tol_fy) & (fx >= (expected_fx - tol_fx))
    valid = torch.isfinite(fx) & torch.isfinite(fy) & (fx > 1e-6) & (fy > 1e-6)

    return valid & (full_erp | partial_erp)


def _depth_to_camera_coords_pinhole(
    depthmap: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    u_grid: torch.Tensor,
    v_grid: torch.Tensor,
) -> torch.Tensor:
    bsz = depthmap.shape[0]
    fx = camera_intrinsics[:, 0, 0]
    fy = camera_intrinsics[:, 1, 1]
    cx = camera_intrinsics[:, 0, 2]
    cy = camera_intrinsics[:, 1, 2]

    z_cam = depthmap
    x_cam = (u_grid - cx.view(bsz, 1, 1)) * z_cam / fx.view(bsz, 1, 1)
    y_cam = (v_grid - cy.view(bsz, 1, 1)) * z_cam / fy.view(bsz, 1, 1)
    return torch.stack([x_cam, y_cam, z_cam], dim=-1)


def _depth_to_camera_coords_spherical(
    depthmap: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    u_grid: torch.Tensor,
    v_grid: torch.Tensor,
) -> torch.Tensor:
    bsz = depthmap.shape[0]
    fx = camera_intrinsics[:, 0, 0]
    fy = camera_intrinsics[:, 1, 1]
    cx = camera_intrinsics[:, 0, 2]
    cy = camera_intrinsics[:, 1, 2]

    u_center = u_grid + 0.5
    v_center = v_grid + 0.5
    lon = (u_center - cx.view(bsz, 1, 1)) / fx.view(bsz, 1, 1)
    lat = 0.5 * np.pi - (v_center - cy.view(bsz, 1, 1)) / fy.view(bsz, 1, 1)

    cos_lat = torch.cos(lat)
    dir_x = cos_lat * torch.sin(lon)
    dir_y = -torch.sin(lat)
    dir_z = cos_lat * torch.cos(lon)
    dirs = torch.stack([dir_x, dir_y, dir_z], dim=-1)

    return dirs * depthmap[..., None]


def depth_to_camera_coords(depthmap, camera_intrinsics):
    """
    Convert depth map to 3D camera coordinates.
    
    Args:
        depthmap (BxHxW tensor): Batch of depth maps
        camera_intrinsics (Bx3x3 tensor): Camera intrinsics matrix for each camera
        
    Returns:
        X_cam (BxHxWx3 tensor): 3D points in camera coordinates
        valid_mask (BxHxW tensor): Mask indicating valid depth pixels
    """
    B, H, W = depthmap.shape
    device = depthmap.device
    dtype = depthmap.dtype
    
    # Ensure intrinsics are float
    camera_intrinsics = camera_intrinsics.float()
    
    # Generate pixel grid
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing='ij'
    )
    
    # Reshape for broadcasting: (1, H, W)
    u_grid = u_grid.unsqueeze(0)
    v_grid = v_grid.unsqueeze(0)

    is_spherical = _is_spherical_intrinsics(camera_intrinsics, H, W)
    if bool(is_spherical.all()):
        X_cam = _depth_to_camera_coords_spherical(depthmap, camera_intrinsics, u_grid, v_grid)
    elif bool((~is_spherical).all()):
        X_cam = _depth_to_camera_coords_pinhole(depthmap, camera_intrinsics, u_grid, v_grid)
    else:
        cam_pinhole = _depth_to_camera_coords_pinhole(depthmap, camera_intrinsics, u_grid, v_grid)
        cam_spherical = _depth_to_camera_coords_spherical(depthmap, camera_intrinsics, u_grid, v_grid)
        spherical_mask = is_spherical.view(B, 1, 1, 1)
        X_cam = torch.where(spherical_mask, cam_spherical, cam_pinhole)
    
    # Valid depth mask
    valid_mask = depthmap > 0.0
    
    return X_cam, valid_mask

def depth_to_world_coords_points(
    depth_map: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor, eps=1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batch of depth maps to world coordinates.

    Args:
        depth_map (torch.Tensor): (B, H, W) Depth map
        extrinsic (torch.Tensor): (B, 4, 4) Camera extrinsic matrix (camera-to-world transformation)
        intrinsic (torch.Tensor): (B, 3, 3) Camera intrinsic matrix

    Returns:
        world_coords_points (torch.Tensor): (B, H, W, 3) World coordinates
        camera_points (torch.Tensor): (B, H, W, 3) Camera coordinates
        point_mask (torch.Tensor): (B, H, W) Valid depth mask
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask (B, H, W)
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates (B, H, W, 3)
    camera_points, _ = depth_to_camera_coords(depth_map, intrinsic)

    # Apply extrinsic matrix (camera -> world)
    R_cam_to_world = extrinsic[:, :3, :3]   # (B, 3, 3)
    t_cam_to_world = extrinsic[:, :3, 3]    # (B, 3)

    # Transform (B, H, W, 3) x (B, 3, 3)^T + (B, 3) -> (B, H, W, 3)
    world_coords_points = torch.einsum('bhwi,bji->bhwj', camera_points, R_cam_to_world) + t_cam_to_world[:, None, None, :]

    return world_coords_points, camera_points, point_mask


def closed_form_inverse_se3(se3: torch.Tensor) -> torch.Tensor:
    """
    Efficiently invert batched SE(3) matrices of shape (B, 4, 4).

    Args:
        se3 (torch.Tensor): (B, 4, 4) Transformation matrices

    Returns:
        out (torch.Tensor): (B, 4, 4) Inverse transformation matrices
    """
    assert se3.ndim == 3 and se3.shape[1:] == (4, 4), f"se3 must be (B, 4, 4), got {se3.shape}"
    R = se3[:, :3, :3]        # (B, 3, 3)
    t = se3[:, :3, 3]         # (B, 3)
    Rt = R.transpose(1, 2)    # (B, 3, 3)
    t_inv = -torch.bmm(Rt, t.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    out = se3.new_zeros(se3.shape)
    out[:, :3, :3] = Rt
    out[:, :3, 3] = t_inv
    out[:, 3, 3] = 1.0
    return out


def create_pixel_coordinate_grid(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices for all frames.
    Returns:
        tuple: A tuple containing:
            - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                            with x, y coordinates and frame indices
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf