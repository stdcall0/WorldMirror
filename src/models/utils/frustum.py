import einops
import torch


def _is_spherical_intrinsics(intrinsics: torch.Tensor, height: int, width: int, rel_tol: float = 0.05) -> torch.Tensor:
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    expected_fx = float(width) / (2.0 * torch.pi)
    expected_fy = float(height) / torch.pi
    tol_fx = max(expected_fx * rel_tol, 1e-6)
    tol_fy = max(expected_fy * rel_tol, 1e-6)

    full_erp = ((fx - expected_fx).abs() <= tol_fx) & ((fy - expected_fy).abs() <= tol_fy)
    # Allow horizontal FoV-cropped ERP intrinsics: fy follows ERP rule while fx can be larger.
    partial_erp = ((fy - expected_fy).abs() <= tol_fy) & (fx >= (expected_fx - tol_fx))
    valid = torch.isfinite(fx) & torch.isfinite(fy) & (fx > 1e-6) & (fy > 1e-6)
    return valid & (full_erp | partial_erp)


# Calculate the loss mask for the target views in the batch
@torch.no_grad()
def calculate_unprojected_mask(views, context_nums):
    '''Calcuate the loss mask for the target views in the batch'''
    target_depth = views["depthmap"][:, context_nums:]
    target_intrinsics = views["camera_intrs"][:, context_nums:]
    target_c2w = views["camera_poses"][:, context_nums:]
    context_depth = views["depthmap"][:, :context_nums]
    context_intrinsics = views["camera_intrs"][:, :context_nums]
    context_c2w = views["camera_poses"][:, :context_nums]

    target_intrinsics = target_intrinsics[..., :3, :3]
    context_intrinsics = context_intrinsics[..., :3, :3]

    mask = calculate_in_frustum_mask(
        target_depth, target_intrinsics, target_c2w,
        context_depth, context_intrinsics, context_c2w
    )
    return mask

@torch.no_grad()
def calculate_in_frustum_mask(depth_1, intrinsics_1, c2w_1, depth_2, intrinsics_2, c2w_2):
    """
    A function that takes in the depth, intrinsics and c2w matrices of two sets
    of views, and then works out which of the pixels in the first set of views
    has a direct corresponding pixel in any of views in the second set

    Args:
        depth_1: (b, v1, h, w)
        intrinsics_1: (b, v1, 3, 3)
        c2w_1: (b, v1, 4, 4)
        depth_2: (b, v2, h, w)
        intrinsics_2: (b, v2, 3, 3)
        c2w_2: (b, v2, 4, 4)

    Returns:
        torch.Tensor: valid mask with shape (b, v1, v2, h, w).
    """

    _, v1, h, w = depth_1.shape
    _, v2, _, _ = depth_2.shape

    # Unproject the depth to get the 3D points in world space
    points_3d = unproject_depth(depth_1[..., None], intrinsics_1, c2w_1)  # (b, v1, h, w, 3)

    # Project the 3D points into the pixel space of all the second views simultaneously
    camera_points = world_space_to_camera_space(points_3d, c2w_2)  # (b, v1, v2, h, w, 3)
    points_2d = camera_space_to_pixel_space(camera_points, intrinsics_2)  # (b, v1, v2, h, w, 2)

    # Calculate depth according to the projection model of target views.
    rendered_depth = camera_space_to_depth(camera_points, intrinsics_2, h, w)  # (b, v1, v2, h, w)

    # We use three conditions to determine if a point should be masked

    # Condition 1: Check if the points are in the frustum of any of the v2 views
    in_frustum_mask = (
        (points_2d[..., 0] > 0) &
        (points_2d[..., 0] < w) &
        (points_2d[..., 1] > 0) &
        (points_2d[..., 1] < h)
    )  # (b, v1, v2, h, w)
    in_frustum_mask = in_frustum_mask.any(dim=-3)  # (b, v1, h, w)

    # Condition 2: Check if the points have non-zero (i.e. valid) depth in the input view
    non_zero_depth = depth_1 > 1e-6

    # Condition 3: Check if the points have matching depth to any of the v2
    # views torch.nn.functional.grid_sample expects the input coordinates to
    # be normalized to the range [-1, 1], so we normalize first
    points_2d[..., 0] /= w
    points_2d[..., 1] /= h
    points_2d = points_2d * 2 - 1
    matching_depth = torch.ones_like(rendered_depth, dtype=torch.bool)
    for b in range(depth_1.shape[0]):
        for i in range(v1):
            for j in range(v2):
                depth = einops.rearrange(depth_2[b, j], 'h w -> 1 1 h w')
                coords = einops.rearrange(points_2d[b, i, j], 'h w c -> 1 h w c')
                sampled_depths = torch.nn.functional.grid_sample(depth, coords, align_corners=False)[0, 0]
                matching_depth[b, i, j] = torch.isclose(rendered_depth[b, i, j], sampled_depths, atol=1e-1)

    matching_depth = matching_depth.any(dim=-3)  # (..., v1, h, w)

    mask = in_frustum_mask & non_zero_depth & matching_depth
    return mask

# --- Projections ---
def homogenize_points(points):
    """Append a '1' along the final dimension of the tensor (i.e. convert xyz->xyz1)"""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def normalize_homogenous_points(points):
    """Normalize the point vectors"""
    return points / points[..., -1:]


def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    """
    Convert pixel space points to camera space points.

    Args:
        pixel_space_points (torch.Tensor): Pixel space points with shape (h, w, 2)
        depth (torch.Tensor): Depth map with shape (b, v, h, w, 1)
        intrinsics (torch.Tensor): Camera intrinsics with shape (b, v, 3, 3)

    Returns:
        torch.Tensor: Camera space points with shape (b, v, h, w, 3).
    """
    h, w = depth.shape[-3], depth.shape[-2]
    is_spherical = _is_spherical_intrinsics(intrinsics, h, w)

    # Pinhole path.
    pixel_space_points_h = homogenize_points(pixel_space_points)
    pinhole_points = torch.einsum('b v i j , h w j -> b v h w i', intrinsics.inverse(), pixel_space_points_h)
    pinhole_points = pinhole_points * depth

    if bool((~is_spherical).all()):
        return pinhole_points

    # ERP spherical path (depth is interpreted as ray distance).
    u = pixel_space_points[..., 0]
    v = pixel_space_points[..., 1]
    fx = intrinsics[..., 0, 0][..., None, None]
    fy = intrinsics[..., 1, 1][..., None, None]
    cx = intrinsics[..., 0, 2][..., None, None]
    cy = intrinsics[..., 1, 2][..., None, None]

    lon = (u + 0.5 - cx) / fx
    lat = 0.5 * torch.pi - (v + 0.5 - cy) / fy
    cos_lat = torch.cos(lat)
    dir_x = cos_lat * torch.sin(lon)
    dir_y = -torch.sin(lat)
    dir_z = cos_lat * torch.cos(lon)
    spherical_dirs = torch.stack([dir_x, dir_y, dir_z], dim=-1)
    spherical_points = spherical_dirs * depth

    if bool(is_spherical.all()):
        return spherical_points

    spherical_mask = is_spherical[..., None, None, None]
    return torch.where(spherical_mask, spherical_points, pinhole_points)


def camera_space_to_world_space(camera_space_points, c2w):
    """
    Convert camera space points to world space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum('b v i j , b v h w j -> b v h w i', c2w, camera_space_points)
    return world_space_points[..., :3]


def camera_space_to_pixel_space(camera_space_points, intrinsics):
    """
    Convert camera space points to pixel space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v1, v2, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 3, 3)

    Returns:
        torch.Tensor: World space points with shape (b, v1, v2, h, w, 2).
    """
    h, w = camera_space_points.shape[-3], camera_space_points.shape[-2]
    is_spherical = _is_spherical_intrinsics(intrinsics, h, w)

    # Pinhole path.
    camera_points_norm = normalize_homogenous_points(camera_space_points)
    pinhole_pixels = torch.einsum('b u i j , b v u h w j -> b v u h w i', intrinsics, camera_points_norm)[..., :2]

    if bool((~is_spherical).all()):
        return pinhole_pixels

    # ERP spherical path.
    x = camera_space_points[..., 0]
    y = camera_space_points[..., 1]
    z = camera_space_points[..., 2]
    radius = torch.linalg.vector_norm(camera_space_points, dim=-1).clamp_min(1e-6)

    lon = torch.atan2(x, z)
    lat = torch.asin((-y / radius).clamp(-1.0 + 1e-6, 1.0 - 1e-6))

    fx = intrinsics[:, None, :, 0, 0][:, :, :, None, None]
    fy = intrinsics[:, None, :, 1, 1][:, :, :, None, None]
    cx = intrinsics[:, None, :, 0, 2][:, :, :, None, None]
    cy = intrinsics[:, None, :, 1, 2][:, :, :, None, None]

    u = lon * fx + cx - 0.5
    v = (0.5 * torch.pi - lat) * fy + cy - 0.5
    spherical_pixels = torch.stack([u, v], dim=-1)

    if bool(is_spherical.all()):
        return spherical_pixels

    spherical_mask = is_spherical[:, None, :, None, None, None]
    return torch.where(spherical_mask, spherical_pixels, pinhole_pixels)


def camera_space_to_depth(camera_space_points, intrinsics, height: int, width: int):
    """Return depth value in the same convention as the camera projection model."""
    is_spherical = _is_spherical_intrinsics(intrinsics, height, width)
    z_depth = camera_space_points[..., 2]
    if bool((~is_spherical).all()):
        return z_depth

    ray_depth = torch.linalg.vector_norm(camera_space_points, dim=-1)
    if bool(is_spherical.all()):
        return ray_depth

    spherical_mask = is_spherical[:, None, :, None, None]
    return torch.where(spherical_mask, ray_depth, z_depth)


def world_space_to_camera_space(world_space_points, c2w):
    """
    Convert world space points to pixel space points.

    Args:
        world_space_points (torch.Tensor): World space points with shape (b, v1, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, v2, h, w, 3).
    """
    world_space_points = homogenize_points(world_space_points)
    camera_space_points = torch.einsum('b u i j , b v h w j -> b v u h w i', c2w.inverse(), world_space_points)
    return camera_space_points[..., :3]


def unproject_depth(depth, intrinsics, c2w):
    """
    Turn the depth map into a 3D point cloud in world space

    Args:
        depth: (b, v, h, w, 1)
        intrinsics: (b, v, 3, 3)
        c2w: (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """

    # Compute indices of pixels
    h, w = depth.shape[-3], depth.shape[-2]
    x_grid, y_grid = torch.meshgrid(
        torch.arange(w, device=depth.device, dtype=torch.float32),
        torch.arange(h, device=depth.device, dtype=torch.float32),
        indexing='xy'
    )  # (h, w), (h, w)

    # Compute coordinates of pixels in camera space
    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)  # (..., h, w, 2)
    camera_points = pixel_space_to_camera_space(pixel_space_points, depth, intrinsics)  # (..., h, w, 3)

    # Convert points to world space
    world_points = camera_space_to_world_space(camera_points, c2w)  # (..., h, w, 3)

    return world_points