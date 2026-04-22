import math

import torch
from .rotation import quat_to_rotmat, rotmat_to_quat


def build_erp_intrinsics(image_hw, cam_like, center_offsets=None):
    """Build pseudo intrinsics for equirectangular projection geometry."""
    h, w = image_hw
    shape = cam_like.shape[:-1] + (3, 3)
    intr = torch.zeros(shape, device=cam_like.device, dtype=cam_like.dtype)

    fx = float(w) / (2.0 * math.pi)
    fy = float(h) / math.pi
    cx = (float(w) - 1.0) * 0.5
    cy = (float(h) - 1.0) * 0.5

    if center_offsets is not None:
        offset_x = center_offsets[..., 0].clamp(-1.0, 1.0)
        offset_y = center_offsets[..., 1].clamp(-1.0, 1.0)
        cx = cx + offset_x * (float(w) * 0.5)
        cy = cy + offset_y * (float(h) * 0.5)

    intr[..., 0, 0] = fx
    intr[..., 1, 1] = fy
    intr[..., 0, 2] = cx
    intr[..., 1, 2] = cy
    intr[..., 2, 2] = 1.0
    return intr


def camera_params_to_vector(
    ext, intr, image_hw=None, camera_model="perspective"
):
    """Convert camera matrices to a compact vector."""
    # ext: (..., 3, 4) or (..., 4, 4)
    # intr: (..., 3, 3): Intrinsics
    # image_hw: (h, w)
    R = ext[..., :3, :3]           # Rotation part
    t = ext[..., :3, 3]            # Translation part
    q = rotmat_to_quat(R)  # Quaternion (wxyz)

    if camera_model == "spherical":
        if image_hw is None:
            raise ValueError("image_hw is required for spherical camera vector conversion")
        h, w = image_hw
        cx_ref = (float(w) - 1.0) * 0.5
        cy_ref = (float(h) - 1.0) * 0.5
        if intr is None:
            offset_x = torch.zeros_like(t[..., 0])
            offset_y = torch.zeros_like(t[..., 1])
        else:
            denom_x = max(float(w) * 0.5, 1.0)
            denom_y = max(float(h) * 0.5, 1.0)
            offset_x = ((intr[..., 0, 2] - cx_ref) / denom_x).clamp(-1.0, 1.0)
            offset_y = ((intr[..., 1, 2] - cy_ref) / denom_y).clamp(-1.0, 1.0)
        vec = torch.stack([
            t[..., 0], t[..., 1], t[..., 2],
            q[..., 0], q[..., 1], q[..., 2], q[..., 3],
            offset_x, offset_y,
        ], dim=-1).float()
        return vec

    if intr is None or image_hw is None:
        raise ValueError("intr and image_hw are required for perspective camera vector conversion")

    h, w = image_hw
    fov_v = 2.0 * torch.atan(h * 0.5 / intr[..., 1, 1])  # Vertical FOV
    fov_u = 2.0 * torch.atan(w * 0.5 / intr[..., 0, 0])  # Horizontal FOV
    vec = torch.stack([
        t[..., 0], t[..., 1], t[..., 2],
        q[..., 0], q[..., 1], q[..., 2], q[..., 3],
        fov_v, fov_u
    ], dim=-1).float()
    return vec

def extrinsics_to_vector(ext):
    """Convert extrinsics to [t, q] vector."""
    # ext: (..., 3, 4)
    R = ext[..., :3, :3]
    t = ext[..., :3, 3]
    q = rotmat_to_quat(R)
    vec = torch.stack([
        t[..., 0], t[..., 1], t[..., 2],
        q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    ], dim=-1).float()
    return vec

def vector_to_extrinsics(cam_vec):
    """Convert [t, q] vector to extrinsic matrix."""
    # cam_vec: (..., 7)
    q = cam_vec[..., 3:7]
    t = cam_vec[..., :3]
    R = quat_to_rotmat(q)
    ext = torch.cat([R, t.unsqueeze(-1)], dim=-1)
    return ext

def vector_to_camera_matrices(
    cam_vec, image_hw=None, build_intr=True, camera_model="perspective"
):
    """Reconstruct extrinsic and intrinsic matrix from vector."""
    # cam_vec: (..., 9) or (..., 7)
    intr = None
    # Decompose vector
    t = cam_vec[..., 0:3]
    q = cam_vec[..., 3:7]

    # Build extrinsic: [R|t]
    R = quat_to_rotmat(q)
    ext = torch.cat([R, t.unsqueeze(-1)], dim=-1)

    # Build intrinsic if needed
    if build_intr:
        if image_hw is None:
            raise ValueError("image_hw is required when build_intr=True")
        h, w = image_hw
        use_spherical = (camera_model == "spherical") or (cam_vec.shape[-1] < 9)
        if use_spherical:
            offsets = cam_vec[..., 7:9] if cam_vec.shape[-1] >= 9 else None
            intr = build_erp_intrinsics((h, w), cam_vec, offsets)
        else:
            fov_v = cam_vec[..., 7].clamp(min=1e-4, max=math.pi - 1e-4)
            fov_u = cam_vec[..., 8].clamp(min=1e-4, max=math.pi - 1e-4)
            fy = h * 0.5 / torch.tan(fov_v * 0.5)
            fx = w * 0.5 / torch.tan(fov_u * 0.5)
            shape = cam_vec.shape[:-1] + (3, 3)
            intr = torch.zeros(shape, device=cam_vec.device, dtype=cam_vec.dtype)
            intr[..., 0, 0] = fx
            intr[..., 1, 1] = fy
            intr[..., 0, 2] = (float(w) - 1.0) * 0.5
            intr[..., 1, 2] = (float(h) - 1.0) * 0.5
            intr[..., 2, 2] = 1.0

    return ext, intr
