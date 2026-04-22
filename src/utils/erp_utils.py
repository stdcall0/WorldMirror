import math
from typing import Tuple

import torch


def _round_to_patch_multiple(value: int, patch_size: int = 14) -> int:
    """Round integer size to nearest positive patch-size multiple."""
    value = max(int(value), patch_size)
    return max(patch_size, int(round(value / patch_size)) * patch_size)


def get_erp_resize_hw(target_height: int, patch_size: int = 14, aspect_ratio: float = 2.0) -> Tuple[int, int]:
    """Return ERP resize target (height, width) aligned to patch-size grid."""
    height = _round_to_patch_multiple(target_height, patch_size)
    width = _round_to_patch_multiple(int(height * aspect_ratio), patch_size)
    return height, width


def build_erp_raymap(height: int, width: int, device=None, dtype=torch.float32, normalize_area: bool = True) -> torch.Tensor:
    """Build dense ERP raymap with 4 channels: (x, y, z, area_weight)."""
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)

    lat = 0.5 * math.pi - y * math.pi
    lon = x * (2.0 * math.pi) - math.pi

    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")

    cos_lat = torch.cos(lat_grid)
    ray_x = cos_lat * torch.sin(lon_grid)
    ray_y = torch.sin(lat_grid)
    ray_z = cos_lat * torch.cos(lon_grid)

    area_weight = cos_lat.clamp_min(1e-6)
    if normalize_area:
        area_weight = area_weight / area_weight.mean().clamp_min(1e-6)

    return torch.stack([ray_x, ray_y, ray_z, area_weight], dim=-1)


def build_erp_raymap_batch(images: torch.Tensor, normalize_area: bool = True) -> torch.Tensor:
    """Build batched dense ERP raymaps for images shaped [B, S, C, H, W]."""
    if images.ndim != 5:
        raise ValueError(f"Expected images with shape [B, S, C, H, W], got {tuple(images.shape)}")

    bsz, seq, _, height, width = images.shape
    raymap = build_erp_raymap(
        height,
        width,
        device=images.device,
        dtype=images.dtype,
        normalize_area=normalize_area,
    )
    return raymap.view(1, 1, height, width, 4).expand(bsz, seq, height, width, 4).contiguous()


def rotate_raymap_to_world(raymap: torch.Tensor, camera_poses: torch.Tensor) -> torch.Tensor:
    """Rotate camera-space ERP raymap directions to world space using c2w camera poses."""
    if raymap.ndim != 5 or raymap.shape[-1] != 4:
        raise ValueError(f"Expected raymap shape [B, S, H, W, 4], got {tuple(raymap.shape)}")
    if camera_poses.ndim != 4 or camera_poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected camera_poses shape [B, S, 4, 4], got {tuple(camera_poses.shape)}")

    dirs_cam = raymap[..., :3]
    weights = raymap[..., 3:4]
    rotation = camera_poses[..., :3, :3]

    dirs_world = torch.einsum("bsij,bshwj->bshwi", rotation, dirs_cam)
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat([dirs_world, weights], dim=-1)


def circular_pad_last_dim(tensor: torch.Tensor, pad_left: int = 0, pad_right: int = 0) -> torch.Tensor:
    """Circularly pad along last dimension using wrapped slices."""
    if pad_left < 0 or pad_right < 0:
        raise ValueError("pad_left and pad_right must be non-negative")

    if pad_left == 0 and pad_right == 0:
        return tensor

    pieces = []
    if pad_left > 0:
        pieces.append(tensor[..., -pad_left:])
    pieces.append(tensor)
    if pad_right > 0:
        pieces.append(tensor[..., :pad_right])
    return torch.cat(pieces, dim=-1)


def apply_erp_circular_padding(images: torch.Tensor, pad_pixels: int = 0) -> torch.Tensor:
    """Apply horizontal circular padding for tensors [C,H,W], [B,C,H,W], or [B,S,C,H,W]."""
    if pad_pixels <= 0:
        return images

    if images.ndim == 3:
        return circular_pad_last_dim(images, pad_pixels, pad_pixels)
    if images.ndim == 4:
        return circular_pad_last_dim(images, pad_pixels, pad_pixels)
    if images.ndim == 5:
        return circular_pad_last_dim(images, pad_pixels, pad_pixels)

    raise ValueError(
        f"Unsupported tensor ndim={images.ndim}. Expected 3D/4D/5D image tensor."
    )


def remove_erp_circular_padding(images: torch.Tensor, pad_pixels: int = 0) -> torch.Tensor:
    """Remove horizontal circular padding added by apply_erp_circular_padding."""
    if pad_pixels <= 0:
        return images
    if images.shape[-1] <= 2 * pad_pixels:
        raise ValueError("Cannot remove circular padding: width is smaller than 2 * pad_pixels")
    return images[..., pad_pixels:-pad_pixels]
