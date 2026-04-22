import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch


def _to_4x4_matrix(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = arr
        return out
    if arr.size == 16:
        return arr.reshape(4, 4).astype(np.float32)
    if arr.size == 12:
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = arr.reshape(3, 4)
        return out
    raise ValueError(f"Unsupported pose shape {arr.shape} with size={arr.size}")


def _iter_pose_entries(data) -> Iterable:
    if isinstance(data, dict):
        if "poses" in data:
            return data["poses"]
        if "camera_poses" in data:
            return data["camera_poses"]
        if "frames" in data:
            return data["frames"]
        raise ValueError("JSON pose file must contain one of keys: poses, camera_poses, frames")
    if isinstance(data, list):
        return data
    raise ValueError("JSON pose file must be a dict or list")


def _parse_pose_entry(entry) -> np.ndarray:
    if isinstance(entry, dict):
        for key in ["pose", "transform_matrix", "matrix", "c2w", "w2c"]:
            if key in entry:
                return _to_4x4_matrix(entry[key])
        raise ValueError("Pose entry dict must contain one of keys: pose, transform_matrix, matrix, c2w, w2c")
    return _to_4x4_matrix(entry)


def load_poses_json(file_path: str) -> np.ndarray:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = _iter_pose_entries(data)
    poses = [_parse_pose_entry(entry) for entry in entries]
    if not poses:
        raise ValueError("No poses found in JSON file")
    return np.stack(poses, axis=0).astype(np.float32)


def load_poses_txt(file_path: str) -> np.ndarray:
    poses: List[np.ndarray] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(v) for v in line.replace(",", " ").split()]
            poses.append(_to_4x4_matrix(values))

    if not poses:
        raise ValueError("No poses found in TXT file")
    return np.stack(poses, axis=0).astype(np.float32)


def load_pose_sequence(file_path: str, pose_format: str = "c2w") -> torch.Tensor:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        poses = load_poses_json(str(path))
    elif suffix in {".txt", ".csv"}:
        poses = load_poses_txt(str(path))
    else:
        raise ValueError(f"Unsupported pose file extension: {suffix}")

    if pose_format not in {"c2w", "w2c"}:
        raise ValueError("pose_format must be 'c2w' or 'w2c'")

    poses_t = torch.from_numpy(poses)
    if pose_format == "w2c":
        poses_t = torch.linalg.inv(poses_t)

    if torch.isnan(poses_t).any() or torch.isinf(poses_t).any():
        raise ValueError("Pose file contains NaN/Inf values")

    return poses_t.float()
