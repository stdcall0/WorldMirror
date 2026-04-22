import os
import os.path as osp
import re

import cv2
import numpy as np
from PIL import Image

from training.data.multiview_dataset import MultiViewDataset
from training.utils.image import imread_cv2


TAG_FLOAT = 202021.25


def read_depth_dpt(path: str) -> np.ndarray:
    """Read Sintel-style .dpt depth file as float32 HxW array."""
    with open(path, "rb") as f:
        tag = np.fromfile(f, dtype=np.float32, count=1)
        if tag.size != 1 or float(tag[0]) != TAG_FLOAT:
            raise ValueError(f"Invalid .dpt tag in {path}: got {tag}")

        width = np.fromfile(f, dtype=np.int32, count=1)
        height = np.fromfile(f, dtype=np.int32, count=1)
        if width.size != 1 or height.size != 1:
            raise ValueError(f"Invalid .dpt shape header in {path}")

        w = int(width[0])
        h = int(height[0])
        if w <= 0 or h <= 0 or (w * h) > 100000000:
            raise ValueError(f"Invalid .dpt size in {path}: w={w}, h={h}")

        depth = np.fromfile(f, dtype=np.float32, count=-1)
        if depth.size != (w * h):
            raise ValueError(f"Unexpected .dpt payload size in {path}: got {depth.size}, expected {w*h}")

    return depth.reshape(h, w).astype(np.float32)


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion in XYZW order to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-8:
        return np.eye(3, dtype=np.float32)
    q = q / n
    x, y, z, w = q.tolist()

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def parse_pose_txt(path: str, quat_order: str = "xywz") -> np.ndarray:
    """Read pose txt with 7 values: tx ty tz + quaternion, return 4x4 c2w."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line = f.readline().strip()

    cleaned = re.sub(r"[^0-9eE+\-\.\s]", " ", line)
    arr = np.fromstring(cleaned, sep=" ", dtype=np.float32)
    if arr.size < 7:
        raise ValueError(f"Invalid pose file {path}: parsed {arr.size} values")

    center = arr[:3]
    quat_raw = arr[3:7]

    order = quat_order.lower()
    if order == "xyzw":
        quat_xyzw = quat_raw
    elif order == "xywz":
        quat_xyzw = np.array([quat_raw[0], quat_raw[1], quat_raw[3], quat_raw[2]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported quaternion order: {quat_order}")

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = quat_xyzw_to_rotmat(quat_xyzw)
    pose[:3, 3] = center
    return pose


def build_erp_intrinsics(width: int, height: int) -> np.ndarray:
    """Build pseudo intrinsics for ERP geometry helpers."""
    fx = float(width) / (2.0 * np.pi)
    fy = float(height) / np.pi
    cx = float(width) * 0.5
    cy = float(height) * 0.5
    intr = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intr


class Matterport3D_ERP(MultiViewDataset):
    """Matterport3D ERP multi-view training dataset.

    Expected layout:
      ROOT/
        scene_id/
          <stem>_rgb.png
          <stem>_depth.dpt
          <stem>_pose.txt
          <stem>_vis.png (optional)
    """

    def __init__(
        self,
        *args,
        split,
        ROOT,
        max_interval=12,
        scene_split_seed=42,
        train_scene_ratio=0.9,
        quat_order="xywz",
        **kwargs,
    ):
        self.ROOT = osp.expanduser(ROOT)
        self.video = True
        self.is_metric = True
        self.max_interval = int(max_interval)
        self.scene_split_seed = int(scene_split_seed)
        self.train_scene_ratio = float(train_scene_ratio)
        self.quat_order = quat_order

        super().__init__(*args, **kwargs)
        self.split = str(split).lower()

        self.scenes = []
        self.scene_stems = []
        self.start_samples = []
        self._load_data()

    def _split_scenes(self, scenes):
        if len(scenes) <= 1:
            return scenes, scenes

        rng = np.random.default_rng(self.scene_split_seed)
        perm = np.array(scenes, dtype=object)
        rng.shuffle(perm)

        train_count = int(round(len(perm) * self.train_scene_ratio))
        train_count = min(max(train_count, 1), len(perm) - 1)
        train_scenes = sorted(perm[:train_count].tolist())
        val_scenes = sorted(perm[train_count:].tolist())
        return train_scenes, val_scenes

    def _collect_scene_stems(self, scene_dir):
        stems = []
        for name in sorted(os.listdir(scene_dir)):
            if not name.endswith("_rgb.png"):
                continue
            stem = name[: -len("_rgb.png")]
            depth_path = osp.join(scene_dir, f"{stem}_depth.dpt")
            pose_path = osp.join(scene_dir, f"{stem}_pose.txt")
            if osp.isfile(depth_path) and osp.isfile(pose_path):
                stems.append(stem)
        return stems

    def _load_data(self):
        if not osp.isdir(self.ROOT):
            raise FileNotFoundError(f"Matterport3D root not found: {self.ROOT}")

        all_scenes = sorted([d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))])
        if len(all_scenes) == 0:
            raise RuntimeError(f"No scene folders found in {self.ROOT}")

        train_scenes, val_scenes = self._split_scenes(all_scenes)
        if self.split in ("train", "training"):
            use_scenes = train_scenes
        elif self.split in ("val", "valid", "validation", "test"):
            use_scenes = val_scenes
        else:
            use_scenes = all_scenes

        cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)

        for scene in use_scenes:
            scene_dir = osp.join(self.ROOT, scene)
            stems = self._collect_scene_stems(scene_dir)
            if len(stems) < cut_off:
                continue

            scene_idx = len(self.scenes)
            self.scenes.append(scene)
            self.scene_stems.append(stems)
            for start_idx in range(0, len(stems) - cut_off + 1):
                self.start_samples.append((scene_idx, start_idx))

        if len(self.start_samples) == 0:
            raise RuntimeError(
                f"No valid Matterport3D samples found under {self.ROOT} for split={self.split}. "
                f"Need at least {cut_off} matched frames (rgb/depth/pose) per scene."
            )

    def __len__(self):
        # Mirror HyperSim behavior to increase sampling diversity.
        return len(self.start_samples) * 10

    def _fetch_views(self, idx, resolution, rng, num_views, *args, **kwargs):
        base_idx = (idx // 10) % len(self.start_samples)
        scene_idx, start_idx = self.start_samples[base_idx]
        scene = self.scenes[scene_idx]
        stems = self.scene_stems[scene_idx]

        all_ids = list(range(len(stems)))
        max_interval = max(1, min(self.max_interval, len(stems) - 1))
        pos, ordered_video = self.extract_view_sequence(
            num_views,
            start_idx,
            all_ids,
            rng,
            max_interval=max_interval,
            block_shuffle=16,
        )

        target_w, target_h = None, None
        if resolution is not None:
            target_w, target_h = int(resolution[0]), int(resolution[1])
        views = []
        for view_pos in pos:
            stem = stems[view_pos]
            scene_dir = osp.join(self.ROOT, scene)

            rgb_path = osp.join(scene_dir, f"{stem}_rgb.png")
            depth_path = osp.join(scene_dir, f"{stem}_depth.dpt")
            pose_path = osp.join(scene_dir, f"{stem}_pose.txt")

            rgb = imread_cv2(rgb_path, cv2.IMREAD_COLOR)
            depth = read_depth_dpt(depth_path)
            pose = parse_pose_txt(pose_path, quat_order=self.quat_order)

            if target_w is None or target_h is None:
                target_h, target_w = int(rgb.shape[0]), int(rgb.shape[1])

            if rgb.shape[1] != target_w or rgb.shape[0] != target_h:
                rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            if depth.shape[1] != target_w or depth.shape[0] != target_h:
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            depth = depth.astype(np.float32)
            depth[~np.isfinite(depth)] = 0.0
            depth[depth <= 0.0] = 0.0

            intrinsics = build_erp_intrinsics(target_w, target_h)

            views.append(
                dict(
                    img=Image.fromarray(rgb),
                    depthmap=depth,
                    camera_poses=pose.astype(np.float32),
                    camera_intrs=intrinsics.astype(np.float32),
                    dataset="matterport3d_erp",
                    label=f"{scene}_{stem}",
                    instance=f"{scene}_{stem}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    is_erp=True,
                    erp_curriculum=True,
                    nvs_sample=True,
                    scale_norm=True,
                    cam_align=True,
                )
            )

        assert len(views) == num_views
        return views
