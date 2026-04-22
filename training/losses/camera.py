import torch

from .container import BaseLoss
from .utils import check_and_fix_inf_nan

from src.models.utils.geometry import closed_form_inverse_se3
from src.models.utils.camera_utils import camera_params_to_vector


class CameraLoss(BaseLoss):
    """Camera pose loss"""
    
    def __init__(self, weight_T=1.0, weight_R=1.0, weight_fl=0.5, camera_model="spherical"):
        super().__init__()
        self.weight_T = weight_T
        self.weight_R = weight_R
        self.weight_fl = weight_fl
        self.camera_model = camera_model

    def _quat_geodesic(self, pred_quat, gt_quat):
        pred_quat = torch.nn.functional.normalize(pred_quat, dim=-1, eps=1e-8)
        gt_quat = torch.nn.functional.normalize(gt_quat, dim=-1, eps=1e-8)
        cos_theta = torch.abs((pred_quat * gt_quat).sum(dim=-1)).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        return 2.0 * torch.acos(cos_theta)

    def _get_valid_frame_mask(self, gts, shape, device):
        if "valid_mask" not in gts or gts["valid_mask"] is None:
            return torch.ones(shape, device=device, dtype=torch.bool)

        point_masks = gts["valid_mask"]
        if point_masks.ndim < 4:
            return torch.ones(shape, device=device, dtype=torch.bool)

        frame_valid = point_masks.sum(dim=[-1, -2]) > 100
        if frame_valid.shape != shape:
            return torch.ones(shape, device=device, dtype=torch.bool)
        return frame_valid

    def _masked_mean(self, values, mask):
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        weight = mask.float()
        if weight.shape != values.shape:
            weight = weight.expand_as(values)
        denom = weight.sum().clamp_min(1.0)
        return (values * weight).sum() / denom
    
    def compute_loss(self, preds, gts):
        B, S, _, H, W = gts['img'].shape
    
        # Convert ground truth camera matrices to compact vector representation
        # Extrinsics: world-to-camera -> camera-to-world transformation
        gt_extrinsics = gts['camera_poses']
        gt_intrinsics = gts.get('camera_intrs', None)
        gt_extrinsics = closed_form_inverse_se3(gt_extrinsics.flatten(0, 1)).reshape(B, S, 4, 4)
        
        # Encode ground truth as camera vector according to camera model.
        gt_camera_params = camera_params_to_vector(
            gt_extrinsics,
            gt_intrinsics,
            (H, W),
            camera_model=self.camera_model,
        )
        
        # Extract predicted camera parameters (B, S, 9)
        pred_camera_params = preds['camera_params']
        
        # Check if frames have valid points (at least 100 valid pixels)
        valid_frame_mask = self._get_valid_frame_mask(gts, (B, S), pred_camera_params.device)
        
        # If no valid frames, return zero loss
        if valid_frame_mask.sum() == 0:
            zero = (pred_camera_params * 0).mean()
            loss_dict = {
                "loss_T": zero,
                "loss_R": zero,
                "loss_FL": zero,
            }
        else:
            # Translation and rotation losses are always supervised.
            loss_T = (pred_camera_params[..., :3] - gt_camera_params[..., :3]).abs()
            loss_R = self._quat_geodesic(pred_camera_params[..., 3:7], gt_camera_params[..., 3:7])

            # Tail loss depends on camera model: FoV (perspective) or ERP center offsets (spherical).
            pred_tail = pred_camera_params[..., 7:9] if pred_camera_params.shape[-1] >= 9 else None
            gt_tail = gt_camera_params[..., 7:9] if gt_camera_params.shape[-1] >= 9 else None
            if pred_tail is None or gt_tail is None:
                loss_FL = torch.zeros_like(loss_R)
            else:
                loss_FL = (pred_tail - gt_tail).abs()
            
            # Check and fix numerical issues (NaN/Inf) in loss components
            loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
            loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
            loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")
            
            # Clamp translation loss to prevent gradient explosion, then average
            loss_T = self._masked_mean(loss_T.clamp(max=100), valid_frame_mask)
            loss_R = self._masked_mean(loss_R, valid_frame_mask)
            loss_FL = self._masked_mean(loss_FL, valid_frame_mask)
            
            loss_dict = {
                "loss_T": loss_T,
                "loss_R": loss_R,
                "loss_FL": loss_FL,
            }
        
        # Compute total camera loss
        loss = loss_dict["loss_T"] * self.weight_T + loss_dict["loss_R"] * self.weight_R + loss_dict["loss_FL"] * self.weight_fl
        return loss, loss_dict
    
    @property
    def name(self):
        return f"CameraLoss"
    