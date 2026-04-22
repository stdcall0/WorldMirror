from typing import Callable, Optional

import numpy as np
from torch.utils.data import Sampler, DistributedSampler


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self._aspect_ratio = None
        self._view_idxs = None
        self._source_view_idxs = None
        self._target_pixels = None
        self._target_fov = None
        self._target_circular_pad_pixels = 0

    def __iter__(self):
        """
        Yields a sequence of (sample_idxs, _aspect_ratio, (_view_idxs, _source_view_idxs, _target_fov, _target_circular_pad_pixels), _target_pixels).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for sample_idxs in indices_iter:
            yield (
                sample_idxs,
                self._aspect_ratio,
                (
                    self._view_idxs,
                    self._source_view_idxs,
                    self._target_fov,
                    self._target_circular_pad_pixels,
                ),
                self._target_pixels,
            )

    def update_parameters(
        self,
        aspect_ratio,
        view_idxs,
        source_view_idxs,
        target_pixels,
        target_fov=None,
        target_circular_pad_pixels=0,
    ):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            view_idxs: The number of images to set.
            source_view_idxs: The number of source images to set.
            target_pixels: The number of pixels per image to set.
            target_fov: The target FoV value (degrees) for ERP curriculum.
            target_circular_pad_pixels: Horizontal circular padding pixels for ERP post-curriculum augmentation.
        """
        self._aspect_ratio = aspect_ratio
        self._view_idxs = view_idxs
        self._source_view_idxs = source_view_idxs
        self._target_pixels = target_pixels
        self._target_fov = target_fov
        self._target_circular_pad_pixels = int(target_circular_pad_pixels)


class DynamicBatchSampler(Sampler):
    def __init__(self, sampler, min_view_size, max_view_size, epoch=0, seed=42,
                 max_img_per_gpu=24, aspect_ratio_range=None, num_pixels_range=[100000, 250000], decay=0.5, allview_p=0.2,
                 enable_fov_curriculum=False, fov_start_deg=90.0, fov_end_deg=360.0, fov_curriculum_total_steps=100000,
                 enable_circular_pad_after_fov_unlock=False,
                 circular_pad_pixels_after_fov_unlock=0,
                 circular_pad_prob_after_fov_unlock=0.5):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            min_view_size: min_images numbers per sample.
            max_view_size: max_images numbers per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            num_pixels_range: List containing [min_pixels, max_pixels] for target resolution sampling.
            decay: Decay parameter for source view sampling weight distribution. Higher values favor more source views.
            allview_p: Probability of using all views as source views (no novel views).
            enable_fov_curriculum: Whether to sample ERP FoV using a continuous curriculum.
            fov_start_deg: Curriculum starting FoV in degrees.
            fov_end_deg: Curriculum final FoV in degrees.
            fov_curriculum_total_steps: Number of sampler steps to reach final FoV.
            enable_circular_pad_after_fov_unlock: Whether to enable ERP circular padding after FoV curriculum is fully unlocked.
            circular_pad_pixels_after_fov_unlock: Horizontal circular padding pixels when the above augmentation is enabled.
            circular_pad_prob_after_fov_unlock: Sampling probability for circular padding after FoV curriculum is unlocked.
        """
        self.sampler = sampler
        self.min_view_size = min_view_size
        self.max_view_size = max_view_size
        self.rng = np.random.default_rng(seed=seed)
        self.aspect_ratio_range = aspect_ratio_range
        self.num_pixels_range = num_pixels_range

        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: 1.0 for num_images in range(min_view_size, max_view_size+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if min_view_size <= n <= max_view_size])

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        self.decay = decay
        self.allview_p = allview_p
        self.enable_fov_curriculum = bool(enable_fov_curriculum)
        self.fov_start_deg = float(fov_start_deg)
        self.fov_end_deg = float(fov_end_deg)
        if fov_curriculum_total_steps is None or int(fov_curriculum_total_steps) <= 0:
            self.fov_curriculum_total_steps = None
        else:
            self.fov_curriculum_total_steps = int(fov_curriculum_total_steps)
        self.enable_circular_pad_after_fov_unlock = bool(enable_circular_pad_after_fov_unlock)
        self.circular_pad_pixels_after_fov_unlock = max(0, int(circular_pad_pixels_after_fov_unlock))
        self.circular_pad_prob_after_fov_unlock = float(np.clip(circular_pad_prob_after_fov_unlock, 0.0, 1.0))
        self.global_batch_step = 0

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)
    
    def _sample_source_view_idxs(self, _view_idxs):
        """
        Sample the number of source views using a weighted distribution.
        Higher number of source views have relatively higher probability.
        """
        if _view_idxs <= 2 or self.rng.random() < self.allview_p:
            return _view_idxs
        
        # Valid range for source views count
        min_source = max(self.min_view_size, int(_view_idxs//2 + 0.5))
        max_source = min(self.max_view_size - 1, _view_idxs - 1)  # Reserve space for at least 1 novel view
        
        if min_source > max_source:
            return _view_idxs
        
        # Create weights: higher counts have higher weights, but not linear growth
        counts = list(range(min_source, max_source + 1))
        weights = [(count - min_source + 1)**self.decay for count in counts]

        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        return self.rng.choice(counts, p=weights)

    def _sample_view_idxs_and_ar_and_tp(self):
        """Sample view_idxs and aspect_ratio according to the specified rules."""
        _view_idxs = int(self.rng.choice(self.possible_nums, p=self.normalized_weights))
        _source_view_idxs = self._sample_source_view_idxs(_view_idxs)
        if self.aspect_ratio_range is not None:
            _aspect_ratio = float(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1], size=1))
        else:
            _aspect_ratio = 1.0
        min_pixels = self.num_pixels_range[0]
        max_pixels = self.num_pixels_range[1]
        _target_pixels = int(self.rng.integers(min_pixels, max_pixels + 1))
        _target_fov = self._sample_target_fov_deg()
        _target_circular_pad_pixels = self._sample_target_circular_pad_pixels()

        return _view_idxs, _source_view_idxs, _aspect_ratio, _target_pixels, _target_fov, _target_circular_pad_pixels

    def _is_fov_curriculum_unlocked(self):
        if not self.enable_fov_curriculum:
            return False
        if self.fov_curriculum_total_steps is None:
            return True
        unlock_step = max(0, self.fov_curriculum_total_steps - 1)
        return self.global_batch_step >= unlock_step

    def _sample_target_circular_pad_pixels(self):
        """Sample circular padding pixels only after FoV curriculum is fully unlocked."""
        if not self.enable_circular_pad_after_fov_unlock:
            return 0
        if self.circular_pad_pixels_after_fov_unlock <= 0:
            return 0
        if not self._is_fov_curriculum_unlocked():
            return 0
        if self.rng.random() <= self.circular_pad_prob_after_fov_unlock:
            return int(self.circular_pad_pixels_after_fov_unlock)
        return 0

    def _sample_target_fov_deg(self):
        """Sample target FoV (degrees) with continuous step-wise curriculum."""
        if not self.enable_fov_curriculum:
            return None

        start_fov = min(self.fov_start_deg, self.fov_end_deg)
        end_fov = max(self.fov_start_deg, self.fov_end_deg)

        if self.fov_curriculum_total_steps is None:
            progress = 1.0
        else:
            denom = max(1, self.fov_curriculum_total_steps - 1)
            progress = min(1.0, float(self.global_batch_step) / float(denom))

        current_max_fov = start_fov + (end_fov - start_fov) * progress
        if current_max_fov <= start_fov + 1e-6:
            return float(start_fov)

        return float(self.rng.uniform(start_fov, current_max_fov))

    def _batch_size_for(self, view_idxs: int) -> int:
        """Calculate batch_size based on max_img_per_gpu and view count (floor division, minimum 1)."""
        bs = int(np.floor(self.max_img_per_gpu / max(1, view_idxs)))
        return max(1, bs)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng = np.random.default_rng(seed=epoch + 777)

    def __iter__(self):
        """
        Dynamically sample and consume the underlying sampler.
        All samples in each batch share the same aspect_ratio / view_idxs.
        """
        sampler_iterator = iter(self.sampler)
        remaining = len(self.sampler)
        
        while remaining > 0:
            v, sv, ar, tp, target_fov, target_circular_pad_pixels = self._sample_view_idxs_and_ar_and_tp()
            bs = self._batch_size_for(v)
            
            # Synchronize dynamic parameters to the underlying sampler (for dataset usage)
            self.sampler.update_parameters(
                aspect_ratio=ar,
                view_idxs=v,
                source_view_idxs=sv,
                target_pixels=tp,
                target_fov=target_fov,
                target_circular_pad_pixels=target_circular_pad_pixels,
            )
            
            current_batch = []
            for _ in range(bs):
                try:
                    item = next(sampler_iterator)
                    current_batch.append(item)
                    remaining -= 1
                except StopIteration:
                    break
            
            if current_batch:
                self.global_batch_step += 1
                yield current_batch

    def __len__(self):
        # Return the length of the underlying sampler as an estimate of sample count
        return len(self.sampler)