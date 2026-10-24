"""Object regions from Segment Anything (SAM) for bleed-free colour refinement.

SAM (Kirillov et al., 2023, Apache-2.0) is used only as a segmenter: it says
where one object ends and the next begins, never which colour anything is.
The stock ``SamAutomaticMaskGenerator`` spends most of its time run-length
encoding thousands of full-resolution masks on the CPU; this module keeps the
whole grid decode, quality filtering and duplicate suppression on the GPU at
SAM's 256x256 mask resolution and only upsamples the masks it keeps.
"""
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "pretrained" / "sam" / "sam_vit_b_01ec64.pth"


def _fill_uncovered(labels, min_fraction=0.0005):
    """Give every connected uncovered area its own id so colour cannot leak into it either."""
    uncovered = (labels < 0).astype(np.uint8)
    if not uncovered.any():
        return labels
    count, components, stats, _ = cv2.connectedComponentsWithStats(uncovered, connectivity=4)
    next_id = int(labels.max()) + 1
    minimum = min_fraction * labels.size
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= minimum:
            labels[components == component] = next_id
            next_id += 1
    return labels


class ObjectSegmenter:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, model_type="vit_b", device=None,
                 points_per_side=20, points_per_batch=32, iou_threshold=0.80,
                 stability_threshold=0.88, duplicate_iou=0.70, min_area_fraction=0.0005):
        from segment_anything import SamPredictor, sam_model_registry

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(self.device).eval()
        self.predictor = SamPredictor(model)
        self.points_per_side, self.points_per_batch = points_per_side, points_per_batch
        self.iou_threshold, self.stability_threshold = iou_threshold, stability_threshold
        self.duplicate_iou, self.min_area_fraction = duplicate_iou, min_area_fraction
        self._lock = threading.Lock()

    @classmethod
    def try_create(cls, checkpoint=DEFAULT_CHECKPOINT, device=None):
        """Return a segmenter, or ``None`` when the package or weights are unavailable."""
        try:
            if not Path(checkpoint).is_file():
                return None
            return cls(checkpoint, device=device)
        except ImportError:
            return None

    @torch.inference_mode()
    def labels(self, rgb):
        """Integer object id per pixel of ``rgb`` (uint8 HxWx3); -1 where no object was found."""
        height, width = rgb.shape[:2]
        # The mask decoder copies the image embedding once per prompt; small FP16 batches
        # keep that copy inside a 6 GB laptop GPU instead of spilling to shared memory.
        with self._lock, torch.autocast(self.device.type, dtype=torch.float16,
                                        enabled=self.device.type == "cuda"):
            self.predictor.set_image(np.ascontiguousarray(rgb))
            logits, scores = self._grid_masks(height, width, self._grid(self.points_per_side, height, width),
                                              self.iou_threshold, self.stability_threshold)
            # Second pass: a denser grid, but only where the first pass found no object.
            # Small heads, hair and dark clothing are easily missed by a 20x20 grid.
            dense = self._grid(2 * self.points_per_side, height, width)
            if logits is not None:
                covered = (F.interpolate((logits > 0).any(dim=0)[None, None].float(),
                                         size=(height, width))[0, 0] > 0)
                rows = dense[:, 1].long().clamp(max=height - 1)
                cols = dense[:, 0].long().clamp(max=width - 1)
                dense = dense[~covered[rows, cols]]
            extra_logits, extra_scores = self._grid_masks(height, width, dense, self.iou_threshold - 0.08,
                                                          self.stability_threshold - 0.06)
            if extra_logits is not None:
                logits = extra_logits if logits is None else torch.cat([logits, extra_logits])
                scores = extra_scores if scores is None else torch.cat([scores, extra_scores])
        if logits is None:
            return _fill_uncovered(np.full((height, width), -1, dtype=np.int32))
        keep = self._suppress_duplicates(logits > 0, scores)
        return _fill_uncovered(self._paint(logits[keep], height, width))

    def _grid(self, per_side, height, width):
        steps = (torch.arange(per_side, device=self.device) + 0.5) / per_side
        return torch.stack(torch.meshgrid(steps * width, steps * height, indexing="xy"), dim=-1).reshape(-1, 2)

    def _grid_masks(self, height, width, grid, iou_threshold, stability_threshold):
        if not len(grid):
            return None, None
        model, predictor = self.predictor.model, self.predictor
        valid_h, valid_w = [int(np.ceil(size / 4)) for size in predictor.input_size]
        kept_logits, kept_scores = [], []
        for start in range(0, len(grid), self.points_per_batch):
            points = predictor.transform.apply_coords_torch(grid[start:start + self.points_per_batch],
                                                            (height, width))
            sparse, dense = model.prompt_encoder(
                points=(points[:, None, :], torch.ones(len(points), 1, device=self.device)),
                boxes=None, masks=None)
            low_res, iou = model.mask_decoder(
                image_embeddings=predictor.features, image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True)
            low_res = low_res[..., :valid_h, :valid_w].flatten(0, 1)
            iou = iou.flatten()
            # Stability: the mask barely changes when the logit threshold moves by +-1.
            inner = (low_res > 1.0).sum(dim=(1, 2)).float()
            outer = (low_res > -1.0).sum(dim=(1, 2)).float().clamp_min(1)
            stability = inner / outer
            area = (low_res > 0).float().mean(dim=(1, 2))
            good = (iou > iou_threshold) & (stability > stability_threshold) & \
                   (area > self.min_area_fraction) & (area < 0.95)
            kept_logits.append(low_res[good])
            kept_scores.append((iou * stability)[good])
        logits = torch.cat(kept_logits)
        return (logits, torch.cat(kept_scores)) if len(logits) else (None, None)

    def _suppress_duplicates(self, masks, scores):
        order = torch.argsort(scores, descending=True)
        small = F.adaptive_avg_pool2d(masks[order].float()[:, None], (64, 64)).flatten(1) > 0.5
        small = small.half()
        intersection = small @ small.T
        area = small.sum(dim=1)
        overlap = (intersection / (area[:, None] + area[None, :] - intersection).clamp_min(1)).cpu().numpy()
        keep = []
        for index in range(len(order)):
            if all(overlap[index, kept] < self.duplicate_iou for kept in keep):
                keep.append(index)
        return order[keep]

    def _paint(self, logits, height, width):
        labels = torch.full((height, width), -1, dtype=torch.int32, device=self.device)
        areas = (logits > 0).float().mean(dim=(1, 2))
        # Large regions first so that smaller, more specific objects inside them win.
        for index in torch.argsort(areas, descending=True).tolist():
            mask = F.interpolate(logits[index][None, None], size=(height, width), mode="bilinear",
                                 align_corners=False)[0, 0] > 0
            labels[mask] = index
        return labels.cpu().numpy()
