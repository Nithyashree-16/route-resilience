from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from src.data.occlusion import apply_occlusions


class RoadDataset(Dataset):
    """
    Road segmentation dataset for SpaceNet and compatible datasets.

    Training:
        - Reads image/mask tile pairs from the manifest.
        - Normalizes imagery using training-set statistics.
        - Optionally applies synthetic occlusion to positive examples.
        - Returns the actual spatial occlusion mask.

    Validation/Test:
        - Normally clean when occlusion_enabled=False.
        - Can explicitly enable synthetic occlusion for robustness evaluation.
        - When enabled, occlusion_probability controls how often an image
          receives synthetic occlusion.

    Returned dictionary:
        image:
            Float tensor, shape [3, H, W]

        mask:
            Float tensor, shape [1, H, W]

        scene_id:
            Source scene identifier

        tile:
            Tile filename

        has_road:
            Whether the tile contains reference road pixels

        occlusion_effects:
            List of applied effect names

        occlusion_mask:
            Float tensor, shape [1, H, W]
            1 = spatially occluded
            0 = not spatially occluded
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        stats_path: str | Path,
        occlusion_enabled: bool = False,
        occlusion_probability: float = 0.0,
    ) -> None:
        super().__init__()

        self.manifest_path = Path(
            manifest_path
        )

        self.stats_path = Path(
            stats_path
        )

        self.split = str(split)

        if self.split not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(
                "split must be one of: train, val, test"
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        if not self.stats_path.exists():
            raise FileNotFoundError(
                f"Statistics file not found: {self.stats_path}"
            )

        manifest = pd.read_csv(
            self.manifest_path
        )

        required_columns = {
            "split",
            "image_path",
            "mask_path",
            "scene_id",
            "tile",
            "has_road",
        }

        missing = required_columns.difference(
            manifest.columns
        )

        if missing:
            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing))
            )

        self.data = manifest[
            manifest["split"] == self.split
        ].reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(
                f"No samples found for split='{self.split}'"
            )

        stats = self._load_stats(
            self.stats_path
        )

        self.mean = np.asarray(
            stats["mean"],
            dtype=np.float32,
        )

        self.std = np.asarray(
            stats["std"],
            dtype=np.float32,
        )

        self.scale_max = np.asarray(
            stats.get(
                "max",
                stats.get(
                    "scale_max",
                    [1.0, 1.0, 1.0],
                ),
            ),
            dtype=np.float32,
        )

        if self.mean.shape != (3,):
            raise ValueError(
                f"Expected 3 channel mean values, got "
                f"{self.mean.shape}"
            )

        if self.std.shape != (3,):
            raise ValueError(
                f"Expected 3 channel std values, got "
                f"{self.std.shape}"
            )

        if self.scale_max.shape != (3,):
            raise ValueError(
                f"Expected 3 channel max values, got "
                f"{self.scale_max.shape}"
            )

        if np.any(self.std <= 0):
            raise ValueError(
                "Channel std must be > 0."
            )

        if np.any(self.scale_max <= 0):
            raise ValueError(
                "Channel max must be > 0."
            )

        # IMPORTANT:
        # Do not restrict this to split == "train".
        #
        # Training can use synthetic occlusion.
        # Validation can also explicitly request synthetic occlusion
        # for Occlusion Recall evaluation.
        #
        # Test remains clean because its caller will normally leave
        # occlusion_enabled=False.
        self.occlusion_enabled = bool(
            occlusion_enabled
        )

        self.occlusion_probability = float(
            occlusion_probability
        )

        if not 0.0 <= self.occlusion_probability <= 1.0:
            raise ValueError(
                "occlusion_probability must be between 0 and 1."
            )

        self.project_root = (
            self.manifest_path.resolve().parents[2]
        )

    # ------------------------------------------------------------------
    # Statistics loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_stats(
        stats_path: Path,
    ) -> dict:
        import json

        with stats_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            stats = json.load(f)

        if "mean" not in stats:
            raise ValueError(
                "Statistics file must contain 'mean'."
            )

        if "std" not in stats:
            raise ValueError(
                "Statistics file must contain 'std'."
            )

        return stats

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------

    def _resolve_path(
        self,
        relative_path: str,
    ) -> Path:
        path = Path(
            relative_path
        )

        if path.is_absolute():
            return path

        return self.project_root / path

    # ------------------------------------------------------------------
    # Image normalization
    # ------------------------------------------------------------------

    def _normalize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        return (
            image - self.mean[:, None, None]
        ) / self.std[:, None, None]

    # ------------------------------------------------------------------
    # Synthetic occlusion
    # ------------------------------------------------------------------

    def _apply_occlusion(
        self,
        image: np.ndarray,
        has_road: bool,
    ) -> tuple[
        np.ndarray,
        list[str],
        np.ndarray,
    ]:
        """
        Apply synthetic occlusion and return the exact spatial mask.

        Returns:
            image:
                Raw-scale image after augmentation.

            effects:
                Names of applied augmentation effects.

            occlusion_mask:
                H x W binary mask.
                1 = spatially occluded
                0 = not spatially occluded
        """
        h, w = image.shape[1:]

        empty_mask = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        if not self.occlusion_enabled:
            return (
                image,
                [],
                empty_mask,
            )

        # Preserve the original behavior:
        # only positive tiles receive synthetic occlusion.
        if not has_road:
            return (
                image,
                [],
                empty_mask,
            )

        if random.random() > self.occlusion_probability:
            return (
                image,
                [],
                empty_mask,
            )

        # Convert from C x H x W raw imagery to H x W x C.
        image_rgb = image.transpose(
            1,
            2,
            0,
        )

        # Convert approximately to [0, 1] RGB.
        image_rgb = (
            image_rgb
            / self.scale_max[
                None,
                None,
                :,
            ]
        )

        image_rgb = np.clip(
            image_rgb,
            0.0,
            1.0,
        )

        # IMPORTANT:
        # Use the exact occlusion mask generated together with the
        # actual augmentation. This prevents a mismatch between the
        # image being evaluated and a separately regenerated mask.
        image_rgb, effects, occlusion_mask = apply_occlusions(
            image_rgb,
            min_effects=1,
            max_effects=3,
            return_mask=True,
        )

        # Convert back to raw imagery scale.
        image = (
            image_rgb
            * self.scale_max[
                None,
                None,
                :,
            ]
        )

        image = image.transpose(
            2,
            0,
            1,
        )

        return (
            image.astype(
                np.float32,
                copy=False,
            ),
            effects,
            occlusion_mask.astype(
                np.uint8,
                copy=False,
            ),
        )

    # ------------------------------------------------------------------
    # Read a raster
    # ------------------------------------------------------------------

    @staticmethod
    def _read_raster(
        path: Path,
    ) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(
                f"Raster file not found: {path}"
            )

        with rasterio.open(path) as src:
            array = src.read()

        return array

    # ------------------------------------------------------------------
    # Main sample retrieval
    # ------------------------------------------------------------------

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        row = self.data.iloc[index]

        image_path = self._resolve_path(
            str(row["image_path"])
        )

        mask_path = self._resolve_path(
            str(row["mask_path"])
        )

        image = self._read_raster(
            image_path
        )

        mask = self._read_raster(
            mask_path
        )

        if image.ndim != 3:
            raise ValueError(
                f"Expected image with shape "
                f"[C,H,W], got {image.shape}"
            )

        if image.shape[0] != 3:
            raise ValueError(
                f"Expected 3 image channels, "
                f"got {image.shape[0]}"
            )

        # Masks are expected to be single-band.
        if mask.ndim == 3:
            mask = mask[0]

        if mask.ndim != 2:
            raise ValueError(
                f"Expected mask with shape "
                f"[H,W], got {mask.shape}"
            )

        if image.shape[1:] != mask.shape:
            raise ValueError(
                "Image/mask shape mismatch:\n"
                f"Image: {image.shape}\n"
                f"Mask: {mask.shape}\n"
                f"Image path: {image_path}\n"
                f"Mask path: {mask_path}"
            )

        image = image.astype(
            np.float32,
            copy=False,
        )

        mask = mask.astype(
            np.float32,
            copy=False,
        )

        has_road = bool(
            row["has_road"]
        )

        # Apply synthetic occlusion BEFORE normalization.
        #
        # The function returns the exact spatial mask that was applied
        # to the image.
        (
            image,
            effects,
            occlusion_mask,
        ) = self._apply_occlusion(
            image,
            has_road,
        )

        # Standardize after augmentation.
        image = self._normalize(
            image
        )

        # Binary segmentation target.
        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        image_tensor = torch.from_numpy(
            image.copy()
        ).float()

        mask_tensor = torch.from_numpy(
            mask.copy()
        ).float().unsqueeze(0)

        occlusion_mask_tensor = (
            torch.from_numpy(
                occlusion_mask.copy()
            )
            .float()
            .unsqueeze(0)
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "scene_id": row["scene_id"],
            "tile": row["tile"],
            "has_road": has_road,
            "occlusion_effects": effects,
            "occlusion_mask": occlusion_mask_tensor,
        }