from __future__ import annotations

import random
from typing import Tuple

import cv2
import numpy as np


def _clip_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)


def _random_rect(
    height: int,
    width: int,
    min_size: int,
    max_size: int,
) -> Tuple[int, int, int, int]:
    w = random.randint(
        min_size,
        min(max_size, width),
    )
    h = random.randint(
        min_size,
        min(max_size, height),
    )

    x1 = random.randint(
        0,
        max(0, width - w),
    )
    y1 = random.randint(
        0,
        max(0, height - h),
    )

    return x1, y1, x1 + w, y1 + h


# ---------------------------------------------------------------------
# Internal effect implementations
# ---------------------------------------------------------------------
# Each spatial effect returns:
#   output_image
#   binary_occlusion_mask
#
# The mask represents the actual spatial region affected by the effect.
# Illumination is an appearance-only effect, so its mask is all zeros.
# ---------------------------------------------------------------------


def _apply_tree_canopy(
    image: np.ndarray,
    strength: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    output = image.copy()
    h, w, _ = output.shape

    geometry_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    count = random.randint(3, 8)

    for _ in range(count):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)

        rx = random.randint(
            max(8, w // 20),
            max(12, w // 6),
        )

        ry = random.randint(
            max(8, h // 20),
            max(12, h // 6),
        )

        cv2.ellipse(
            geometry_mask,
            (cx, cy),
            (rx, ry),
            random.uniform(0, 180),
            0,
            360,
            255,
            -1,
        )

    # Preserve the original soft canopy appearance.
    soft_mask = cv2.GaussianBlur(
        geometry_mask,
        (21, 21),
        0,
    )

    alpha = (
        soft_mask.astype(np.float32) / 255.0
    ) * strength

    canopy_color = np.array(
        [0.18, 0.24, 0.12],
        dtype=np.float32,
    )

    output = (
        output * (1.0 - alpha[..., None])
        + canopy_color * alpha[..., None]
    )

    return (
        _clip_image(output),
        (geometry_mask > 0).astype(np.uint8),
    )


def _apply_vehicles(
    image: np.ndarray,
    strength: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    output = image.copy()
    h, w, _ = output.shape

    geometry_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    count = random.randint(2, 10)

    for _ in range(count):
        x1, y1, x2, y2 = _random_rect(
            h,
            w,
            4,
            18,
        )

        color = np.random.uniform(
            0.05,
            0.35,
            size=3,
        ).astype(np.float32)

        output[y1:y2, x1:x2] = (
            output[y1:y2, x1:x2] * (1.0 - strength)
            + color * strength
        )

        geometry_mask[y1:y2, x1:x2] = 255

    return (
        _clip_image(output),
        (geometry_mask > 0).astype(np.uint8),
    )


def _apply_shadows(
    image: np.ndarray,
    strength: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    output = image.copy()
    h, w, _ = output.shape

    geometry_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    points = np.array(
        [
            [
                random.randint(0, w - 1),
                random.randint(0, h - 1),
            ]
            for _ in range(4)
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(
        geometry_mask,
        [points],
        255,
    )

    soft_mask = cv2.GaussianBlur(
        geometry_mask,
        (31, 31),
        0,
    )

    alpha = (
        soft_mask.astype(np.float32) / 255.0
    ) * strength

    output *= (
        1.0 - alpha[..., None]
    )

    return (
        _clip_image(output),
        (geometry_mask > 0).astype(np.uint8),
    )


def _apply_clouds(
    image: np.ndarray,
    strength: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    output = image.copy()
    h, w, _ = output.shape

    geometry_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    count = random.randint(1, 3)

    for _ in range(count):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)

        rx = random.randint(
            w // 8,
            w // 3,
        )

        ry = random.randint(
            h // 10,
            h // 4,
        )

        cv2.ellipse(
            geometry_mask,
            (cx, cy),
            (rx, ry),
            random.uniform(0, 180),
            0,
            360,
            255,
            -1,
        )

    soft_mask = cv2.GaussianBlur(
        geometry_mask,
        (51, 51),
        0,
    )

    alpha = (
        soft_mask.astype(np.float32) / 255.0
    ) * strength

    cloud_color = np.ones(
        (h, w, 3),
        dtype=np.float32,
    )

    output = (
        output * (1.0 - alpha[..., None])
        + cloud_color * alpha[..., None]
    )

    return (
        _clip_image(output),
        (geometry_mask > 0).astype(np.uint8),
    )


def _apply_urban_clutter(
    image: np.ndarray,
    strength: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    output = image.copy()
    h, w, _ = output.shape

    geometry_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    count = random.randint(5, 15)

    for _ in range(count):
        x1, y1, x2, y2 = _random_rect(
            h,
            w,
            3,
            15,
        )

        color = np.random.uniform(
            0.1,
            0.9,
            size=3,
        ).astype(np.float32)

        output[y1:y2, x1:x2] = (
            output[y1:y2, x1:x2] * (1.0 - strength)
            + color * strength
        )

        geometry_mask[y1:y2, x1:x2] = 255

    return (
        _clip_image(output),
        (geometry_mask > 0).astype(np.uint8),
    )


def _apply_illumination(
    image: np.ndarray,
    strength: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Appearance-only augmentation.

    Illumination changes the whole image appearance but does not create
    a spatially occluded region, so its evaluation mask is all zeros.
    """
    output = image.copy()

    brightness = random.uniform(
        1.0 - strength,
        1.0 + strength,
    )

    contrast = random.uniform(
        1.0 - strength,
        1.0 + strength,
    )

    output = (
        (output - 0.5) * contrast
        + 0.5
    )

    output *= brightness

    mask = np.zeros(
        output.shape[:2],
        dtype=np.uint8,
    )

    return (
        _clip_image(output),
        mask,
    )


# ---------------------------------------------------------------------
# Public individual effect functions
# ---------------------------------------------------------------------


def apply_tree_canopy(
    image: np.ndarray,
    strength: float = 0.75,
) -> np.ndarray:
    output, _ = _apply_tree_canopy(
        image,
        strength,
    )
    return output


def apply_vehicles(
    image: np.ndarray,
    strength: float = 0.85,
) -> np.ndarray:
    output, _ = _apply_vehicles(
        image,
        strength,
    )
    return output


def apply_shadows(
    image: np.ndarray,
    strength: float = 0.55,
) -> np.ndarray:
    output, _ = _apply_shadows(
        image,
        strength,
    )
    return output


def apply_clouds(
    image: np.ndarray,
    strength: float = 0.55,
) -> np.ndarray:
    output, _ = _apply_clouds(
        image,
        strength,
    )
    return output


def apply_urban_clutter(
    image: np.ndarray,
    strength: float = 0.65,
) -> np.ndarray:
    output, _ = _apply_urban_clutter(
        image,
        strength,
    )
    return output


def apply_illumination(
    image: np.ndarray,
    strength: float = 0.35,
) -> np.ndarray:
    output, _ = _apply_illumination(
        image,
        strength,
    )
    return output


# ---------------------------------------------------------------------
# Combined augmentation
# ---------------------------------------------------------------------


def apply_occlusions(
    image: np.ndarray,
    min_effects: int = 1,
    max_effects: int = 3,
    seed: int | None = None,
    return_mask: bool = False,
):
    """
    Apply a random combination of occlusion/appearance effects.

    Input:
        image:
            H x W x 3 float32 image in [0, 1]

    Output when return_mask=False:
        augmented image
        names of applied effects

    Output when return_mask=True:
        augmented image
        names of applied effects
        binary spatial occlusion mask

    The returned mask is generated from the exact same spatial geometry
    that was used to modify the image.

    Illumination is included as an appearance effect but contributes
    zero pixels to the spatial occlusion mask.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    effects = [
        ("tree_canopy", _apply_tree_canopy),
        ("vehicles", _apply_vehicles),
        ("shadows", _apply_shadows),
        ("clouds", _apply_clouds),
        ("urban_clutter", _apply_urban_clutter),
        ("illumination", _apply_illumination),
    ]

    number = random.randint(
        min_effects,
        max_effects,
    )

    selected = random.sample(
        effects,
        k=number,
    )

    output = image.copy()

    occlusion_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    applied = []

    for name, function in selected:
        output, effect_mask = function(
            output
        )

        # Union all spatially occluded regions.
        occlusion_mask = np.maximum(
            occlusion_mask,
            effect_mask,
        )

        applied.append(name)

    output = _clip_image(output)

    if return_mask:
        return (
            output,
            applied,
            occlusion_mask,
        )

    return output, applied


# ---------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------


def create_occlusion_mask(
    image_shape: tuple[int, int, int],
    min_effects: int = 1,
    max_effects: int = 3,
    seed: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Create the exact spatial occlusion mask associated with the same
    occlusion generation process used by apply_occlusions().

    The mask is H x W:
        0 = not spatially occluded
        1 = spatially occluded

    Illumination does not contribute to the mask because it is an
    appearance/illumination change rather than a spatially hidden region.

    Passing the same seed to apply_occlusions() and this function will
    reproduce the same selected effects and spatial regions.
    """
    h, w, channels = image_shape

    if channels != 3:
        raise ValueError(
            "image_shape must be (H, W, 3)."
        )

    dummy_image = np.zeros(
        (h, w, channels),
        dtype=np.float32,
    )

    _, selected, mask = apply_occlusions(
        dummy_image,
        min_effects=min_effects,
        max_effects=max_effects,
        seed=seed,
        return_mask=True,
    )

    return mask.astype(
        np.uint8
    ), selected