from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.road_dataset import RoadDataset
from src.models.occlusion_unet import OcclusionAwareUNet

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "spacenet_paris_tiles.csv"
)

STATS = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "spacenet_paris_band_stats.json"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "spacenet_occlusion_aware_positive_sampled.pt"
)


# ---------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------

THRESHOLD = 0.25

BASE_CHANNELS = 16
TRANSFORMER_DIM = 256

OCCLUSION_PROBABILITY = 1.0

# Fixed seed so the synthetic evaluation is reproducible.
RANDOM_SEED = 12345


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def load_model(
    device: torch.device,
) -> torch.nn.Module:

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    model = OcclusionAwareUNet(
        in_channels=3,
        out_channels=1,
        base_channels=BASE_CHANNELS,
        transformer_dim=TRANSFORMER_DIM,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ---------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------

def calculate_metrics(
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, float]:

    denominator_iou = (
        tp + fp + fn
    )

    denominator_precision = (
        tp + fp
    )

    denominator_recall = (
        tp + fn
    )

    denominator_dice = (
        2 * tp + fp + fn
    )

    iou = (
        tp / denominator_iou
        if denominator_iou > 0
        else 0.0
    )

    dice = (
        2 * tp / denominator_dice
        if denominator_dice > 0
        else 0.0
    )

    precision = (
        tp / denominator_precision
        if denominator_precision > 0
        else 0.0
    )

    recall = (
        tp / denominator_recall
        if denominator_recall > 0
        else 0.0
    )

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


# ---------------------------------------------------------------------
# Evaluate one dataset
# ---------------------------------------------------------------------

def evaluate_dataset(
    model: torch.nn.Module,
    dataset: RoadDataset,
    device: torch.device,
    threshold: float,
    calculate_occlusion_recall: bool = False,
) -> dict[str, float]:

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    occlusion_tp = 0
    occlusion_fn = 0
    occluded_road_pixels = 0

    with torch.no_grad():

        for index in range(len(dataset)):

            sample = dataset[index]

            image = (
                sample["image"]
                .unsqueeze(0)
                .to(device)
            )

            target = (
                sample["mask"]
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(bool)
            )

            logits = model(image)

            probability = (
                torch.sigmoid(logits)
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
            )

            prediction = (
                probability >= threshold
            )

            tp += int(
                np.logical_and(
                    prediction,
                    target,
                ).sum()
            )

            fp += int(
                np.logical_and(
                    prediction,
                    np.logical_not(target),
                ).sum()
            )

            fn += int(
                np.logical_and(
                    np.logical_not(prediction),
                    target,
                ).sum()
            )

            tn += int(
                np.logical_and(
                    np.logical_not(prediction),
                    np.logical_not(target),
                ).sum()
            )

            # ---------------------------------------------------------
            # Occlusion Recall
            # ---------------------------------------------------------

            if calculate_occlusion_recall:

                occlusion_mask = (
                    sample["occlusion_mask"]
                    .squeeze(0)
                    .numpy()
                    .astype(bool)
                )

                # Only reference road pixels inside the actual
                # synthetic spatial occlusion region count here.
                occluded_road = np.logical_and(
                    target,
                    occlusion_mask,
                )

                occluded_correct = np.logical_and(
                    prediction,
                    occluded_road,
                )

                occlusion_tp += int(
                    occluded_correct.sum()
                )

                occlusion_fn += int(
                    np.logical_and(
                        np.logical_not(prediction),
                        occluded_road,
                    ).sum()
                )

                occluded_road_pixels += int(
                    occluded_road.sum()
                )

    metrics = calculate_metrics(
        tp=tp,
        fp=fp,
        fn=fn,
    )

    if calculate_occlusion_recall:

        if (
            occlusion_tp + occlusion_fn
            > 0
        ):
            occlusion_recall = (
                occlusion_tp
                / (
                    occlusion_tp
                    + occlusion_fn
                )
            )
        else:
            occlusion_recall = 0.0

        metrics["occlusion_recall"] = (
            occlusion_recall
        )

        metrics["occlusion_tp"] = (
            float(occlusion_tp)
        )

        metrics["occlusion_fn"] = (
            float(occlusion_fn)
        )

        metrics["occluded_road_pixels"] = (
            float(occluded_road_pixels)
        )

    metrics["tp"] = float(tp)
    metrics["fp"] = float(fp)
    metrics["fn"] = float(fn)
    metrics["tn"] = float(tn)

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    set_seed(
        RANDOM_SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "SpaceNet Occlusion-Aware U-Net"
    )
    print(
        "=" * 72
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        f"Checkpoint: {CHECKPOINT.name}"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        f"Validation manifest: "
        f"{MANIFEST.name}"
    )

    # ---------------------------------------------------------------
    # Load model once
    # ---------------------------------------------------------------

    model = load_model(
        device
    )

    # ---------------------------------------------------------------
    # CLEAN VALIDATION
    # ---------------------------------------------------------------

    print()
    print(
        "CLEAN VALIDATION"
    )
    print(
        "-" * 72
    )

    clean_dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="val",
        stats_path=STATS,
        occlusion_enabled=False,
        occlusion_probability=0.0,
    )

    print(
        f"Validation tiles: "
        f"{len(clean_dataset)}"
    )

    clean_metrics = evaluate_dataset(
        model=model,
        dataset=clean_dataset,
        device=device,
        threshold=THRESHOLD,
        calculate_occlusion_recall=False,
    )

    print(
        f"TP:        "
        f"{int(clean_metrics['tp']):,}"
    )

    print(
        f"FP:        "
        f"{int(clean_metrics['fp']):,}"
    )

    print(
        f"FN:        "
        f"{int(clean_metrics['fn']):,}"
    )

    print(
        f"TN:        "
        f"{int(clean_metrics['tn']):,}"
    )

    print(
        f"IoU:       "
        f"{clean_metrics['iou']:.6f}"
    )

    print(
        f"Dice:      "
        f"{clean_metrics['dice']:.6f}"
    )

    print(
        f"Precision: "
        f"{clean_metrics['precision']:.6f}"
    )

    print(
        f"Recall:    "
        f"{clean_metrics['recall']:.6f}"
    )

    # ---------------------------------------------------------------
    # OCCLUDED VALIDATION
    # ---------------------------------------------------------------

    print()
    print(
        "OCCLUDED VALIDATION"
    )
    print(
        "-" * 72
    )

    # Reset seed so this evaluation is reproducible.
    set_seed(
        RANDOM_SEED
    )

    occluded_dataset = RoadDataset(
        manifest_path=MANIFEST,
        split="val",
        stats_path=STATS,
        occlusion_enabled=True,
        occlusion_probability=OCCLUSION_PROBABILITY,
    )

    print(
        f"Validation tiles: "
        f"{len(occluded_dataset)}"
    )

    print(
        "Synthetic occlusion: enabled"
    )

    print(
        f"Occlusion probability: "
        f"{OCCLUSION_PROBABILITY:.1f}"
    )

    occluded_metrics = evaluate_dataset(
        model=model,
        dataset=occluded_dataset,
        device=device,
        threshold=THRESHOLD,
        calculate_occlusion_recall=True,
    )

    print(
        f"TP:        "
        f"{int(occluded_metrics['tp']):,}"
    )

    print(
        f"FP:        "
        f"{int(occluded_metrics['fp']):,}"
    )

    print(
        f"FN:        "
        f"{int(occluded_metrics['fn']):,}"
    )

    print(
        f"TN:        "
        f"{int(occluded_metrics['tn']):,}"
    )

    print(
        f"IoU:       "
        f"{occluded_metrics['iou']:.6f}"
    )

    print(
        f"Dice:      "
        f"{occluded_metrics['dice']:.6f}"
    )

    print(
        f"Precision: "
        f"{occluded_metrics['precision']:.6f}"
    )

    print(
        f"Recall:    "
        f"{occluded_metrics['recall']:.6f}"
    )

    print()

    print(
        "OCCLUSION-REGION ANALYSIS"
    )
    print(
        "-" * 72
    )

    print(
        "Occluded road pixels: "
        f"{int(occluded_metrics['occluded_road_pixels']):,}"
    )

    print(
        "Correctly detected "
        "occluded road pixels: "
        f"{int(occluded_metrics['occlusion_tp']):,}"
    )

    print(
        "Missed occluded road pixels: "
        f"{int(occluded_metrics['occlusion_fn']):,}"
    )

    print(
        f"Occlusion Recall: "
        f"{occluded_metrics['occlusion_recall']:.6f}"
    )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "FINAL ADVANCED MODEL SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"Clean IoU:            "
        f"{clean_metrics['iou']:.6f}"
    )

    print(
        f"Clean Dice:           "
        f"{clean_metrics['dice']:.6f}"
    )

    print(
        f"Clean Precision:      "
        f"{clean_metrics['precision']:.6f}"
    )

    print(
        f"Clean Recall:         "
        f"{clean_metrics['recall']:.6f}"
    )

    print()

    print(
        f"Occluded IoU:         "
        f"{occluded_metrics['iou']:.6f}"
    )

    print(
        f"Occluded Dice:        "
        f"{occluded_metrics['dice']:.6f}"
    )

    print(
        f"Occluded Precision:   "
        f"{occluded_metrics['precision']:.6f}"
    )

    print(
        f"Occluded Recall:      "
        f"{occluded_metrics['recall']:.6f}"
    )

    print(
        f"Occlusion Recall:     "
        f"{occluded_metrics['occlusion_recall']:.6f}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()