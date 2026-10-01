from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from skimage.morphology import disk


PRED_MASK_PATH = Path(
    "data/processed/spacenet_paris_predictions/"
    "AOI_3_Paris_img84_road_mask.tif"
)

VALID_MASK_PATH = Path(
    "data/processed/spacenet_paris_predictions/"
    "AOI_3_Paris_img84_valid_data_mask.tif"
)

GROUND_TRUTH_GEOJSON = Path(
    "data/raw/spacenet/SpaceNet_Roads_Sample/"
    "AOI_3_Paris_Roads_Sample/geojson/spacenetroads/"
    "spacenetroads_AOI_3_Paris_img84.geojson"
)

OUTPUT_GT_MASK = Path(
    "data/processed/spacenet_paris_predictions/"
    "AOI_3_Paris_img84_ground_truth_5px_mask.tif"
)

OUTPUT_JSON = Path(
    "data/processed/spacenet_paris_graph/"
    "AOI_3_Paris_img84_mask_validation.json"
)


GROUND_TRUTH_DILATION_PIXELS = 5
TOLERANCES = [3, 5]


def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def calculate_dice(pred, gt):
    pred_count = pred.sum()
    gt_count = gt.sum()

    denominator = pred_count + gt_count

    if denominator == 0:
        return 1.0

    intersection = np.logical_and(pred, gt).sum()

    return float(
        2.0 * intersection / denominator
    )


def calculate_precision_recall(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(
        pred,
        np.logical_not(gt)
    ).sum()
    fn = np.logical_and(
        np.logical_not(pred),
        gt
    ).sum()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    return (
        int(tp),
        int(fp),
        int(fn),
        float(precision),
        float(recall),
    )


def main():
    print("=" * 72)
    print("SpaceNet Test Mask Evaluation")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load prediction raster
    # ------------------------------------------------------------------
    if not PRED_MASK_PATH.exists():
        raise FileNotFoundError(
            f"Prediction mask not found:\n{PRED_MASK_PATH}"
        )

    if not VALID_MASK_PATH.exists():
        raise FileNotFoundError(
            f"Valid-data mask not found:\n{VALID_MASK_PATH}"
        )

    if not GROUND_TRUTH_GEOJSON.exists():
        raise FileNotFoundError(
            f"Ground-truth GeoJSON not found:\n"
            f"{GROUND_TRUTH_GEOJSON}"
        )

    with rasterio.open(PRED_MASK_PATH) as src:
        pred = src.read(1)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        height = src.height
        width = src.width

    with rasterio.open(VALID_MASK_PATH) as src:
        valid_data = src.read(1).astype(bool)

    pred = pred.astype(bool)

    if pred.shape != valid_data.shape:
        raise RuntimeError(
            "Prediction mask and valid-data mask have "
            "different dimensions."
        )

    print(
        f"Prediction raster: {width} x {height}"
    )
    print(f"Prediction CRS: {crs}")
    print(
        f"Predicted road pixels: "
        f"{int(pred.sum()):,}"
    )
    print(
        f"Valid pixels: "
        f"{int(valid_data.sum()):,}"
    )

    # ------------------------------------------------------------------
    # 2. Load SpaceNet road-vector ground truth
    # ------------------------------------------------------------------
    print()
    print("Loading SpaceNet ground-truth GeoJSON...")

    gdf = gpd.read_file(GROUND_TRUTH_GEOJSON)

    print(
        f"Ground-truth features: "
        f"{len(gdf):,}"
    )

    print(
        "Geometry types:",
        gdf.geometry.geom_type.value_counts().to_dict()
    )

    if gdf.empty:
        raise RuntimeError(
            "Ground-truth GeoJSON contains no features."
        )

    # SpaceNet Road GeoJSONs normally use geographic coordinates.
    # Assign EPSG:4326 if the file does not explicitly declare a CRS.
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    print(
        f"Ground-truth CRS: {gdf.crs}"
    )

    # Reproject ground truth to prediction raster CRS.
    if crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # ------------------------------------------------------------------
    # 3. Rasterize centerlines
    # ------------------------------------------------------------------
    shapes = [
        (geom, 1)
        for geom in gdf.geometry
        if geom is not None
        and not geom.is_empty
    ]

    if not shapes:
        raise RuntimeError(
            "No valid geometries available after filtering."
        )

    centerline_mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    print()
    print(
        f"Rasterized centerline pixels: "
        f"{int(centerline_mask.sum()):,}"
    )

    # ------------------------------------------------------------------
    # 4. Create 5-pixel SpaceNet reference road mask
    # ------------------------------------------------------------------
    #
    # The project previously used a 5-pixel dilation to transform
    # SpaceNet road centerlines into training road masks. We reproduce
    # that same convention here for the reference evaluation.
    #
    gt_mask = binary_dilation(
        centerline_mask,
        structure=disk(
            GROUND_TRUTH_DILATION_PIXELS
        ),
    )

    # Evaluate only inside valid source imagery.
    gt_mask &= valid_data
    pred &= valid_data

    print(
        f"Ground-truth 5px road pixels: "
        f"{int(gt_mask.sum()):,}"
    )

    # ------------------------------------------------------------------
    # 5. Save reference mask
    # ------------------------------------------------------------------
    gt_profile = profile.copy()

    gt_profile.update(
        dtype="uint8",
        count=1,
        compress="deflate",
    )

    OUTPUT_GT_MASK.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        OUTPUT_GT_MASK,
        "w",
        **gt_profile,
    ) as dst:
        dst.write(
            gt_mask.astype(np.uint8),
            1,
        )

    print(
        f"Saved ground-truth mask: "
        f"{OUTPUT_GT_MASK}"
    )

    # ------------------------------------------------------------------
    # 6. Standard metrics
    # ------------------------------------------------------------------
    iou = calculate_iou(
        pred,
        gt_mask,
    )

    dice = calculate_dice(
        pred,
        gt_mask,
    )

    (
        tp,
        fp,
        fn,
        precision,
        recall,
    ) = calculate_precision_recall(
        pred,
        gt_mask,
    )

    print()
    print("-" * 72)
    print("STANDARD METRICS")
    print("-" * 72)

    print(
        f"TP:         {tp:,}"
    )
    print(
        f"FP:         {fp:,}"
    )
    print(
        f"FN:         {fn:,}"
    )
    print(
        f"IoU:        {iou:.6f}"
    )
    print(
        f"Dice:       {dice:.6f}"
    )
    print(
        f"Precision:  {precision:.6f}"
    )
    print(
        f"Recall:     {recall:.6f}"
    )

    # ------------------------------------------------------------------
    # 7. Tolerance / relaxed IoU
    # ------------------------------------------------------------------
    #
    # For each tolerance radius:
    #
    #   relaxed IoU =
    #       IoU(dilate(pred, r), dilate(gt, r))
    #
    # This is a symmetric buffer-tolerant IoU.
    #
    # Length-complete recall =
    #       fraction of GT road pixels covered by
    #       the prediction after a tolerance buffer.
    #
    # Relaxed precision =
    #       fraction of predicted road pixels lying
    #       within the tolerance buffer of GT.
    # ------------------------------------------------------------------

    tolerance_results = {}

    print()
    print("-" * 72)
    print("RELAXED / TOLERANCE METRICS")
    print("-" * 72)

    for tolerance in TOLERANCES:

        struct = disk(tolerance)

        pred_buffer = binary_dilation(
            pred,
            structure=struct,
        )

        gt_buffer = binary_dilation(
            gt_mask,
            structure=struct,
        )

        # Keep evaluation within actual valid imagery.
        pred_buffer &= valid_data
        gt_buffer &= valid_data

        relaxed_iou = calculate_iou(
            pred_buffer,
            gt_buffer,
        )

        # GT road pixels that fall inside the buffered prediction.
        gt_covered = np.logical_and(
            gt_mask,
            pred_buffer,
        ).sum()

        gt_total = gt_mask.sum()

        length_completeness = (
            gt_covered / gt_total
            if gt_total > 0
            else 0.0
        )

        # Predicted road pixels that fall inside buffered GT.
        pred_covered = np.logical_and(
            pred,
            gt_buffer,
        ).sum()

        pred_total = pred.sum()

        relaxed_precision = (
            pred_covered / pred_total
            if pred_total > 0
            else 0.0
        )

        tolerance_results[str(tolerance)] = {
            "tolerance_pixels": tolerance,
            "relaxed_iou": float(
                relaxed_iou
            ),
            "length_completeness_recall": float(
                length_completeness
            ),
            "relaxed_precision": float(
                relaxed_precision
            ),
            "ground_truth_pixels_covered": int(
                gt_covered
            ),
            "ground_truth_total_pixels": int(
                gt_total
            ),
            "predicted_pixels_covered": int(
                pred_covered
            ),
            "predicted_total_pixels": int(
                pred_total
            ),
        }

        print(
            f"{tolerance}px tolerance:"
        )

        print(
            f"  Relaxed IoU:          "
            f"{relaxed_iou:.6f}"
        )

        print(
            f"  Length completeness:   "
            f"{length_completeness:.6f}"
        )

        print(
            f"  Relaxed precision:     "
            f"{relaxed_precision:.6f}"
        )

    # ------------------------------------------------------------------
    # 8. Save complete result
    # ------------------------------------------------------------------
    result = {
        "dataset": "SpaceNet 3 Paris",
        "scene": "AOI_3_Paris_img84",
        "prediction": {
            "path": str(PRED_MASK_PATH),
            "road_pixels": int(pred.sum()),
        },
        "ground_truth": {
            "source": str(GROUND_TRUTH_GEOJSON),
            "geometry_features": int(len(gdf)),
            "centerline_pixels": int(
                centerline_mask.sum()
            ),
            "dilation_pixels": (
                GROUND_TRUTH_DILATION_PIXELS
            ),
            "road_mask_pixels": int(
                gt_mask.sum()
            ),
        },
        "evaluation_domain": {
            "valid_pixels": int(
                valid_data.sum()
            ),
        },
        "standard_metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
        },
        "tolerance_metrics": tolerance_results,
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("-" * 72)
    print(
        f"Saved evaluation: {OUTPUT_JSON}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()