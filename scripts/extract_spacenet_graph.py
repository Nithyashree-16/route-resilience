from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SKELETON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spacenet_paris_graph"
    / "AOI_3_Paris_img84_skeleton.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spacenet_paris_graph"
)

NODES_PATH = (
    OUTPUT_DIR
    / "AOI_3_Paris_img84_nodes.geojson"
)

EDGES_PATH = (
    OUTPUT_DIR
    / "AOI_3_Paris_img84_edges.geojson"
)

NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEOD = Geod(ellps="WGS84")


def neighbours(
    pixel: tuple[int, int],
    skeleton: np.ndarray,
) -> list[tuple[int, int]]:
    row, col = pixel

    height, width = skeleton.shape

    result = []

    for dr, dc in NEIGHBOR_OFFSETS:
        nr = row + dr
        nc = col + dc

        if (
            0 <= nr < height
            and 0 <= nc < width
            and skeleton[nr, nc]
        ):
            result.append((nr, nc))

    return result


def pixel_to_xy(
    transform,
    pixel: tuple[int, int],
) -> tuple[float, float]:
    row, col = pixel

    x, y = rasterio.transform.xy(
        transform,
        row,
        col,
        offset="center",
    )

    return float(x), float(y)


def geodesic_length(
    coordinates: list[tuple[float, float]],
) -> float:
    if len(coordinates) < 2:
        return 0.0

    total = 0.0

    for i in range(len(coordinates) - 1):
        lon1, lat1 = coordinates[i]
        lon2, lat2 = coordinates[i + 1]

        _, _, distance = GEOD.inv(
            lon1,
            lat1,
            lon2,
            lat2,
        )

        total += abs(distance)

    return float(total)


def cluster_node_pixels(
    skeleton: np.ndarray,
) -> tuple[
    np.ndarray,
    dict[int, set[tuple[int, int]]],
]:
    """
    Pixels whose skeleton degree is not 2 are candidate node pixels.

    Adjacent candidate pixels are grouped into one logical node.
    """
    candidate = np.zeros(
        skeleton.shape,
        dtype=bool,
    )

    rows, cols = np.nonzero(skeleton)

    for row, col in zip(rows, cols):
        pixel = (int(row), int(col))

        degree = len(
            neighbours(
                pixel,
                skeleton,
            )
        )

        if degree != 0 and degree != 2:
          candidate[row, col] = True

    labels, count = ndimage.label(
        candidate,
        structure=np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
    )

    node_pixels = {}

    for label_id in range(1, count + 1):
        coords = np.argwhere(
            labels == label_id
        )

        node_pixels[label_id] = {
            (int(row), int(col))
            for row, col in coords
        }

    return labels, node_pixels


def build_pixel_to_node_map(
    node_pixels: dict[
        int,
        set[tuple[int, int]],
    ],
) -> dict[
    tuple[int, int],
    int,
]:
    mapping = {}

    for node_id, pixels in node_pixels.items():
        for pixel in pixels:
            mapping[pixel] = node_id

    return mapping


def node_boundary_starts(
    node_id: int,
    pixels: set[tuple[int, int]],
    skeleton: np.ndarray,
    pixel_to_node: dict[
        tuple[int, int],
        int,
    ],
) -> list[
    tuple[
        tuple[int, int],
        tuple[int, int],
    ]
]:
    """
    Return pairs:

        (node_pixel, first_road_pixel)

    The node_pixel is retained because the first road pixel is naturally
    connected to both the node and the next road pixel.
    """
    starts = set()

    for node_pixel in pixels:

        for neighbor in neighbours(
            node_pixel,
            skeleton,
        ):

            if neighbor in pixel_to_node:
                continue

            starts.add(
                (
                    node_pixel,
                    neighbor,
                )
            )

    return sorted(starts)


def trace_edge(
    start_node: int,
    start_node_pixel: tuple[int, int],
    start_pixel: tuple[int, int],
    skeleton: np.ndarray,
    pixel_to_node: dict[
        tuple[int, int],
        int,
    ],
) -> tuple[
    int | None,
    list[tuple[int, int]],
]:
    """
    Trace from a logical node through degree-2 skeleton pixels until
    another logical node is reached.

    The previous pixel starts as the node pixel. This is the important
    fix that prevents the first road pixel from being treated as an
    ambiguous branch.
    """
    previous = start_node_pixel
    current = start_pixel

    path = []

    max_steps = int(
        skeleton.size
    )

    steps = 0

    while steps < max_steps:

        steps += 1

        # We reached another logical node.
        if current in pixel_to_node:
            end_node = pixel_to_node[current]

            if end_node != start_node:
                return end_node, path

            return None, path

        path.append(current)

        current_neighbors = neighbours(
            current,
            skeleton,
        )

        # Remove the pixel we came from.
        next_candidates = [
            neighbor
            for neighbor in current_neighbors
            if neighbor != previous
        ]

        # Look directly for a logical node among the neighbours.
        node_neighbors = [
            neighbor
            for neighbor in next_candidates
            if neighbor in pixel_to_node
        ]

        if node_neighbors:
            end_node = pixel_to_node[
                node_neighbors[0]
            ]

            if end_node != start_node:
                return end_node, path

            return None, path

        # Normal road corridor: exactly one continuation.
        if len(next_candidates) == 1:
            previous = current
            current = next_candidates[0]
            continue

        # Dead end.
        if len(next_candidates) == 0:
            return None, path

        # Unexpected branch not represented as a logical node.
        # Stop rather than inventing topology.
        return None, path

    return None, path


def classify_node(
    pixels: set[tuple[int, int]],
    skeleton: np.ndarray,
) -> tuple[str, int]:
    """
    Degree is the number of distinct skeleton neighbours leaving the
    logical node cluster.
    """
    boundary_neighbors = set()

    for pixel in pixels:
        for neighbor in neighbours(
            pixel,
            skeleton,
        ):
            if neighbor not in pixels:
                boundary_neighbors.add(
                    neighbor
                )

    degree = len(boundary_neighbors)

    if degree <= 1:
        node_type = "endpoint"
    elif degree >= 3:
        node_type = "intersection"
    else:
        node_type = "junction"

    return node_type, degree


def main() -> None:

    if not SKELETON_PATH.exists():
        raise FileNotFoundError(
            f"Skeleton not found:\n{SKELETON_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "SpaceNet Skeleton -> Graph Extraction"
    )
    print("=" * 72)

    with rasterio.open(
        SKELETON_PATH
    ) as src:

        skeleton = (
            src.read(1) > 0
        )

        transform = src.transform
        crs = src.crs

    print(
        "Skeleton size:",
        f"{skeleton.shape[1]} x {skeleton.shape[0]}",
    )

    print(
        "Skeleton pixels:",
        f"{int(skeleton.sum()):,}",
    )

    print(
        "CRS:",
        crs,
    )

    # ---------------------------------------------------------------
    # Node detection
    # ---------------------------------------------------------------

    _, node_pixels = (
        cluster_node_pixels(
            skeleton
        )
    )

    pixel_to_node = (
        build_pixel_to_node_map(
            node_pixels
        )
    )

    print(
        "Logical nodes:",
        len(node_pixels),
    )

    node_features = []

    node_metadata = {}

    for node_id, pixels in node_pixels.items():

        rows = np.array(
            [pixel[0] for pixel in pixels],
            dtype=np.float64,
        )

        cols = np.array(
            [pixel[1] for pixel in pixels],
            dtype=np.float64,
        )

        center_row = float(
            rows.mean()
        )

        center_col = float(
            cols.mean()
        )

        x, y = rasterio.transform.xy(
            transform,
            center_row,
            center_col,
            offset="center",
        )

        node_type, degree = classify_node(
            pixels,
            skeleton,
        )

        node_metadata[node_id] = {
            "node_id": int(node_id),
            "node_type": node_type,
            "degree": int(degree),
            "x": float(x),
            "y": float(y),
        }

        node_features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(x),
                        float(y),
                    ],
                },
                "properties": {
                    "node_id": int(node_id),
                    "node_type": node_type,
                    "degree": int(degree),
                },
            }
        )

    nodes_geojson = {
        "type": "FeatureCollection",
        "features": node_features,
    }

    with NODES_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            nodes_geojson,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Edge extraction
    # ---------------------------------------------------------------

    edge_features = []

    visited_half_edges = set()

    edge_id = 0

    for start_node, pixels in node_pixels.items():

        starts = node_boundary_starts(
            start_node,
            pixels,
            skeleton,
            pixel_to_node,
        )

        for (
            start_node_pixel,
            start_pixel,
        ) in starts:

            half_edge_key = (
                start_node,
                start_node_pixel,
                start_pixel,
            )

            if half_edge_key in visited_half_edges:
                continue

            visited_half_edges.add(
                half_edge_key
            )

            end_node, pixel_path = (
                trace_edge(
                    start_node=start_node,
                    start_node_pixel=start_node_pixel,
                    start_pixel=start_pixel,
                    skeleton=skeleton,
                    pixel_to_node=pixel_to_node,
                )
            )

            if end_node is None:
                continue

            if end_node == start_node:
                continue

            # Mark the reverse direction when we can identify its
            # starting boundary pixel. This keeps the final graph
            # undirected while avoiding duplicate edges.
            reverse_starts = (
                node_boundary_starts(
                    end_node,
                    node_pixels[end_node],
                    skeleton,
                    pixel_to_node,
                )
            )

            # -------------------------------------------------------
            # Coordinates
            # -------------------------------------------------------

            coordinates = []

            start_pixels_array = np.array(
                list(
                    node_pixels[start_node]
                ),
                dtype=np.float64,
            )

            start_center = (
                start_pixels_array.mean(
                    axis=0
                )
            )

            start_xy = (
                rasterio.transform.xy(
                    transform,
                    float(start_center[0]),
                    float(start_center[1]),
                    offset="center",
                )
            )

            coordinates.append(
                (
                    float(start_xy[0]),
                    float(start_xy[1]),
                )
            )

            for pixel in pixel_path:
                coordinates.append(
                    pixel_to_xy(
                        transform,
                        pixel,
                    )
                )

            end_pixels_array = np.array(
                list(
                    node_pixels[end_node]
                ),
                dtype=np.float64,
            )

            end_center = (
                end_pixels_array.mean(
                    axis=0
                )
            )

            end_xy = (
                rasterio.transform.xy(
                    transform,
                    float(end_center[0]),
                    float(end_center[1]),
                    offset="center",
                )
            )

            coordinates.append(
                (
                    float(end_xy[0]),
                    float(end_xy[1]),
                )
            )

            # Remove consecutive duplicate coordinates.
            cleaned_coordinates = []

            for coordinate in coordinates:
                if (
                    not cleaned_coordinates
                    or coordinate
                    != cleaned_coordinates[-1]
                ):
                    cleaned_coordinates.append(
                        coordinate
                    )

            length_m = geodesic_length(
                cleaned_coordinates
            )

            if length_m <= 0:
                continue

            # Canonical path representation prevents adding the exact
            # same road twice in opposite directions.
            endpoint_pair = tuple(
                sorted(
                    (
                        int(start_node),
                        int(end_node),
                    )
                )
            )

            # Do not collapse distinct parallel roads between the same
            # logical nodes unless their exact pixel path is identical.
            pixel_signature = tuple(
                pixel_path
            )

            reverse_signature = tuple(
                reversed(
                    pixel_path
                )
            )

            duplicate_found = False

            for existing in edge_features:
                props = existing["properties"]

                if tuple(
                    sorted(
                        (
                            props["start_node"],
                            props["end_node"],
                        )
                    )
                ) != endpoint_pair:
                    continue

                existing_signature = tuple(
                    existing.get(
                        "_pixel_path",
                        [],
                    )
                )

                if (
                    pixel_signature
                    == existing_signature
                    or reverse_signature
                    == existing_signature
                ):
                    duplicate_found = True
                    break

            if duplicate_found:
                continue

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [
                            float(x),
                            float(y),
                        ]
                        for x, y
                        in cleaned_coordinates
                    ],
                },
                "properties": {
                    "edge_id": int(edge_id),
                    "start_node": int(
                        start_node
                    ),
                    "end_node": int(
                        end_node
                    ),
                    "length_m": float(
                        length_m
                    ),
                    "pixel_count": int(
                        len(pixel_path)
                    ),
                },
                # Internal field used only during duplicate detection.
                "_pixel_path": pixel_signature,
            }

            edge_features.append(
                feature
            )

            edge_id += 1

    # Remove internal helper fields before writing GeoJSON.
    clean_edge_features = []

    for feature in edge_features:

        clean_feature = {
            "type": "Feature",
            "geometry": feature["geometry"],
            "properties": feature["properties"],
        }

        clean_edge_features.append(
            clean_feature
        )

    edges_geojson = {
        "type": "FeatureCollection",
        "features": clean_edge_features,
    }

    with EDGES_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            edges_geojson,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    endpoint_count = sum(
        1
        for info in node_metadata.values()
        if info["node_type"] == "endpoint"
    )

    intersection_count = sum(
        1
        for info in node_metadata.values()
        if info["node_type"] == "intersection"
    )

    junction_count = sum(
        1
        for info in node_metadata.values()
        if info["node_type"] == "junction"
    )

    total_length_m = sum(
        feature["properties"]["length_m"]
        for feature in clean_edge_features
    )

    print()
    print(
        "Endpoints:",
        endpoint_count,
    )

    print(
        "Intersections:",
        intersection_count,
    )

    print(
        "Junctions:",
        junction_count,
    )

    print(
        "Edges:",
        len(clean_edge_features),
    )

    print(
        "Total edge length (m):",
        f"{total_length_m:,.2f}",
    )

    print()
    print(
        "Nodes:",
        NODES_PATH,
    )

    print(
        "Edges:",
        EDGES_PATH,
    )

    print("=" * 72)
    print(
        "Graph extraction complete."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()