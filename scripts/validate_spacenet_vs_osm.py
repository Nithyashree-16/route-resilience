from pathlib import Path
import json
import random
import statistics

import networkx as nx
import numpy as np
import osmnx as ox
import rasterio


GRAPH_PATH = Path(
    "data/processed/spacenet_paris_graph/"
    "AOI_3_Paris_img84_weighted.graphml"
)

RASTER_PATH = Path(
    "data/processed/spacenet_paris_predictions/"
    "AOI_3_Paris_img84_road_mask.tif"
)
OUTPUT_PATH = Path(
    "data/processed/spacenet_paris_graph/"
    "AOI_3_Paris_img84_osm_validation.json"
)

OSM_GRAPH_PATH = Path(
    "data/processed/spacenet_paris_graph/"
    "AOI_3_Paris_img84_osm_drive.graphml"
)

NUM_RANDOM_PAIRS = 500
RANDOM_SEED = 42


def normalize_predicted_graph(G):
    """
    Convert GraphML node IDs back to integers where possible.
    """
    mapping = {}

    for node in G.nodes:
        try:
            mapping[node] = int(node)
        except (TypeError, ValueError):
            mapping[node] = node

    if any(k != v for k, v in mapping.items()):
        G = nx.relabel_nodes(G, mapping, copy=True)

    return G


def get_node_lon_lat(G, node):
    """
    Read longitude/latitude from predicted graph node attributes.
    """
    attrs = G.nodes[node]

    lon = None
    lat = None

    for key in ["x", "lon", "longitude"]:
        if key in attrs:
            lon = float(attrs[key])
            break

    for key in ["y", "lat", "latitude"]:
        if key in attrs:
            lat = float(attrs[key])
            break

    if lon is None or lat is None:
        raise RuntimeError(
            f"Could not find lon/lat attributes for node {node}. "
            f"Available attributes: {list(attrs.keys())}"
        )

    return lon, lat


def main():
    print("=" * 72)
    print("SpaceNet vs OSM Topological Validation")
    print("=" * 72)

    random.seed(RANDOM_SEED)

    # ---------------------------------------------------------------
    # 1. Load predicted SpaceNet graph
    # ---------------------------------------------------------------
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Predicted graph not found:\n{GRAPH_PATH}"
        )

    G_pred = nx.read_graphml(GRAPH_PATH)
    G_pred = normalize_predicted_graph(G_pred)

    print(
        f"Predicted graph: "
        f"{G_pred.number_of_nodes()} nodes, "
        f"{G_pred.number_of_edges()} edges"
    )

    # ---------------------------------------------------------------
    # 2. Read SpaceNet raster extent
    # ---------------------------------------------------------------
    if not RASTER_PATH.exists():
        raise FileNotFoundError(
            f"SpaceNet raster not found:\n{RASTER_PATH}"
        )

    with rasterio.open(RASTER_PATH) as src:
        bounds = src.bounds
        crs = src.crs

    print(f"Raster CRS: {crs}")

    print(
        "AOI bounds: "
        f"left={bounds.left:.8f}, "
        f"bottom={bounds.bottom:.8f}, "
        f"right={bounds.right:.8f}, "
        f"top={bounds.top:.8f}"
    )

    if crs is None or str(crs) != "EPSG:4326":
        raise RuntimeError(
            "Expected SpaceNet raster CRS to be EPSG:4326."
        )

    # OSMnx bbox order:
    # (left/west, bottom/south, right/east, top/north)
    pad = 0.005

    bbox = (
    bounds.left - pad,
    bounds.bottom - pad,
    bounds.right + pad,
    bounds.top + pad,
)

    # ---------------------------------------------------------------
    # 3. Download OSM driving network
    # ---------------------------------------------------------------
    print()
    print("Downloading OSM drive network...")
    print("This may take some time depending on Overpass response.")

    ox.settings.use_cache = True
    ox.settings.requests_timeout = 180

    G_osm = ox.graph.graph_from_bbox(
        bbox=bbox,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    print(
        f"OSM graph before undirected conversion: "
        f"{G_osm.number_of_nodes()} nodes, "
        f"{G_osm.number_of_edges()} edges"
    )

    # Convert to undirected structure so that the comparison focuses
    # on topology/path structure rather than one-way restrictions.
    G_osm = ox.convert.to_undirected(G_osm)

    print(
        f"OSM graph after undirected conversion: "
        f"{G_osm.number_of_nodes()} nodes, "
        f"{G_osm.number_of_edges()} edges"
    )

    # ---------------------------------------------------------------
    # 4. Check predicted node coordinates
    # ---------------------------------------------------------------
    pred_nodes = list(G_pred.nodes)

    if len(pred_nodes) < 10:
        raise RuntimeError(
            "Too few predicted graph nodes for validation."
        )

    node_coords = {}

    for node in pred_nodes:
        node_coords[node] = get_node_lon_lat(
            G_pred,
            node
        )

    # ---------------------------------------------------------------
    # 5. Snap predicted nodes to nearest OSM nodes
    # ---------------------------------------------------------------
    pred_lons = np.array(
        [node_coords[n][0] for n in pred_nodes],
        dtype=float,
    )

    pred_lats = np.array(
        [node_coords[n][1] for n in pred_nodes],
        dtype=float,
    )

    print()
    print("Snapping predicted graph nodes to nearest OSM nodes...")

    osm_nearest_nodes = ox.distance.nearest_nodes(
        G_osm,
        X=pred_lons,
        Y=pred_lats,
    )

    pred_to_osm = {
        pred_nodes[i]: int(osm_nearest_nodes[i])
        for i in range(len(pred_nodes))
    }

    # ---------------------------------------------------------------
    # 6. Select random origin-destination pairs
    # ---------------------------------------------------------------
    sampled_pairs = []

    max_attempts = NUM_RANDOM_PAIRS * 30
    attempts = 0

    while (
        len(sampled_pairs) < NUM_RANDOM_PAIRS
        and attempts < max_attempts
    ):
        attempts += 1

        source, target = random.sample(pred_nodes, 2)

        osm_source = pred_to_osm[source]
        osm_target = pred_to_osm[target]

        if osm_source == osm_target:
            continue

        pair = (
            source,
            target,
            osm_source,
            osm_target,
        )

        if pair not in sampled_pairs:
            sampled_pairs.append(pair)

    print(
        f"Sampled common OD pairs: "
        f"{len(sampled_pairs)}"
    )

    # ---------------------------------------------------------------
    # 7. Compare shortest-path lengths
    # ---------------------------------------------------------------
    print()
    print("Comparing shortest-path lengths...")

    results = []

    for (
        pred_source,
        pred_target,
        osm_source,
        osm_target,
    ) in sampled_pairs:

        # Predicted graph path
        try:
            pred_length = nx.shortest_path_length(
                G_pred,
                pred_source,
                pred_target,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            continue

        # OSM graph path
        try:
            osm_length = nx.shortest_path_length(
                G_osm,
                osm_source,
                osm_target,
                weight="length",
            )
        except nx.NetworkXNoPath:
            continue

        if osm_length <= 0:
            continue

        absolute_error = abs(
            float(pred_length) - float(osm_length)
        )

        percentage_error = (
            absolute_error / float(osm_length)
        ) * 100.0

        results.append(
            {
                "pred_source": int(pred_source),
                "pred_target": int(pred_target),
                "osm_source": int(osm_source),
                "osm_target": int(osm_target),
                "predicted_path_length_m": float(
                    pred_length
                ),
                "osm_path_length_m": float(
                    osm_length
                ),
                "absolute_error_m": float(
                    absolute_error
                ),
                "percentage_error": float(
                    percentage_error
                ),
            }
        )

    if not results:
        raise RuntimeError(
            "No common reachable OD pairs were available "
            "for comparison."
        )

    # ---------------------------------------------------------------
    # 8. Aggregate validation metrics
    # ---------------------------------------------------------------
    absolute_errors = [
        r["absolute_error_m"]
        for r in results
    ]

    percentage_errors = [
        r["percentage_error"]
        for r in results
    ]

    predicted_lengths = [
        r["predicted_path_length_m"]
        for r in results
    ]

    osm_lengths = [
        r["osm_path_length_m"]
        for r in results
    ]

    mean_absolute_error = statistics.mean(
        absolute_errors
    )

    median_absolute_error = statistics.median(
        absolute_errors
    )

    mean_percentage_error = statistics.mean(
        percentage_errors
    )

    median_percentage_error = statistics.median(
        percentage_errors
    )

    max_percentage_error = max(
        percentage_errors
    )

    mean_predicted_length = statistics.mean(
        predicted_lengths
    )

    mean_osm_length = statistics.mean(
        osm_lengths
    )

    # ---------------------------------------------------------------
    # 9. Save results
    # ---------------------------------------------------------------
    result = {
        "dataset": "SpaceNet 3 Paris",
        "scene": "AOI_3_Paris_img84",

        "predicted_graph": {
            "nodes": G_pred.number_of_nodes(),
            "edges": G_pred.number_of_edges(),
        },

        "osm_graph": {
            "nodes": G_osm.number_of_nodes(),
            "edges": G_osm.number_of_edges(),
            "network_type": "drive",
            "comparison_mode": "undirected",
        },

        "sampling": {
            "requested_pairs": NUM_RANDOM_PAIRS,
            "sampled_pairs": len(sampled_pairs),
            "evaluated_pairs": len(results),
            "random_seed": RANDOM_SEED,
        },

        "path_length_validation": {
            "mean_predicted_path_m": mean_predicted_length,
            "mean_osm_path_m": mean_osm_length,
            "mean_absolute_path_length_error_m":
                mean_absolute_error,
            "median_absolute_path_length_error_m":
                median_absolute_error,
            "mean_path_length_error_percent":
                mean_percentage_error,
            "median_path_length_error_percent":
                median_percentage_error,
            "maximum_path_length_error_percent":
                max_percentage_error,
        },

        "pairs": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # 10. Save downloaded OSM graph
    # ---------------------------------------------------------------
    ox.io.save_graphml(
        G_osm,
        filepath=OSM_GRAPH_PATH,
    )

    # ---------------------------------------------------------------
    # 11. Print results
    # ---------------------------------------------------------------
    print()
    print("-" * 72)
    print("RESULTS")
    print("-" * 72)

    print(
        f"Evaluated OD pairs:         "
        f"{len(results)}"
    )

    print(
        f"Mean predicted path:        "
        f"{mean_predicted_length:.6f} m"
    )

    print(
        f"Mean OSM path:              "
        f"{mean_osm_length:.6f} m"
    )

    print(
        f"Mean absolute path error:   "
        f"{mean_absolute_error:.6f} m"
    )

    print(
        f"Median absolute error:      "
        f"{median_absolute_error:.6f} m"
    )

    print(
        f"Mean path-length error:     "
        f"{mean_percentage_error:.6f}%"
    )

    print(
        f"Median path-length error:   "
        f"{median_percentage_error:.6f}%"
    )

    print(
        f"Maximum path-length error:  "
        f"{max_percentage_error:.6f}%"
    )

    print()
    print(
        f"Saved validation: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved OSM graph:  "
        f"{OSM_GRAPH_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()