from pathlib import Path
import json


BASE = Path("data/processed/spacenet_paris_graph")


OUTPUT = BASE / "AOI_3_Paris_img84_final_results.json"


def load_json(path):
    if not path.exists():
        print(f"WARNING: missing {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 72)
    print("SpaceNet 3 Paris - Final Results Summary")
    print("=" * 72)

    mask = load_json(
        BASE / "AOI_3_Paris_img84_mask_validation.json"
    )

    healing = load_json(
        BASE / "AOI_3_Paris_img84_healing_report.json"
    )

    weighted = load_json(
        BASE / "AOI_3_Paris_img84_weighted_graph_report.json"
    )

    baseline = load_json(
        BASE / "AOI_3_Paris_img84_baseline_network.json"
    )

    betweenness = load_json(
        BASE / "AOI_3_Paris_img84_betweenness_report.json"
    )

    node_failure = load_json(
        BASE / "AOI_3_Paris_img84_node_failure_results.json"
    )

    rerouting = load_json(
        BASE / "AOI_3_Paris_img84_rerouting_results.json"
    )

    edge_failure = load_json(
        BASE / "AOI_3_Paris_img84_reroutable_edge_299_310.json"
    )

    osm = load_json(
        BASE / "AOI_3_Paris_img84_osm_validation.json"
    )

    standard = mask.get("standard_metrics", {})
    tolerance = mask.get("tolerance_metrics", {})

    path_validation = osm.get(
        "path_length_validation", {}
    )

    summary = {
        "dataset": "SpaceNet 3",
        "aoi": "AOI_3_Paris",
        "scene": "AOI_3_Paris_img84",

        "segmentation": {
            "standard_iou": standard.get("iou"),
            "dice": standard.get("dice"),
            "precision": standard.get("precision"),
            "recall": standard.get("recall"),
            "tp": standard.get("tp"),
            "fp": standard.get("fp"),
            "fn": standard.get("fn"),

            "relaxed_3px_iou":
                tolerance.get("3", {}).get(
                    "relaxed_iou"
                ),

            "relaxed_3px_length_completeness":
                tolerance.get("3", {}).get(
                    "length_completeness_recall"
                ),

            "relaxed_5px_iou":
                tolerance.get("5", {}).get(
                    "relaxed_iou"
                ),

            "relaxed_5px_length_completeness":
                tolerance.get("5", {}).get(
                    "length_completeness_recall"
                ),
        },

        "graph_reconstruction": {
            "nodes":
                weighted.get("nodes"),

            "edges":
                weighted.get("edges"),

            "connected_components":
                weighted.get("connected_components"),

            "total_network_length_m":
                weighted.get("total_network_length_m"),

            "average_edge_length_m":
                weighted.get("average_edge_length_m"),

            "min_edge_length_m":
                weighted.get("min_edge_length_m"),

            "max_edge_length_m":
                weighted.get("max_edge_length_m"),

            "bridge_edges":
                weighted.get("bridge_edges"),

            "average_node_degree":
                weighted.get("average_node_degree"),

            "original_road_length_m":
                healing.get("original_road_length_m"),

            "added_bridge_length_m":
                healing.get("added_bridge_length_m"),

            "healed_road_length_m":
                healing.get("healed_road_length_m"),

            "connectivity_ratio_percent":
                healing.get("connectivity_ratio_percent"),
        },

        "network_analysis": {
            "baseline_average_shortest_path_m":
                baseline.get(
                    "baseline_average_shortest_path_m"
                ),

            "baseline_global_efficiency":
                baseline.get(
                    "baseline_global_efficiency"
                ),

            "top_gatekeeper_node":
                betweenness.get(
                    "top_10_gatekeepers",
                    [{}]
                )[0] if betweenness.get(
                    "top_10_gatekeepers"
                ) else None,
        },

        "node_failure": {
            "results": node_failure
        },

        "rerouting": {
            "results": rerouting
        },

        "reroutable_edge": {
            "node_a": edge_failure.get(
                "edge", {}
            ).get("node_a"),

            "node_b": edge_failure.get(
                "edge", {}
            ).get("node_b"),

            "edge_betweenness":
                edge_failure.get(
                    "edge", {}
                ).get("edge_betweenness"),

            "affected_pairs":
                edge_failure.get(
                    "impact", {}
                ).get("affected_pairs"),

            "reroutable_pairs":
                edge_failure.get(
                    "impact", {}
                ).get("reroutable_pairs"),

            "unreachable_pairs":
                edge_failure.get(
                    "impact", {}
                ).get("unreachable_pairs"),

            "mean_path_increase_m":
                edge_failure.get(
                    "path_increase_m", {}
                ).get("mean"),

            "median_path_increase_m":
                edge_failure.get(
                    "path_increase_m", {}
                ).get("median"),

            "p95_path_increase_m":
                edge_failure.get(
                    "path_increase_m", {}
                ).get("p95"),

            "maximum_path_increase_m":
                edge_failure.get(
                    "path_increase_m", {}
                ).get("maximum"),

            "assumed_speed_kmh":
                edge_failure.get(
                    "travel_time", {}
                ).get("assumed_speed_kmh"),

            "mean_time_increase_seconds":
                edge_failure.get(
                    "travel_time", {}
                ).get("mean_increase_seconds"),

            "maximum_time_increase_seconds":
                edge_failure.get(
                    "travel_time", {}
                ).get("maximum_increase_seconds"),
        },

        "osm_validation": {
            "evaluated_od_pairs":
                osm.get(
                    "sampling", {}
                ).get("evaluated_pairs"),

            "osm_nodes":
                osm.get(
                    "osm_graph", {}
                ).get("nodes"),

            "osm_edges":
                osm.get(
                    "osm_graph", {}
                ).get("edges"),

            "mean_predicted_path_m":
                path_validation.get(
                    "mean_predicted_path_m"
                ),

            "mean_osm_path_m":
                path_validation.get(
                    "mean_osm_path_m"
                ),

            "mean_absolute_path_length_error_m":
                path_validation.get(
                    "mean_absolute_path_length_error_m"
                ),

            "mean_path_length_error_percent":
                path_validation.get(
                    "mean_path_length_error_percent"
                ),

            "median_path_length_error_percent":
                path_validation.get(
                    "median_path_length_error_percent"
                ),

            "maximum_path_length_error_percent":
                path_validation.get(
                    "maximum_path_length_error_percent"
                ),

            "interpretation":
                "OSM shortest-path discrepancy using "
                "buffered OSM drive network and nearest-node "
                "matching; not a pure segmentation accuracy metric."
        }
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(f"Saved: {OUTPUT}")
    print()
    print("SpaceNet evaluation is frozen.")
    print("No further SpaceNet model tuning is required.")
    print("=" * 72)


if __name__ == "__main__":
    main()