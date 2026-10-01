from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GRAPH_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spacenet_paris_graph"
)

GRAPHML_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_weighted.graphml"
)

BETWEENNESS_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_betweenness_report.json"
)

OUTPUT_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_rerouting_results.json"
)

TOP_GATEKEEPERS = 10


def load_graph() -> nx.Graph:

    graph = nx.read_graphml(
        GRAPHML_PATH
    )

    mapping = {}

    for node in graph.nodes:
        try:
            new_id = int(node)
        except (TypeError, ValueError):
            continue

        if node != new_id:
            mapping[node] = new_id

    if mapping:
        graph = nx.relabel_nodes(
            graph,
            mapping,
        )

    for _, _, data in graph.edges(
        data=True
    ):
        data["weight"] = float(
            data["weight"]
        )

    return graph


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def main() -> None:

    print("=" * 72)
    print(
        "SpaceNet Gatekeeper Rerouting Analysis"
    )
    print("=" * 72)

    graph = load_graph()

    report = load_json(
        BETWEENNESS_PATH
    )

    gatekeepers = report.get(
        "ranked_nodes",
        [],
    )[:TOP_GATEKEEPERS]

    results = []

    for rank, gatekeeper in enumerate(
        gatekeepers,
        start=1,
    ):

        failed_node = int(
            gatekeeper["node_id"]
        )

        if failed_node not in graph:
            continue

        perturbed = graph.copy()

        # -----------------------------------------------------------
        # Find all baseline shortest paths through the failed node.
        # -----------------------------------------------------------

        affected_pairs = []

        nodes = list(
            graph.nodes
        )

        for source_index, source in enumerate(
            nodes
        ):

            if source == failed_node:
                continue

            try:
                lengths, paths = (
                    nx.single_source_dijkstra(
                        graph,
                        source,
                        weight="weight",
                    )
                )
            except nx.NetworkXError:
                continue

            for target in nodes[
                source_index + 1:
            ]:

                if target == failed_node:
                    continue

                if target not in paths:
                    continue

                baseline_path = paths[
                    target
                ]

                if (
                    failed_node
                    not in baseline_path
                ):
                    continue

                baseline_length = float(
                    lengths[target]
                )

                affected_pairs.append(
                    {
                        "source": int(source),
                        "target": int(target),
                        "baseline_length_m": (
                            baseline_length
                        ),
                        "baseline_path": [
                            int(node)
                            for node in baseline_path
                        ],
                    }
                )

        print()
        print(
            f"Gatekeeper {rank}: node={failed_node}"
        )

        print(
            "  Baseline shortest paths through node:",
            len(affected_pairs),
        )

        # -----------------------------------------------------------
        # Remove the failed node.
        # -----------------------------------------------------------

        perturbed.remove_node(
            failed_node
        )

        reachable_affected = []
        unreachable_affected = []

        for pair in affected_pairs:

            source = pair["source"]
            target = pair["target"]

            if (
                source not in perturbed
                or target not in perturbed
            ):
                unreachable_affected.append(
                    pair
                )
                continue

            try:
                rerouted_length, rerouted_path = (
                    nx.single_source_dijkstra(
                        perturbed,
                        source,
                        target=target,
                        weight="weight",
                    )
                )
            except nx.NetworkXNoPath:
                unreachable_affected.append(
                    pair
                )
                continue

            rerouted_length = float(
                rerouted_length
            )

            increase_m = (
                rerouted_length
                - pair[
                    "baseline_length_m"
                ]
            )

            increase_percent = (
                (
                    rerouted_length
                    - pair[
                        "baseline_length_m"
                    ]
                )
                / pair[
                    "baseline_length_m"
                ]
                * 100.0
                if pair[
                    "baseline_length_m"
                ] > 0
                else 0.0
            )

            reachable_affected.append(
                {
                    **pair,
                    "rerouted_length_m": (
                        rerouted_length
                    ),
                    "rerouted_path": [
                        int(node)
                        for node in rerouted_path
                    ],
                    "travel_time_increase_m": (
                        increase_m
                    ),
                    "travel_time_increase_percent": (
                        increase_percent
                    ),
                }
            )

        # -----------------------------------------------------------
        # Summary statistics
        # -----------------------------------------------------------

        if reachable_affected:

            average_baseline = (
                sum(
                    item[
                        "baseline_length_m"
                    ]
                    for item in reachable_affected
                )
                / len(
                    reachable_affected
                )
            )

            average_rerouted = (
                sum(
                    item[
                        "rerouted_length_m"
                    ]
                    for item in reachable_affected
                )
                / len(
                    reachable_affected
                )
            )

            average_increase = (
                average_rerouted
                - average_baseline
            )

            average_increase_percent = (
                (
                    average_rerouted
                    - average_baseline
                )
                / average_baseline
                * 100.0
                if average_baseline > 0
                else 0.0
            )

            maximum_increase = max(
                item[
                    "travel_time_increase_m"
                ]
                for item in reachable_affected
            )

            maximum_increase_percent = max(
                item[
                    "travel_time_increase_percent"
                ]
                for item in reachable_affected
            )

        else:

            average_baseline = None
            average_rerouted = None
            average_increase = None
            average_increase_percent = None
            maximum_increase = None
            maximum_increase_percent = None

        reachable_count = len(
            reachable_affected
        )

        unreachable_count = len(
            unreachable_affected
        )

        total_affected = (
            len(affected_pairs)
        )

        if total_affected > 0:

            affected_reachability = (
                reachable_count
                / total_affected
            )

            affected_unreachability = (
                unreachable_count
                / total_affected
            )

        else:

            affected_reachability = 0.0
            affected_unreachability = 0.0

        scenario = {
            "rank": rank,
            "failed_node": failed_node,
            "betweenness_centrality": float(
                gatekeeper[
                    "betweenness_centrality"
                ]
            ),
            "affected_pairs": total_affected,
            "reachable_affected_pairs": (
                reachable_count
            ),
            "unreachable_affected_pairs": (
                unreachable_count
            ),
            "affected_pair_reachability": (
                affected_reachability
            ),
            "affected_pair_unreachability": (
                affected_unreachability
            ),
            "average_baseline_path_m": (
                average_baseline
            ),
            "average_rerouted_path_m": (
                average_rerouted
            ),
            "average_travel_time_increase_m": (
                average_increase
            ),
            "average_travel_time_increase_percent": (
                average_increase_percent
            ),
            "maximum_travel_time_increase_m": (
                maximum_increase
            ),
            "maximum_travel_time_increase_percent": (
                maximum_increase_percent
            ),
            "reachable_routes": (
                reachable_affected
            ),
            "unreachable_routes": (
                unreachable_affected
            ),
        }

        results.append(
            scenario
        )

        # -----------------------------------------------------------
        # Console report
        # -----------------------------------------------------------

        print(
            "  Reachable affected pairs:",
            reachable_count,
        )

        print(
            "  Unreachable affected pairs:",
            unreachable_count,
        )

        if average_increase is not None:

            print(
                "  Average rerouted path:",
                f"{average_rerouted:.6f} m",
            )

            print(
                "  Average path increase:",
                f"{average_increase:.6f} m",
            )

            print(
                "  Average path increase:",
                f"{average_increase_percent:.2f}%",
            )

            print(
                "  Maximum path increase:",
                f"{maximum_increase:.6f} m",
            )

        else:

            print(
                "  No affected pair remained reachable."
            )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    output = {
        "scene": "AOI_3_Paris_img84",
        "gatekeepers_tested": len(
            results
        ),
        "definition": (
            "OD pairs whose baseline shortest "
            "path passed through the failed "
            "gatekeeper node"
        ),
        "results": results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "Rerouting analysis complete."
    )
    print(
        "Results:",
        OUTPUT_PATH,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()