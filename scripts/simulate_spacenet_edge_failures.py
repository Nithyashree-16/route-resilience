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

OUTPUT_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_edge_failure_results.json"
)

TOP_EDGES = 10


def load_graph() -> nx.Graph:

    graph = nx.read_graphml(
        GRAPHML_PATH
    )

    mapping = {}

    for node in graph.nodes:
        try:
            numeric_id = int(node)
        except (TypeError, ValueError):
            continue

        if node != numeric_id:
            mapping[node] = numeric_id

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

        data["length_m"] = float(
            data.get(
                "length_m",
                data["weight"],
            )
        )

    return graph


def main() -> None:

    print("=" * 72)
    print(
        "SpaceNet Edge Failure + Rerouting Analysis"
    )
    print("=" * 72)

    graph = load_graph()

    print(
        "Nodes:",
        graph.number_of_nodes(),
    )

    print(
        "Edges:",
        graph.number_of_edges(),
    )

    # ---------------------------------------------------------------
    # Edge betweenness
    # ---------------------------------------------------------------

    print()
    print(
        "Computing weighted edge betweenness..."
    )

    edge_betweenness = (
        nx.edge_betweenness_centrality(
            graph,
            weight="weight",
            normalized=True,
        )
    )

    ranked_edges = sorted(
        edge_betweenness.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    selected_edges = ranked_edges[
        :TOP_EDGES
    ]

    results = []

    # ---------------------------------------------------------------
    # Simulate each critical edge failure
    # ---------------------------------------------------------------

    for rank, (
        edge,
        centrality,
    ) in enumerate(
        selected_edges,
        start=1,
    ):

        node_a, node_b = edge

        edge_data = graph[
            node_a
        ][
            node_b
        ]

        edge_id = int(
            edge_data.get(
                "edge_id",
                -1,
            )
        )

        edge_length = float(
            edge_data["weight"]
        )

        print()
        print(
            f"Critical edge {rank}: "
            f"{node_a} -> {node_b}"
        )

        print(
            "  Edge ID:",
            edge_id,
        )

        print(
            "  Edge betweenness:",
            f"{centrality:.6f}",
        )

        print(
            "  Edge length:",
            f"{edge_length:.6f} m",
        )

        # -----------------------------------------------------------
        # Identify baseline OD routes using this edge
        # -----------------------------------------------------------

        affected_pairs = []

        nodes = list(
            graph.nodes
        )

        for source_index, source in enumerate(
            nodes
        ):

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

                if target not in paths:
                    continue

                path = paths[
                    target
                ]

                uses_edge = False

                for i in range(
                    len(path) - 1
                ):

                    a = path[i]
                    b = path[i + 1]

                    if (
                        (
                            a == node_a
                            and b == node_b
                        )
                        or
                        (
                            a == node_b
                            and b == node_a
                        )
                    ):
                        uses_edge = True
                        break

                if not uses_edge:
                    continue

                affected_pairs.append(
                    {
                        "source": int(
                            source
                        ),
                        "target": int(
                            target
                        ),
                        "baseline_length_m": (
                            float(
                                lengths[target]
                            )
                        ),
                        "baseline_path": [
                            int(x)
                            for x in path
                        ],
                    }
                )

        print(
            "  Baseline routes using edge:",
            len(affected_pairs),
        )

        # -----------------------------------------------------------
        # Remove edge
        # -----------------------------------------------------------

        perturbed = graph.copy()

        perturbed.remove_edge(
            node_a,
            node_b,
        )

        reachable_routes = []
        unreachable_routes = []

        for pair in affected_pairs:

            source = pair[
                "source"
            ]

            target = pair[
                "target"
            ]

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

                unreachable_routes.append(
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

            reachable_routes.append(
                {
                    **pair,
                    "rerouted_length_m": (
                        rerouted_length
                    ),
                    "rerouted_path": [
                        int(x)
                        for x in rerouted_path
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
        # Scenario summary
        # -----------------------------------------------------------

        reachable_count = len(
            reachable_routes
        )

        unreachable_count = len(
            unreachable_routes
        )

        if reachable_routes:

            average_baseline = (
                sum(
                    r[
                        "baseline_length_m"
                    ]
                    for r in reachable_routes
                )
                / reachable_count
            )

            average_rerouted = (
                sum(
                    r[
                        "rerouted_length_m"
                    ]
                    for r in reachable_routes
                )
                / reachable_count
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
                r[
                    "travel_time_increase_m"
                ]
                for r in reachable_routes
            )

            maximum_increase_percent = max(
                r[
                    "travel_time_increase_percent"
                ]
                for r in reachable_routes
            )

        else:

            average_baseline = None
            average_rerouted = None
            average_increase = None
            average_increase_percent = None
            maximum_increase = None
            maximum_increase_percent = None

        result = {
            "rank": int(rank),
            "node_a": int(node_a),
            "node_b": int(node_b),
            "edge_id": int(edge_id),
            "edge_betweenness": float(
                centrality
            ),
            "edge_length_m": float(
                edge_length
            ),
            "affected_pairs": int(
                len(affected_pairs)
            ),
            "reachable_rerouted_pairs": int(
                reachable_count
            ),
            "unreachable_pairs": int(
                unreachable_count
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
                reachable_routes
            ),
            "unreachable_routes": (
                unreachable_routes
            ),
        }

        results.append(
            result
        )

        # -----------------------------------------------------------
        # Console
        # -----------------------------------------------------------

        print(
            "  Reachable rerouted pairs:",
            reachable_count,
        )

        print(
            "  Unreachable pairs:",
            unreachable_count,
        )

        if average_increase is not None:

            print(
                "  Average rerouted path:",
                f"{average_rerouted:.6f} m",
            )

            print(
                "  Average increase:",
                f"{average_increase:.6f} m",
            )

            print(
                "  Average increase:",
                f"{average_increase_percent:.2f}%",
            )

            print(
                "  Maximum increase:",
                f"{maximum_increase:.6f} m",
            )

        else:

            print(
                "  No affected route remained reachable."
            )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    output = {
        "scene": "AOI_3_Paris_img84",
        "edges_tested": len(
            results
        ),
        "definition": (
            "OD pairs whose baseline shortest "
            "path used the failed edge"
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
        "Edge failure analysis complete."
    )

    print(
        "Results:",
        OUTPUT_PATH,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()