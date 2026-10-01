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

BASELINE_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_baseline_network.json"
)

BETWEENNESS_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_betweenness_report.json"
)

OUTPUT_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_node_failure_results.json"
)


def load_graph() -> nx.Graph:

    if not GRAPHML_PATH.exists():
        raise FileNotFoundError(
            f"GraphML not found:\n{GRAPHML_PATH}"
        )

    graph = nx.read_graphml(
        GRAPHML_PATH
    )

    mapping = {}

    for node in graph.nodes:

        try:
            numeric_id = int(node)
        except (
            TypeError,
            ValueError,
        ):
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


def load_json(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def connected_unordered_pairs(
    graph: nx.Graph,
) -> list[tuple[int, int]]:
    """
    Return unordered node pairs that are connected in `graph`.
    """

    pairs = []

    for component in nx.connected_components(
        graph
    ):

        nodes = sorted(
            component,
            key=lambda value: int(value),
        )

        for i in range(
            len(nodes)
        ):

            for j in range(
                i + 1,
                len(nodes),
            ):

                pairs.append(
                    (
                        nodes[i],
                        nodes[j],
                    )
                )

    return pairs


def average_distance_for_pairs(
    graph: nx.Graph,
    pairs: list[tuple[int, int]],
) -> float | None:
    """
    Calculate average weighted shortest-path length for exactly the
    supplied OD-pair population.

    This is the key correction: baseline and perturbed averages use
    exactly the same OD pairs for each failure scenario.
    """

    if not pairs:
        return None

    targets_by_source: dict[
        int,
        list[int],
    ] = {}

    for source, target in pairs:

        targets_by_source.setdefault(
            source,
            [],
        ).append(target)

    # Also include reverse direction internally when needed because
    # the graph is undirected, while counting each pair only once.
    sources = set(
        targets_by_source.keys()
    )

    total_distance = 0.0
    count = 0

    for source in sources:

        distances = (
            nx.single_source_dijkstra_path_length(
                graph,
                source,
                weight="weight",
            )
        )

        for target in targets_by_source[
            source
        ]:

            if target not in distances:
                raise RuntimeError(
                    "A supplied pair is unexpectedly "
                    "unreachable in the graph."
                )

            total_distance += float(
                distances[target]
            )

            count += 1

    if count == 0:
        return None

    return (
        total_distance
        / count
    )


def weighted_global_efficiency(
    graph: nx.Graph,
) -> float:

    nodes = list(
        graph.nodes
    )

    if len(nodes) < 2:
        return 0.0

    total_efficiency = 0.0

    total_possible_pairs = (
        len(nodes)
        * (len(nodes) - 1)
        // 2
    )

    if total_possible_pairs == 0:
        return 0.0

    for source_index, source in enumerate(
        nodes
    ):

        distances = (
            nx.single_source_dijkstra_path_length(
                graph,
                source,
                weight="weight",
            )
        )

        for target in nodes[
            source_index + 1:
        ]:

            if target not in distances:
                # Disconnected pair contributes zero efficiency.
                continue

            distance = float(
                distances[target]
            )

            if distance <= 0:
                continue

            total_efficiency += (
                1.0
                / distance
            )

    return (
        total_efficiency
        / total_possible_pairs
    )


def main() -> None:

    print("=" * 72)
    print(
        "SpaceNet Node Failure Simulation"
    )
    print("=" * 72)

    graph = load_graph()

    baseline = load_json(
        BASELINE_PATH
    )

    centrality_report = load_json(
        BETWEENNESS_PATH
    )

    baseline_average_all = float(
        baseline[
            "baseline_average_shortest_path_m"
        ]
    )

    baseline_efficiency = float(
        baseline[
            "baseline_global_efficiency"
        ]
    )

    ranked_nodes = centrality_report.get(
        "ranked_nodes",
        [],
    )

    gatekeepers = ranked_nodes[
        :10
    ]

    total_original_pairs = (
        graph.number_of_nodes()
        * (graph.number_of_nodes() - 1)
        // 2
    )

    print(
        "Baseline average shortest path:",
        f"{baseline_average_all:.6f} m",
    )

    print(
        "Baseline global efficiency:",
        f"{baseline_efficiency:.10f}",
    )

    print(
        "Total original OD pairs:",
        f"{total_original_pairs:,}",
    )

    print(
        "Gatekeepers tested:",
        len(gatekeepers),
    )

    results = []

    # ---------------------------------------------------------------
    # Node-failure scenarios
    # ---------------------------------------------------------------

    for rank, gatekeeper in enumerate(
        gatekeepers,
        start=1,
    ):

        node_id = int(
            gatekeeper["node_id"]
        )

        centrality = float(
            gatekeeper[
                "betweenness_centrality"
            ]
        )

        if node_id not in graph:
            print(
                f"[SKIP] Node {node_id} "
                "not found."
            )
            continue

        perturbed = graph.copy()

        perturbed.remove_node(
            node_id
        )

        # -----------------------------------------------------------
        # Remaining-node pair population
        # -----------------------------------------------------------

        remaining_nodes = list(
            perturbed.nodes
        )

        possible_remaining_pairs = (
            len(remaining_nodes)
            * (len(remaining_nodes) - 1)
            // 2
        )

        # These are exactly the OD pairs that remain reachable after
        # the failure.
        surviving_pairs = (
            connected_unordered_pairs(
                perturbed
            )
        )

        reachable_pairs = len(
            surviving_pairs
        )

        unreachable_pairs = (
            possible_remaining_pairs
            - reachable_pairs
        )

        # Pairs involving the failed node have also disappeared.
        removed_node_pairs = (
            graph.number_of_nodes()
            - 1
        )

        # -----------------------------------------------------------
        # Baseline and perturbed average path length
        # over the SAME surviving OD pairs
        # -----------------------------------------------------------

        surviving_baseline_average = (
            average_distance_for_pairs(
                graph,
                surviving_pairs,
            )
        )

        perturbed_average = (
            average_distance_for_pairs(
                perturbed,
                surviving_pairs,
            )
        )

        if (
            surviving_baseline_average
            is None
            or perturbed_average
            is None
            or perturbed_average <= 0
        ):

            resilience_index = None
            path_increase = None
            path_increase_percent = None

        else:

            resilience_index = (
                surviving_baseline_average
                / perturbed_average
            )

            path_increase = (
                perturbed_average
                - surviving_baseline_average
            )

            path_increase_percent = (
                (
                    perturbed_average
                    - surviving_baseline_average
                )
                / surviving_baseline_average
                * 100.0
            )

        # -----------------------------------------------------------
        # Efficiency
        # -----------------------------------------------------------

        perturbed_efficiency = (
            weighted_global_efficiency(
                perturbed
            )
        )

        efficiency_change = (
            perturbed_efficiency
            - baseline_efficiency
        )

        efficiency_change_percent = (
            (
                efficiency_change
                / baseline_efficiency
            )
            * 100.0
            if baseline_efficiency != 0
            else None
        )

        # -----------------------------------------------------------
        # Components
        # -----------------------------------------------------------

        component_count = (
            nx.number_connected_components(
                perturbed
            )
        )

        largest_component = max(
            (
                len(component)
                for component
                in nx.connected_components(
                    perturbed
                )
            ),
            default=0,
        )

        # -----------------------------------------------------------
        # Reachability fraction
        # -----------------------------------------------------------

        if possible_remaining_pairs > 0:

            reachability_fraction = (
                reachable_pairs
                / possible_remaining_pairs
            )

            disconnection_fraction = (
                unreachable_pairs
                / possible_remaining_pairs
            )

        else:

            reachability_fraction = 0.0
            disconnection_fraction = 0.0

        result = {
            "rank": int(rank),
            "node_id": int(node_id),
            "betweenness_centrality": float(
                centrality
            ),

            "remaining_nodes": int(
                perturbed.number_of_nodes()
            ),

            "remaining_edges": int(
                perturbed.number_of_edges()
            ),

            "removed_node_pairs": int(
                removed_node_pairs
            ),

            "possible_remaining_pairs": int(
                possible_remaining_pairs
            ),

            "reachable_pairs": int(
                reachable_pairs
            ),

            "unreachable_pairs": int(
                unreachable_pairs
            ),

            "reachability_fraction": float(
                reachability_fraction
            ),

            "disconnection_fraction": float(
                disconnection_fraction
            ),

            "connected_components": int(
                component_count
            ),

            "largest_component_nodes": int(
                largest_component
            ),

            # Baseline and perturbed averages are explicitly based
            # on the SAME surviving OD-pair population.
            "baseline_average_shortest_path_surviving_pairs_m": (
                float(
                    surviving_baseline_average
                )
                if surviving_baseline_average is not None
                else None
            ),

            "perturbed_average_shortest_path_m": (
                float(
                    perturbed_average
                )
                if perturbed_average is not None
                else None
            ),

            "path_increase_m": (
                float(
                    path_increase
                )
                if path_increase is not None
                else None
            ),

            "path_increase_percent": (
                float(
                    path_increase_percent
                )
                if path_increase_percent is not None
                else None
            ),

            "baseline_global_efficiency": float(
                baseline_efficiency
            ),

            "perturbed_global_efficiency": float(
                perturbed_efficiency
            ),

            "efficiency_change": float(
                efficiency_change
            ),

            "efficiency_change_percent": (
                float(
                    efficiency_change_percent
                )
                if efficiency_change_percent is not None
                else None
            ),

            "resilience_index": (
                float(
                    resilience_index
                )
                if resilience_index is not None
                else None
            ),
        }

        results.append(
            result
        )

        # -----------------------------------------------------------
        # Console
        # -----------------------------------------------------------

        print()
        print(
            f"Gatekeeper {rank}: "
            f"node={node_id}"
        )

        print(
            "  Betweenness:",
            f"{centrality:.6f}",
        )

        print(
            "  Components:",
            component_count,
        )

        print(
            "  Largest component:",
            largest_component,
        )

        print(
            "  Reachable pairs:",
            reachable_pairs,
        )

        print(
            "  Unreachable pairs:",
            unreachable_pairs,
        )

        print(
            "  Reachability:",
            f"{reachability_fraction * 100:.2f}%",
        )

        if (
            surviving_baseline_average
            is None
            or perturbed_average
            is None
        ):

            print(
                "  Resilience Index: undefined"
            )

        else:

            print(
                "  Baseline avg path "
                "(same surviving pairs):",
                f"{surviving_baseline_average:.6f} m",
            )

            print(
                "  Perturbed avg path:",
                f"{perturbed_average:.6f} m",
            )

            print(
                "  Path increase:",
                f"{path_increase:.6f} m",
            )

            print(
                "  Path increase:",
                f"{path_increase_percent:.2f}%",
            )

            print(
                "  Resilience Index:",
                f"{resilience_index:.6f}",
            )

        print(
            "  Perturbed efficiency:",
            f"{perturbed_efficiency:.10f}",
        )

    # ---------------------------------------------------------------
    # Save report
    # ---------------------------------------------------------------

    report = {
        "scene": "AOI_3_Paris_img84",

        "baseline_average_shortest_path_m": (
            baseline_average_all
        ),

        "baseline_global_efficiency": (
            baseline_efficiency
        ),

        "total_original_od_pairs": int(
            total_original_pairs
        ),

        "node_failures_tested": int(
            len(results)
        ),

        "resilience_definition": (
            "baseline average shortest path "
            "over the same surviving OD pairs "
            "/ perturbed average shortest path "
            "over those same OD pairs"
        ),

        "results": results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "Corrected node failure simulation complete."
    )

    print(
        "Results:",
        OUTPUT_PATH,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()