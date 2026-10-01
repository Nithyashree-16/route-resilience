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

BETWEENNESS_REPORT = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_betweenness_report.json"
)

OUTPUT_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_baseline_network.json"
)


def load_graph() -> nx.Graph:
    if not GRAPHML_PATH.exists():
        raise FileNotFoundError(
            f"Weighted graph not found:\n{GRAPHML_PATH}"
        )

    graph = nx.read_graphml(
        GRAPHML_PATH
    )

    # Convert GraphML string node IDs back to integers.
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

    # Ensure edge weights are numeric.
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


def weighted_global_efficiency(
    graph: nx.Graph,
) -> float:
    """
    Weighted global network efficiency.

    For every ordered pair of distinct nodes:

        efficiency = 1 / shortest_path_length

    The global efficiency is the mean over all ordered pairs.
    """
    nodes = list(
        graph.nodes
    )

    if len(nodes) < 2:
        return 0.0

    total_efficiency = 0.0
    pair_count = 0

    for source in nodes:

        lengths = nx.single_source_dijkstra_path_length(
            graph,
            source,
            weight="weight",
        )

        for target, distance in lengths.items():

            if target == source:
                continue

            if distance <= 0:
                continue

            total_efficiency += (
                1.0 / distance
            )

            pair_count += 1

    if pair_count == 0:
        return 0.0

    return (
        total_efficiency
        / pair_count
    )


def main() -> None:

    print("=" * 72)
    print(
        "SpaceNet Baseline Network Analysis"
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

    component_count = (
        nx.number_connected_components(
            graph
        )
    )

    print(
        "Connected components:",
        component_count,
    )

    if component_count != 1:
        raise ValueError(
            "Baseline network must be connected."
        )

    # ---------------------------------------------------------------
    # Baseline average shortest path
    # ---------------------------------------------------------------

    print()
    print(
        "Computing weighted average shortest path..."
    )

    average_shortest_path = (
        nx.average_shortest_path_length(
            graph,
            weight="weight",
        )
    )

    # ---------------------------------------------------------------
    # Baseline global efficiency
    # ---------------------------------------------------------------

    print(
        "Computing weighted global efficiency..."
    )

    global_efficiency = (
        weighted_global_efficiency(
            graph
        )
    )

    # ---------------------------------------------------------------
    # Additional baseline information
    # ---------------------------------------------------------------

    all_degrees = [
        degree
        for _, degree
        in graph.degree()
    ]

    total_network_length = sum(
        float(
            data["weight"]
        )
        for _, _, data
        in graph.edges(
            data=True
        )
    )

    average_edge_length = (
        total_network_length
        / graph.number_of_edges()
    )

    # ---------------------------------------------------------------
    # Load gatekeeper information
    # ---------------------------------------------------------------

    gatekeepers = []

    if BETWEENNESS_REPORT.exists():

        with BETWEENNESS_REPORT.open(
            "r",
            encoding="utf-8",
        ) as f:
            centrality_report = json.load(
                f
            )

        gatekeepers = (
            centrality_report.get(
                "ranked_nodes",
                [],
            )[:10]
        )

    # ---------------------------------------------------------------
    # Save baseline report
    # ---------------------------------------------------------------

    report = {
        "scene": "AOI_3_Paris_img84",

        "nodes": int(
            graph.number_of_nodes()
        ),

        "edges": int(
            graph.number_of_edges()
        ),

        "connected_components": int(
            component_count
        ),

        "weight_definition": (
            "geodesic road length in metres"
        ),

        "total_network_length_m": float(
            total_network_length
        ),

        "average_edge_length_m": float(
            average_edge_length
        ),

        "average_node_degree": float(
            sum(all_degrees)
            / len(all_degrees)
        ),

        "baseline_average_shortest_path_m": (
            float(
                average_shortest_path
            )
        ),

        "baseline_global_efficiency": (
            float(
                global_efficiency
            )
        ),

        "gatekeepers": gatekeepers,
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

    # ---------------------------------------------------------------
    # Console output
    # ---------------------------------------------------------------

    print()
    print(
        "Baseline average shortest path:",
        f"{average_shortest_path:.6f} m",
    )

    print(
        "Baseline global efficiency:",
        f"{global_efficiency:.10f}",
    )

    print(
        "Total network length:",
        f"{total_network_length:,.2f} m",
    )

    print(
        "Average edge length:",
        f"{average_edge_length:.6f} m",
    )

    print()
    print(
        "Baseline report:",
        OUTPUT_PATH,
    )

    print("=" * 72)
    print(
        "Baseline network analysis complete."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
    