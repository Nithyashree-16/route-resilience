from __future__ import annotations

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


def main() -> None:

    print("=" * 72)
    print(
        "SpaceNet Reroutable Edge Search"
    )
    print("=" * 72)

    graph = load_graph()

    # -------------------------------------------------------------
    # Edge betweenness
    # -------------------------------------------------------------

    centrality = (
        nx.edge_betweenness_centrality(
            graph,
            weight="weight",
            normalized=True,
        )
    )

    # -------------------------------------------------------------
    # Find non-bridge edges
    # -------------------------------------------------------------

    bridge_edges = set(
        nx.bridges(graph)
    )

    candidates = []

    for edge, value in centrality.items():

        node_a, node_b = edge

        canonical = tuple(
            sorted(
                (
                    node_a,
                    node_b,
                )
            )
        )

        is_bridge = canonical in {
            tuple(
                sorted(
                    bridge
                )
            )
            for bridge in bridge_edges
        }

        if is_bridge:
            continue

        candidates.append(
            (
                float(value),
                node_a,
                node_b,
            )
        )

    candidates.sort(
        reverse=True
    )

    print(
        "Total edges:",
        graph.number_of_edges(),
    )

    print(
        "Bridge edges:",
        len(bridge_edges),
    )

    print(
        "Non-bridge edges:",
        len(candidates),
    )

    if not candidates:
        print()
        print(
            "No non-bridge edges exist in the graph."
        )
        return

    # -------------------------------------------------------------
    # Test candidates until one produces a reroute
    # -------------------------------------------------------------

    nodes = list(
        graph.nodes
    )

    for rank, (
        edge_centrality,
        node_a,
        node_b,
    ) in enumerate(
        candidates,
        start=1,
    ):

        perturbed = graph.copy()

        if not perturbed.has_edge(
            node_a,
            node_b,
        ):
            continue

        perturbed.remove_edge(
            node_a,
            node_b,
        )

        affected_pairs = 0
        rerouted_pairs = 0
        maximum_increase = 0.0

        for source_index, source in enumerate(
            nodes
        ):

            if source not in perturbed:
                continue

            try:
                baseline_lengths, baseline_paths = (
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

                if target == source:
                    continue

                if target not in baseline_paths:
                    continue

                baseline_path = (
                    baseline_paths[target]
                )

                uses_edge = False

                for i in range(
                    len(baseline_path) - 1
                ):

                    a = baseline_path[i]
                    b = baseline_path[i + 1]

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

                affected_pairs += 1

                try:
                    rerouted_length, _ = (
                        nx.single_source_dijkstra(
                            perturbed,
                            source,
                            target=target,
                            weight="weight",
                        )
                    )
                except nx.NetworkXNoPath:
                    continue

                rerouted_pairs += 1

                baseline_length = float(
                    baseline_lengths[target]
                )

                increase = (
                    float(rerouted_length)
                    - baseline_length
                )

                if increase > maximum_increase:
                    maximum_increase = increase

        print(
            f"Candidate {rank}: "
            f"{node_a} -> {node_b} "
            f"edge_betweenness={edge_centrality:.6f} "
            f"affected={affected_pairs} "
            f"reroutable={rerouted_pairs}"
        )

        if rerouted_pairs > 0:
            print()
            print(
                "REROUTABLE EDGE FOUND"
            )
            print(
                "Node A:",
                node_a,
            )
            print(
                "Node B:",
                node_b,
            )
            print(
                "Edge betweenness:",
                f"{edge_centrality:.6f}",
            )
            print(
                "Affected pairs:",
                affected_pairs,
            )
            print(
                "Reroutable pairs:",
                rerouted_pairs,
            )
            print(
                "Maximum path increase:",
                f"{maximum_increase:.6f} m",
            )
            print("=" * 72)
            return

    print()
    print(
        "No non-bridge edge produced a reachable reroute."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()