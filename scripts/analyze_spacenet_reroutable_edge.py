from pathlib import Path
import json
import statistics
import networkx as nx

GRAPH_PATH = Path(
    "data/processed/spacenet_paris_graph/"
    "AOI_3_Paris_img84_weighted.graphml"
)

OUTPUT_PATH = Path(
    "data/processed/spacenet_paris_graph/"
    "AOI_3_Paris_img84_reroutable_edge_299_310.json"
)

NODE_A = 299
NODE_B = 310

# Configurable dashboard scenario.
# 30 km/h is used only to convert extra distance into an estimated time increase.
ASSUMED_SPEED_KMH = 30.0


def normalize_node_ids(G):
    mapping = {}
    for node in G.nodes:
        try:
            mapping[node] = int(node)
        except (TypeError, ValueError):
            mapping[node] = node

    if any(k != v for k, v in mapping.items()):
        G = nx.relabel_nodes(G, mapping, copy=True)

    return G


def edge_key(u, v):
    return tuple(sorted((u, v)))


def main():
    print("=" * 72)
    print("SpaceNet Reroutable Edge Analysis")
    print("=" * 72)

    G = nx.read_graphml(GRAPH_PATH)
    G = normalize_node_ids(G)

    if not G.has_edge(NODE_A, NODE_B):
        raise RuntimeError(
            f"Edge {NODE_A} -> {NODE_B} not found in graph."
        )

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Testing edge: {NODE_A} -> {NODE_B}")

    # Baseline shortest paths.
    baseline_lengths = dict(
        nx.all_pairs_dijkstra_path_length(G, weight="weight")
    )

    # Store one deterministic baseline path for each OD pair.
    baseline_paths = {}

    nodes = sorted(G.nodes)

    for source in nodes:
        paths = nx.single_source_dijkstra_path(
            G,
            source,
            weight="weight"
        )

        baseline_paths[source] = paths

    # Remove the candidate edge.
    G_perturbed = G.copy()
    G_perturbed.remove_edge(NODE_A, NODE_B)

    perturbed_lengths = dict(
        nx.all_pairs_dijkstra_path_length(
            G_perturbed,
            weight="weight"
        )
    )

    increases = []
    affected_pairs = 0
    reroutable_pairs = 0
    unreachable_pairs = 0

    detailed_samples = []

    target_edge = edge_key(NODE_A, NODE_B)

    for i, source in enumerate(nodes):
        for target in nodes[i + 1:]:
            if target not in baseline_lengths[source]:
                continue

            baseline_distance = baseline_lengths[source][target]

            baseline_path = baseline_paths[source].get(target)
            if baseline_path is None:
                continue

            # Was the failed edge used by the baseline shortest path?
            uses_failed_edge = any(
                edge_key(u, v) == target_edge
                for u, v in zip(
                    baseline_path[:-1],
                    baseline_path[1:]
                )
            )

            if not uses_failed_edge:
                continue

            affected_pairs += 1

            if target not in perturbed_lengths.get(source, {}):
                unreachable_pairs += 1
                continue

            perturbed_distance = perturbed_lengths[source][target]
            increase = perturbed_distance - baseline_distance

            reroutable_pairs += 1
            increases.append(increase)

            if len(detailed_samples) < 10:
                rerouted_path = nx.shortest_path(
                    G_perturbed,
                    source,
                    target,
                    weight="weight"
                )

                detailed_samples.append({
                    "source": source,
                    "target": target,
                    "baseline_distance_m": baseline_distance,
                    "perturbed_distance_m": perturbed_distance,
                    "increase_m": increase,
                    "baseline_path": baseline_path,
                    "rerouted_path": rerouted_path,
                })

    if not increases:
        raise RuntimeError(
            "No reachable rerouted paths were found."
        )

    increases_sorted = sorted(increases)

    mean_increase = statistics.mean(increases)
    median_increase = statistics.median(increases)

    p95_index = int(0.95 * (len(increases_sorted) - 1))
    p95_increase = increases_sorted[p95_index]

    max_increase = max(increases)

    meters_per_second = (
        ASSUMED_SPEED_KMH * 1000.0 / 3600.0
    )

    mean_time_seconds = mean_increase / meters_per_second
    median_time_seconds = median_increase / meters_per_second
    p95_time_seconds = p95_increase / meters_per_second
    max_time_seconds = max_increase / meters_per_second

    result = {
        "edge": {
            "node_a": NODE_A,
            "node_b": NODE_B,
            "edge_betweenness": 0.407307,
        },
        "network": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
        "impact": {
            "affected_pairs": affected_pairs,
            "reroutable_pairs": reroutable_pairs,
            "unreachable_pairs": unreachable_pairs,
            "reroutable_fraction": (
                reroutable_pairs / affected_pairs
                if affected_pairs else 0.0
            ),
        },
        "path_increase_m": {
            "mean": mean_increase,
            "median": median_increase,
            "p95": p95_increase,
            "maximum": max_increase,
        },
        "travel_time": {
            "assumed_speed_kmh": ASSUMED_SPEED_KMH,
            "mean_increase_seconds": mean_time_seconds,
            "median_increase_seconds": median_time_seconds,
            "p95_increase_seconds": p95_time_seconds,
            "maximum_increase_seconds": max_time_seconds,
        },
        "sample_reroutes": detailed_samples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"Affected pairs:      {affected_pairs}")
    print(f"Reroutable pairs:    {reroutable_pairs}")
    print(f"Unreachable pairs:   {unreachable_pairs}")
    print()
    print(f"Mean path increase:  {mean_increase:.6f} m")
    print(f"Median increase:     {median_increase:.6f} m")
    print(f"P95 increase:        {p95_increase:.6f} m")
    print(f"Maximum increase:    {max_increase:.6f} m")
    print()
    print(
        f"Assumed speed:       "
        f"{ASSUMED_SPEED_KMH:.1f} km/h"
    )
    print(
        f"Mean time increase:  "
        f"{mean_time_seconds:.3f} s"
    )
    print(
        f"Median time increase: "
        f"{median_time_seconds:.3f} s"
    )
    print(
        f"P95 time increase:   "
        f"{p95_time_seconds:.3f} s"
    )
    print(
        f"Maximum time increase: "
        f"{max_time_seconds:.3f} s"
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()