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

NODES_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_healed_nodes.geojson"
)

CENTRALITY_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_betweenness_nodes.geojson"
)

REPORT_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_betweenness_report.json"
)


# Number of highest-centrality nodes to explicitly mark
# as gatekeepers for the prototype.
GATEKEEPER_COUNT = 10


def load_nodes_geojson(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Node GeoJSON not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def main() -> None:

    print("=" * 72)
    print(
        "SpaceNet Betweenness Centrality"
    )
    print("=" * 72)

    if not GRAPHML_PATH.exists():
        raise FileNotFoundError(
            f"GraphML not found:\n{GRAPHML_PATH}"
        )

    graph = nx.read_graphml(
        GRAPHML_PATH
    )

    # GraphML may load node IDs as strings.
    # Convert them back to integer IDs where possible.
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

    print(
        "Nodes:",
        graph.number_of_nodes(),
    )

    print(
        "Edges:",
        graph.number_of_edges(),
    )

    components = list(
        nx.connected_components(
            graph
        )
    )

    print(
        "Connected components:",
        len(components),
    )

    if len(components) != 1:
        raise ValueError(
            "Betweenness analysis requires "
            "a connected graph."
        )

    # ---------------------------------------------------------------
    # Weighted Betweenness Centrality
    # ---------------------------------------------------------------

    print()
    print(
        "Computing weighted betweenness..."
    )

    centrality = nx.betweenness_centrality(
        graph,
        weight="weight",
        normalized=True,
    )

    # ---------------------------------------------------------------
    # Rank nodes by centrality
    # ---------------------------------------------------------------

    ranked_nodes = sorted(
        centrality.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    gatekeeper_ids = {
        node_id
        for node_id, _ in ranked_nodes[
            :GATEKEEPER_COUNT
        ]
    }

    # Store centrality in graph.
    for node_id, value in centrality.items():
        graph.nodes[node_id][
            "betweenness_centrality"
        ] = float(value)

        graph.nodes[node_id][
            "gatekeeper"
        ] = bool(
            node_id in gatekeeper_ids
        )

    # ---------------------------------------------------------------
    # Load geographic node coordinates
    # ---------------------------------------------------------------

    nodes_geojson = load_nodes_geojson(
        NODES_PATH
    )

    node_lookup = {}

    for feature in nodes_geojson[
        "features"
    ]:

        properties = feature[
            "properties"
        ]

        node_id = int(
            properties["node_id"]
        )

        coordinates = feature[
            "geometry"
        ]["coordinates"]

        node_lookup[node_id] = {
            "x": float(
                coordinates[0]
            ),
            "y": float(
                coordinates[1]
            ),
            "node_type": str(
                properties.get(
                    "node_type",
                    "unknown",
                )
            ),
            "degree": int(
                properties.get(
                    "degree",
                    0,
                )
            ),
        }

    # ---------------------------------------------------------------
    # Create centrality GeoJSON
    # ---------------------------------------------------------------

    features = []

    for node_id in graph.nodes:

        if node_id not in node_lookup:
            continue

        info = node_lookup[
            node_id
        ]

        centrality_value = float(
            centrality.get(
                node_id,
                0.0,
            )
        )

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        info["x"],
                        info["y"],
                    ],
                },
                "properties": {
                    "node_id": int(
                        node_id
                    ),
                    "node_type": info[
                        "node_type"
                    ],
                    "degree": int(
                        graph.degree[
                            node_id
                        ]
                    ),
                    "betweenness_centrality": (
                        centrality_value
                    ),
                    "gatekeeper": bool(
                        node_id
                        in gatekeeper_ids
                    ),
                },
            }
        )

    centrality_geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with CENTRALITY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            centrality_geojson,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    maximum_centrality = max(
        centrality.values(),
        default=0.0,
    )

    mean_centrality = (
        sum(
            centrality.values()
        )
        / len(centrality)
        if centrality
        else 0.0
    )

    report_nodes = []

    for rank, (
        node_id,
        value,
    ) in enumerate(
        ranked_nodes,
        start=1,
    ):

        report_nodes.append(
            {
                "rank": rank,
                "node_id": int(
                    node_id
                ),
                "betweenness_centrality": float(
                    value
                ),
                "degree": int(
                    graph.degree[
                        node_id
                    ]
                ),
                "gatekeeper": bool(
                    node_id
                    in gatekeeper_ids
                ),
            }
        )

    report = {
        "scene": "AOI_3_Paris_img84",
        "nodes": int(
            graph.number_of_nodes()
        ),
        "edges": int(
            graph.number_of_edges()
        ),
        "weighted": True,
        "weight": "length_m",
        "normalized": True,
        "gatekeeper_count": int(
            GATEKEEPER_COUNT
        ),
        "maximum_betweenness": float(
            maximum_centrality
        ),
        "mean_betweenness": float(
            mean_centrality
        ),
        "ranked_nodes": report_nodes,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------

    print()
    print(
        "Maximum betweenness:",
        f"{maximum_centrality:.6f}",
    )

    print(
        "Mean betweenness:",
        f"{mean_centrality:.6f}",
    )

    print()
    print(
        f"Top {GATEKEEPER_COUNT} gatekeeper nodes:"
    )

    print(
        "-" * 72
    )

    for rank, (
        node_id,
        value,
    ) in enumerate(
        ranked_nodes[
            :GATEKEEPER_COUNT
        ],
        start=1,
    ):

        print(
            f"{rank:02d}. "
            f"node={node_id} "
            f"centrality={value:.6f} "
            f"degree={graph.degree[node_id]}"
        )

    print()
    print(
        "Centrality GeoJSON:",
        CENTRALITY_PATH,
    )

    print(
        "Report:",
        REPORT_PATH,
    )

    print("=" * 72)
    print(
        "Betweenness analysis complete."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()