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

NODES_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_healed_nodes.geojson"
)

EDGES_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_healed_edges.geojson"
)

GRAPHML_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_weighted.graphml"
)

JSON_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_weighted_graph.json"
)

REPORT_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_weighted_graph_report.json"
)


def load_geojson(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def build_weighted_graph(
    nodes_geojson: dict,
    edges_geojson: dict,
) -> nx.Graph:

    graph = nx.Graph(
        scene="AOI_3_Paris_img84",
        weight_definition="geodesic_road_length_m",
        crs="EPSG:4326",
    )

    # -------------------------------------------------------------
    # Nodes
    # -------------------------------------------------------------

    for feature in nodes_geojson[
        "features"
    ]:

        properties = feature[
            "properties"
        ]

        node_id = int(
            properties[
                "node_id"
            ]
        )

        coordinates = feature[
            "geometry"
        ]["coordinates"]

        graph.add_node(
            node_id,
            x=float(
                coordinates[0]
            ),
            y=float(
                coordinates[1]
            ),
            node_type=str(
                properties.get(
                    "node_type",
                    "unknown",
                )
            ),
            degree=int(
                properties.get(
                    "degree",
                    0,
                )
            ),
        )

    # -------------------------------------------------------------
    # Edges
    # -------------------------------------------------------------

    for feature in edges_geojson[
        "features"
    ]:

        properties = feature[
            "properties"
        ]

        start_node = int(
            properties[
                "start_node"
            ]
        )

        end_node = int(
            properties[
                "end_node"
            ]
        )

        if (
            start_node not in graph
            or end_node not in graph
        ):
            continue

        length_m = float(
            properties.get(
                "length_m",
                0.0,
            )
        )

        if length_m <= 0:
            continue

        graph.add_edge(
            start_node,
            end_node,
            edge_id=int(
                properties.get(
                    "edge_id",
                    -1,
                )
            ),
            length_m=length_m,
            weight=length_m,
            pixel_count=int(
                properties.get(
                    "pixel_count",
                    0,
                )
            ),
            bridge=bool(
                properties.get(
                    "bridge",
                    False,
                )
            ),
            bridge_method=str(
                properties.get(
                    "bridge_method",
                    "",
                )
            ),
        )

    return graph


def write_json_graph(
    graph: nx.Graph,
    path: Path,
) -> None:

    data = {
        "directed": False,
        "multigraph": False,
        "graph": dict(
            graph.graph
        ),
        "nodes": [],
        "edges": [],
    }

    for node_id, attributes in graph.nodes(
        data=True
    ):

        node_record = {
            "id": int(node_id)
        }

        for key, value in attributes.items():
            node_record[key] = value

        data[
            "nodes"
        ].append(
            node_record
        )

    for node_a, node_b, attributes in graph.edges(
        data=True
    ):

        edge_record = {
            "source": int(
                node_a
            ),
            "target": int(
                node_b
            ),
        }

        for key, value in attributes.items():
            edge_record[key] = value

        data[
            "edges"
        ].append(
            edge_record
        )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )


def main() -> None:

    print("=" * 72)
    print(
        "SpaceNet Weighted Road Graph"
    )
    print("=" * 72)

    nodes_geojson = load_geojson(
        NODES_PATH
    )

    edges_geojson = load_geojson(
        EDGES_PATH
    )

    graph = build_weighted_graph(
        nodes_geojson,
        edges_geojson,
    )

    print(
        "Nodes:",
        graph.number_of_nodes(),
    )

    print(
        "Edges:",
        graph.number_of_edges(),
    )

    print(
        "Connected components:",
        nx.number_connected_components(
            graph
        ),
    )

    if graph.number_of_nodes() == 0:
        raise ValueError(
            "Weighted graph contains no nodes."
        )

    if graph.number_of_edges() == 0:
        raise ValueError(
            "Weighted graph contains no edges."
        )

    # -------------------------------------------------------------
    # Weight sanity check
    # -------------------------------------------------------------

    edge_lengths = [
        float(
            data["weight"]
        )
        for _, _, data
        in graph.edges(
            data=True
        )
    ]

    bridge_edges = [
        data
        for _, _, data
        in graph.edges(
            data=True
        )
        if bool(
            data.get(
                "bridge",
                False,
            )
        )
    ]

    total_length = sum(
        edge_lengths
    )

    minimum_length = min(
        edge_lengths
    )

    maximum_length = max(
        edge_lengths
    )

    average_length = (
        total_length
        / len(edge_lengths)
    )

    # -------------------------------------------------------------
    # Degree statistics
    # -------------------------------------------------------------

    degrees = [
        degree
        for _, degree
        in graph.degree()
    ]

    minimum_degree = min(
        degrees
    )

    maximum_degree = max(
        degrees
    )

    average_degree = (
        sum(degrees)
        / len(degrees)
    )

    # -------------------------------------------------------------
    # Save GraphML
    # -------------------------------------------------------------

    nx.write_graphml(
        graph,
        GRAPHML_PATH,
    )

    # -------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------

    write_json_graph(
        graph,
        JSON_PATH,
    )

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    report = {
        "scene": "AOI_3_Paris_img84",
        "nodes": int(
            graph.number_of_nodes()
        ),
        "edges": int(
            graph.number_of_edges()
        ),
        "connected_components": int(
            nx.number_connected_components(
                graph
            )
        ),
        "weight_definition": (
            "geodesic road length in metres"
        ),
        "total_length_m": float(
            total_length
        ),
        "minimum_edge_length_m": float(
            minimum_length
        ),
        "maximum_edge_length_m": float(
            maximum_length
        ),
        "average_edge_length_m": float(
            average_length
        ),
        "bridge_edges": int(
            len(bridge_edges)
        ),
        "minimum_node_degree": int(
            minimum_degree
        ),
        "maximum_node_degree": int(
            maximum_degree
        ),
        "average_node_degree": float(
            average_degree
        ),
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

    print()
    print(
        "Total network length:",
        f"{total_length:,.2f} m",
    )

    print(
        "Average edge length:",
        f"{average_length:,.2f} m",
    )

    print(
        "Minimum edge length:",
        f"{minimum_length:,.2f} m",
    )

    print(
        "Maximum edge length:",
        f"{maximum_length:,.2f} m",
    )

    print(
        "Bridge edges:",
        len(bridge_edges),
    )

    print(
        "Average node degree:",
        f"{average_degree:.2f}",
    )

    print()
    print(
        "GraphML:",
        GRAPHML_PATH,
    )

    print(
        "JSON:",
        JSON_PATH,
    )

    print(
        "Report:",
        REPORT_PATH,
    )

    print("=" * 72)
    print(
        "Weighted graph construction complete."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()