from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
from pyproj import Geod


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
    / "AOI_3_Paris_img84_nodes.geojson"
)

EDGES_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_edges.geojson"
)

HEALED_EDGES_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_healed_edges.geojson"
)

HEALED_NODES_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_healed_nodes.geojson"
)

REPORT_PATH = (
    GRAPH_DIR
    / "AOI_3_Paris_img84_healing_report.json"
)


# ---------------------------------------------------------------------
# Healing constraints
# ---------------------------------------------------------------------

MAX_GAP_METERS = 100.0
MAX_ANGLE_DEGREES = 45.0

GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------
# Disjoint Set Union
# ---------------------------------------------------------------------

class DisjointSet:

    def __init__(self, items):
        self.parent = {
            item: item
            for item in items
        }

        self.rank = {
            item: 0
            for item in items
        }

    def find(self, item):

        parent = self.parent[item]

        if parent != item:
            self.parent[item] = self.find(parent)

        return self.parent[item]

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

        return True


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def angular_difference(
    a: float,
    b: float,
) -> float:

    return abs(
        (a - b + 180.0) % 360.0
        - 180.0
    )


def bearing_between(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:

    azimuth, _, _ = GEOD.inv(
        x1,
        y1,
        x2,
        y2,
    )

    return azimuth % 360.0


def distance_between(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:

    _, _, distance = GEOD.inv(
        x1,
        y1,
        x2,
        y2,
    )

    return abs(distance)


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


# ---------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------

def build_graph(
    nodes_geojson: dict,
    edges_geojson: dict,
) -> nx.Graph:

    graph = nx.Graph()

    # Nodes
    for feature in nodes_geojson["features"]:

        properties = feature["properties"]

        node_id = int(
            properties["node_id"]
        )

        coordinates = (
            feature["geometry"]["coordinates"]
        )

        graph.add_node(
            node_id,
            x=float(coordinates[0]),
            y=float(coordinates[1]),
            node_type=properties.get(
                "node_type",
                "unknown",
            ),
            original_degree=int(
                properties.get(
                    "degree",
                    0,
                )
            ),
        )

    # Edges
    for feature in edges_geojson["features"]:

        properties = feature["properties"]

        start_node = int(
            properties["start_node"]
        )

        end_node = int(
            properties["end_node"]
        )

        if start_node == end_node:
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
            length_m=float(
                properties.get(
                    "length_m",
                    0.0,
                )
            ),
            pixel_count=int(
                properties.get(
                    "pixel_count",
                    0,
                )
            ),
            bridge=False,
            coordinates=feature[
                "geometry"
            ]["coordinates"],
        )

    return graph


# ---------------------------------------------------------------------
# Remove zero-degree artifacts
# ---------------------------------------------------------------------

def remove_isolated_graph_nodes(
    graph: nx.Graph,
) -> list[int]:

    isolated_nodes = [
        node
        for node in graph.nodes
        if graph.degree[node] == 0
    ]

    graph.remove_nodes_from(
        isolated_nodes
    )

    return isolated_nodes


# ---------------------------------------------------------------------
# Endpoint road direction
# ---------------------------------------------------------------------

def endpoint_road_bearing(
    graph: nx.Graph,
    node_id: int,
) -> float | None:
    """
    Bearing from the endpoint INTO its existing road component.

    For healing, the desired continuation direction is the opposite
    direction, i.e. road_bearing + 180 degrees.
    """

    neighbors = list(
        graph.neighbors(node_id)
    )

    if not neighbors:
        return None

    neighbor = neighbors[0]

    node = graph.nodes[node_id]
    other = graph.nodes[neighbor]

    return bearing_between(
        node["x"],
        node["y"],
        other["x"],
        other["y"],
    )


def endpoint_continuation_bearing(
    graph: nx.Graph,
    node_id: int,
) -> float | None:
    """
    Direction in which a missing continuation should extend from
    the endpoint.

    This is opposite to the bearing toward the existing road.
    """

    road_bearing = endpoint_road_bearing(
        graph,
        node_id,
    )

    if road_bearing is None:
        return None

    return (
        road_bearing + 180.0
    ) % 360.0


# ---------------------------------------------------------------------
# Candidate healing connections
# ---------------------------------------------------------------------

def generate_candidates(
    graph: nx.Graph,
) -> list[dict]:

    components = list(
        nx.connected_components(
            graph
        )
    )

    component_id = {}

    for index, component in enumerate(
        components
    ):
        for node in component:
            component_id[node] = index

    endpoints = [
        node
        for node in graph.nodes
        if graph.degree[node] == 1
    ]

    continuation_bearings = {}

    for node in endpoints:
        continuation_bearings[node] = (
            endpoint_continuation_bearing(
                graph,
                node,
            )
        )

    candidates = []

    for i in range(
        len(endpoints)
    ):

        node_a = endpoints[i]

        for j in range(
            i + 1,
            len(endpoints),
        ):

            node_b = endpoints[j]

            # Only heal different components.
            if (
                component_id[node_a]
                == component_id[node_b]
            ):
                continue

            continuation_a = (
                continuation_bearings[node_a]
            )

            continuation_b = (
                continuation_bearings[node_b]
            )

            if (
                continuation_a is None
                or continuation_b is None
            ):
                continue

            point_a = graph.nodes[
                node_a
            ]

            point_b = graph.nodes[
                node_b
            ]

            distance_m = distance_between(
                point_a["x"],
                point_a["y"],
                point_b["x"],
                point_b["y"],
            )

            if (
                distance_m
                > MAX_GAP_METERS
            ):
                continue

            # Direction of candidate bridge from A toward B.
            bridge_bearing_a = (
                bearing_between(
                    point_a["x"],
                    point_a["y"],
                    point_b["x"],
                    point_b["y"],
                )
            )

            # Direction of candidate bridge from B toward A.
            bridge_bearing_b = (
                bearing_between(
                    point_b["x"],
                    point_b["y"],
                    point_a["x"],
                    point_a["y"],
                )
            )

            # Compare bridge direction against the expected
            # continuation direction, NOT the direction back into
            # the existing road.
            angle_a = angular_difference(
                continuation_a,
                bridge_bearing_a,
            )

            angle_b = angular_difference(
                continuation_b,
                bridge_bearing_b,
            )

            if (
                angle_a
                > MAX_ANGLE_DEGREES
            ):
                continue

            if (
                angle_b
                > MAX_ANGLE_DEGREES
            ):
                continue

            mean_angle = (
                angle_a
                + angle_b
            ) / 2.0

            normalized_distance = (
                distance_m
                / MAX_GAP_METERS
            )

            normalized_angle = (
                mean_angle
                / MAX_ANGLE_DEGREES
            )

            cost = (
                normalized_distance
                + normalized_angle
            )

            candidates.append(
                {
                    "node_a": int(node_a),
                    "node_b": int(node_b),
                    "distance_m": float(
                        distance_m
                    ),
                    "angle_a_deg": float(
                        angle_a
                    ),
                    "angle_b_deg": float(
                        angle_b
                    ),
                    "mean_angle_deg": float(
                        mean_angle
                    ),
                    "cost": float(
                        cost
                    ),
                }
            )

    candidates.sort(
        key=lambda item: item[
            "cost"
        ]
    )

    return candidates


# ---------------------------------------------------------------------
# MST / DSU
# ---------------------------------------------------------------------

def select_bridges(
    graph: nx.Graph,
    candidates: list[dict],
) -> list[dict]:

    dsu = DisjointSet(
        graph.nodes
    )

    # Existing road graph defines initial components.
    for node_a, node_b in graph.edges:
        dsu.union(
            node_a,
            node_b,
        )

    selected = []

    for candidate in candidates:

        node_a = candidate[
            "node_a"
        ]

        node_b = candidate[
            "node_b"
        ]

        if (
            dsu.find(node_a)
            == dsu.find(node_b)
        ):
            continue

        if dsu.union(
            node_a,
            node_b,
        ):
            selected.append(
                candidate
            )

    return selected


# ---------------------------------------------------------------------
# Filtered node GeoJSON
# ---------------------------------------------------------------------

def filtered_nodes_geojson(
    nodes_geojson: dict,
    graph: nx.Graph,
) -> dict:

    valid_nodes = set(
        graph.nodes
    )

    features = []

    for feature in nodes_geojson[
        "features"
    ]:

        node_id = int(
            feature["properties"][
                "node_id"
            ]
        )

        if node_id not in valid_nodes:
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": feature[
                    "geometry"
                ],
                "properties": feature[
                    "properties"
                ],
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
    }


# ---------------------------------------------------------------------
# Write healed edges
# ---------------------------------------------------------------------

def write_healed_edges(
    graph: nx.Graph,
    original_edges: dict,
    selected_bridges: list[dict],
) -> dict:

    features = []

    valid_nodes = set(
        graph.nodes
    )

    # Existing edges
    for feature in original_edges[
        "features"
    ]:

        properties = dict(
            feature["properties"]
        )

        start_node = int(
            properties["start_node"]
        )

        end_node = int(
            properties["end_node"]
        )

        if (
            start_node not in valid_nodes
            or end_node not in valid_nodes
        ):
            continue

        properties["bridge"] = False

        features.append(
            {
                "type": "Feature",
                "geometry": feature[
                    "geometry"
                ],
                "properties": properties,
            }
        )

    # New bridges
    existing_ids = [
        int(
            feature["properties"].get(
                "edge_id",
                -1,
            )
        )
        for feature in original_edges[
            "features"
        ]
    ]

    next_edge_id = (
        max(
            existing_ids or [-1]
        )
        + 1
    )

    for bridge_index, bridge in enumerate(
        selected_bridges
    ):

        node_a = bridge[
            "node_a"
        ]

        node_b = bridge[
            "node_b"
        ]

        point_a = graph.nodes[
            node_a
        ]

        point_b = graph.nodes[
            node_b
        ]

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [
                            float(
                                point_a["x"]
                            ),
                            float(
                                point_a["y"]
                            ),
                        ],
                        [
                            float(
                                point_b["x"]
                            ),
                            float(
                                point_b["y"]
                            ),
                        ],
                    ],
                },
                "properties": {
                    "edge_id": int(
                        next_edge_id
                        + bridge_index
                    ),
                    "start_node": int(
                        node_a
                    ),
                    "end_node": int(
                        node_b
                    ),
                    "length_m": float(
                        bridge["distance_m"]
                    ),
                    "pixel_count": 0,
                    "bridge": True,
                    "bridge_method": "mst_dsu",
                    "gap_distance_m": float(
                        bridge["distance_m"]
                    ),
                    "angle_a_deg": float(
                        bridge["angle_a_deg"]
                    ),
                    "angle_b_deg": float(
                        bridge["angle_b_deg"]
                    ),
                    "mean_angle_deg": float(
                        bridge["mean_angle_deg"]
                    ),
                    "bridge_cost": float(
                        bridge["cost"]
                    ),
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print(
        "SpaceNet MST + DSU Graph Healing"
    )
    print("=" * 72)

    nodes_geojson = load_geojson(
        NODES_PATH
    )

    edges_geojson = load_geojson(
        EDGES_PATH
    )

    graph = build_graph(
        nodes_geojson,
        edges_geojson,
    )

    print(
        "Nodes loaded:",
        graph.number_of_nodes(),
    )

    print(
        "Edges loaded:",
        graph.number_of_edges(),
    )

    # -------------------------------------------------------------
    # Remove degree-0 artifacts
    # -------------------------------------------------------------

    isolated_nodes = (
        remove_isolated_graph_nodes(
            graph
        )
    )

    print(
        "Removed degree-0 graph artifacts:",
        len(isolated_nodes),
    )

    if isolated_nodes:
        print(
            "Removed node IDs:",
            isolated_nodes,
        )

    print(
        "Nodes after filtering:",
        graph.number_of_nodes(),
    )

    print(
        "Edges after filtering:",
        graph.number_of_edges(),
    )

    # -------------------------------------------------------------
    # Components before healing
    # -------------------------------------------------------------

    original_components = list(
        nx.connected_components(
            graph
        )
    )

    original_component_count = len(
        original_components
    )

    largest_before = max(
        (
            len(component)
            for component in original_components
        ),
        default=0,
    )

    print(
        "Original connected components:",
        original_component_count,
    )

    print(
        "Largest component before healing:",
        largest_before,
        "nodes",
    )

    # -------------------------------------------------------------
    # Candidate generation
    # -------------------------------------------------------------

    candidates = generate_candidates(
        graph
    )

    print()
    print(
        "Valid endpoint-gap candidates:",
        len(candidates),
    )

    # -------------------------------------------------------------
    # MST + DSU
    # -------------------------------------------------------------

    selected_bridges = select_bridges(
        graph,
        candidates,
    )

    print(
        "Selected MST/DSU bridges:",
        len(selected_bridges),
    )

    # -------------------------------------------------------------
    # Healed graph
    # -------------------------------------------------------------

    healed_graph = graph.copy()

    for bridge in selected_bridges:

        node_a = bridge[
            "node_a"
        ]

        node_b = bridge[
            "node_b"
        ]

        point_a = healed_graph.nodes[
            node_a
        ]

        point_b = healed_graph.nodes[
            node_b
        ]

        healed_graph.add_edge(
            node_a,
            node_b,
            edge_id=-1,
            length_m=bridge[
                "distance_m"
            ],
            pixel_count=0,
            bridge=True,
            bridge_method="mst_dsu",
            gap_distance_m=bridge[
                "distance_m"
            ],
            angle_a_deg=bridge[
                "angle_a_deg"
            ],
            angle_b_deg=bridge[
                "angle_b_deg"
            ],
            mean_angle_deg=bridge[
                "mean_angle_deg"
            ],
            bridge_cost=bridge[
                "cost"
            ],
            coordinates=[
                [
                    float(
                        point_a["x"]
                    ),
                    float(
                        point_a["y"]
                    ),
                ],
                [
                    float(
                        point_b["x"]
                    ),
                    float(
                        point_b["y"]
                    ),
                ],
            ],
        )

    healed_components = list(
        nx.connected_components(
            healed_graph
        )
    )

    healed_component_count = len(
        healed_components
    )

    largest_after = max(
        (
            len(component)
            for component in healed_components
        ),
        default=0,
    )

    # -------------------------------------------------------------
    # Connectivity ratio
    # -------------------------------------------------------------

    if largest_before > 0:

        connectivity_ratio = (
            (
                largest_after
                - largest_before
            )
            / largest_before
            * 100.0
        )

    else:

        connectivity_ratio = 0.0

    # -------------------------------------------------------------
    # Length metrics
    # -------------------------------------------------------------

    bridge_length = sum(
        bridge["distance_m"]
        for bridge in selected_bridges
    )

    original_length = sum(
        float(
            feature[
                "properties"
            ].get(
                "length_m",
                0.0,
            )
        )
        for feature in edges_geojson[
            "features"
        ]
    )

    healed_length = (
        original_length
        + bridge_length
    )

    # -------------------------------------------------------------
    # Write nodes
    # -------------------------------------------------------------

    healed_nodes_geojson = (
        filtered_nodes_geojson(
            nodes_geojson,
            healed_graph,
        )
    )

    with HEALED_NODES_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            healed_nodes_geojson,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Write edges
    # -------------------------------------------------------------

    healed_edges_geojson = (
        write_healed_edges(
            graph,
            edges_geojson,
            selected_bridges,
        )
    )

    with HEALED_EDGES_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            healed_edges_geojson,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    remaining_components = [
        sorted(
            int(node)
            for node in component
        )
        for component in healed_components
        if len(component)
        < largest_after
    ]

    report = {
        "scene": "AOI_3_Paris_img84",
        "loaded_nodes": int(
            graph.number_of_nodes()
            + len(isolated_nodes)
        ),
        "removed_degree_zero_nodes": int(
            len(isolated_nodes)
        ),
        "nodes_after_filtering": int(
            graph.number_of_nodes()
        ),
        "original_edges": int(
            graph.number_of_edges()
        ),
        "original_components": int(
            original_component_count
        ),
        "largest_component_before_nodes": int(
            largest_before
        ),
        "candidate_gap_count": int(
            len(candidates)
        ),
        "selected_bridge_count": int(
            len(selected_bridges)
        ),
        "healed_components": int(
            healed_component_count
        ),
        "largest_component_after_nodes": int(
            largest_after
        ),
        "connectivity_ratio_percent": float(
            connectivity_ratio
        ),
        "max_gap_meters": float(
            MAX_GAP_METERS
        ),
        "max_angle_degrees": float(
            MAX_ANGLE_DEGREES
        ),
        "original_length_m": float(
            original_length
        ),
        "bridge_length_m": float(
            bridge_length
        ),
        "healed_length_m": float(
            healed_length
        ),
        "remaining_components": remaining_components,
        "bridges": selected_bridges,
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

    # -------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------

    print()
    print(
        "Healed connected components:",
        healed_component_count,
    )

    print(
        "Largest component after healing:",
        largest_after,
        "nodes",
    )

    print(
        "Connectivity Ratio:",
        f"{connectivity_ratio:.2f}%",
    )

    print(
        "Original road length:",
        f"{original_length:,.2f} m",
    )

    print(
        "Added bridge length:",
        f"{bridge_length:,.2f} m",
    )

    print(
        "Healed road length:",
        f"{healed_length:,.2f} m",
    )

    if remaining_components:

        print()
        print(
            "Remaining smaller components:"
        )

        for component in remaining_components:
            print(
                component
            )

    print()
    print(
        "Healed edges:",
        HEALED_EDGES_PATH,
    )

    print(
        "Healed nodes:",
        HEALED_NODES_PATH,
    )

    print(
        "Healing report:",
        REPORT_PATH,
    )

    print("=" * 72)
    print(
        "MST + DSU healing complete."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()