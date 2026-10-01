from pathlib import Path

import networkx as nx
import osmnx as ox
import rasterio


RASTER_PATH = Path(
    "data/processed/spacenet_paris_predictions/"
    "AOI_3_Paris_img84_road_mask.tif"
)


def main():
    print("=" * 72)
    print("SpaceNet / OSM Coverage Diagnostic")
    print("=" * 72)

    with rasterio.open(RASTER_PATH) as src:
        b = src.bounds

    # Add a modest geographic buffer around the SpaceNet scene.
    # This is ONLY for diagnosing OSM coverage, not for the final metric.
    pad = 0.005

    bbox = (
        b.left - pad,
        b.bottom - pad,
        b.right + pad,
        b.top + pad,
    )

    print(
        f"Buffered bbox: "
        f"left={bbox[0]:.8f}, "
        f"bottom={bbox[1]:.8f}, "
        f"right={bbox[2]:.8f}, "
        f"top={bbox[3]:.8f}"
    )

    ox.settings.use_cache = True
    ox.settings.requests_timeout = 180

    print()
    print("Querying OSM road network with buffered AOI...")

    G = ox.graph.graph_from_bbox(
        bbox=bbox,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    print()
    print("OSM graph:")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # Convert to undirected graph for connected-component analysis.
    G_undirected = G.to_undirected()

    components = list(
        nx.connected_components(G_undirected)
    )

    print(
        f"Connected components: "
        f"{len(components)}"
    )

    component_sizes = sorted(
        [len(c) for c in components],
        reverse=True,
    )

    print(
        "Component sizes:",
        component_sizes[:20]
    )

    # Count highway tags.
    highway_counts = {}

    for _, _, data in G.edges(data=True):
        highway = data.get("highway")

        if isinstance(highway, list):
            values = highway
        else:
            values = [highway]

        for value in values:
            if value is not None:
                highway_counts[value] = (
                    highway_counts.get(value, 0) + 1
                )

    print()
    print("Highway types:")

    for highway, count in sorted(
        highway_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {highway}: {count}")

    print()
    print("Sample OSM nodes:")

    for node in list(G.nodes)[:20]:
        data = G.nodes[node]

        print(
            node,
            "lon=", data.get("x"),
            "lat=", data.get("y"),
        )

    output_path = Path(
        "data/processed/spacenet_paris_graph/"
        "AOI_3_Paris_osm_coverage_diagnostic.graphml"
    )

    ox.io.save_graphml(
        G,
        filepath=output_path,
    )

    print()
    print(f"Saved: {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()