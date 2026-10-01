# Route Resilience

## Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility

An end-to-end geospatial deep learning and graph-theoretic framework for extracting road networks from satellite imagery under occlusion, reconstructing topologically connected road graphs, identifying critical infrastructure nodes, and simulating network failures for urban resilience analysis.

## Pipeline

Satellite Imagery
→ Occlusion-Aware Segmentation
→ Skeletonization
→ Graph Construction
→ MST + Disjoint Set Healing
→ Betweenness Centrality
→ Gatekeeper Detection
→ Node Ablation
→ Rerouting
→ Resilience Index
→ Interactive GIS Dashboard

This directory contains local datasets used by the Route Resilience project.

## Raw datasets

- sentinel2/     - Sentinel-2 satellite imagery
- resourcesat/   - Resourcesat LISS-IV imagery
- cartosat/      - Cartosat imagery provided for challenge experimentation
- spacenet/      - SpaceNet road extraction dataset
- deepglobe/     - DeepGlobe road extraction dataset
- opensatmap/    - OpenSatMap imagery/annotations
- osm/           - OpenStreetMap road vectors

## Interim data

- tiles/         - geospatial image tiles
- masks/         - rasterized road masks
- occluded/      - synthetic occlusion samples

## Processed data

- train/
- val/
- test/

## Metadata

Dataset manifests, geographic extents, coordinate reference systems,
tile indexes, and preprocessing metadata.

Raw and processed datasets are intentionally excluded from Git.