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