"""Dynamic-robust RGB-D edge VO.

Implemented so far (see papers/pipeline_flowchart.md):
    dataset.py     - TUM / Bonn sequence loading, timestamp association, camera presets
    preprocess.py  - Box 1: prepare inputs (grayscale, metric depth, validity, pyramid)
    edges.py       - Box 2: find edges (adaptive Canny, chain filtering, depth tagging)
"""
