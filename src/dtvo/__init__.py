"""DTVO - Yan, Li, Zhou 2022, "Robust and efficient edge-based visual odometry" (RGB-D mode).

    tracking.py       Sec. 4   DT maps, weighted edge alignment, motion hypotheses
    local_mapping.py  Sec. 5   keyframes, depth propagation, sliding-window optimisation
    system.py                  tracking + mapping per frame (DTVO class)
    evaluation.py              ATE / RPE / per-axis RMSE against ground truth
    se3.py                     pose maths
"""
from .system import DTVO, DTVOConfig  # noqa: F401
