"""
Cerberus Heads — Modular hot-swappable payload mission modules.
Each head gives Cerberus a different mind for a different mission.
"""

from cerberus.heads.base_head import BaseHead, HeadInfo, HeadState
from cerberus.heads.weed_scanner import WeedScannerHead
from cerberus.heads.surveillance import SurveillanceHead
from cerberus.heads.env_logger import EnvLoggerHead
from cerberus.heads.pest_deterrent import PestDeterrentHead
from cerberus.heads.bird_watcher import BirdWatcherHead
from cerberus.heads.microclimate import MicroclimateHead

__all__ = [
    "BaseHead",
    "HeadInfo",
    "HeadState",
    "WeedScannerHead",
    "SurveillanceHead",
    "EnvLoggerHead",
    "PestDeterrentHead",
    "BirdWatcherHead",
    "MicroclimateHead",
]