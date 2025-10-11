# Initialisation du package LoRaFlexSim
from .node import Node
from .gateway import Gateway
from .channel import Channel
from .advanced_channel import AdvancedChannel
from .multichannel import MultiChannel
from .server import NetworkServer
from .simulator import Simulator
from .duty_cycle import DutyCycleManager
from .smooth_mobility import SmoothMobility
from .random_waypoint import RandomWaypoint
from .planned_random_waypoint import PlannedRandomWaypoint
from .path_mobility import PathMobility
from .terrain_mobility import TerrainMapMobility
from .gauss_markov import GaussMarkov
from .gps_mobility import GPSTraceMobility, MultiGPSTraceMobility
from .trace3d_mobility import Trace3DMobility
from .map_loader import load_map
from .lorawan import LoRaWANFrame, compute_rx1, compute_rx2
from .downlink_scheduler import DownlinkScheduler
from .omnet_model import OmnetModel
from .omnet_phy import OmnetPHY
from .flora_cpp import FloraCppPHY
from .obstacle_loss import ObstacleLoss
from . import (
    adr_standard_1,
    adr_2,
    adr_ml,
    explora_sf,
    explora_at,
    adr_lite,
    adr_max,
    radr,
    qos,
)

# Mapping of ADR strategy names to their implementation modules
ADR_MODULES = {
    "ADR 1": adr_standard_1,
    "ADR 2": adr_2,
    "ADR_ML": adr_ml,
    "EXPLoRa-SF": explora_sf,
    "EXPLoRa-AT": explora_at,
    "ADR-Lite": adr_lite,
    "ADR-Max": adr_max,
    "RADR": radr,
}

__all__ = [
    "Node",
    "Gateway",
    "Channel",
    "AdvancedChannel",
    "MultiChannel",
    "NetworkServer",
    "Simulator",
    "DutyCycleManager",
    "SmoothMobility",
    "RandomWaypoint",
    "PlannedRandomWaypoint",
    "PathMobility",
    "TerrainMapMobility",
    "GaussMarkov",
    "Trace3DMobility",
    "GPSTraceMobility",
    "MultiGPSTraceMobility",
    "load_map",
    "LoRaWANFrame",
    "compute_rx1",
    "compute_rx2",
    "DownlinkScheduler",
    "OmnetModel",
    "OmnetPHY",
    "ObstacleLoss",
    "FloraCppPHY",
    "adr_standard_1",
    "adr_2",
    "adr_ml",
    "adr_lite",
    "explora_sf",
    "explora_at",
    "adr_max",
    "radr",
    "ADR_MODULES",
    "qos",
]

for name in __all__:
    globals()[name] = locals()[name]
