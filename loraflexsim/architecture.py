from dataclasses import dataclass
from typing import List

from .launcher.node import Node as _Node
from .launcher.gateway import Gateway as _Gateway
from .launcher.server import NetworkServer as _NetworkServer
from .launcher.config_loader import load_config


@dataclass
class NodeConfig:
    """Configuration minimale d'un nœud."""

    x: float
    y: float
    sf: int = 7
    tx_power: float = 14.0


@dataclass
class GatewayConfig:
    """Configuration d'une passerelle."""

    x: float
    y: float
    altitude: float = 0.0
    rx_gain_dB: float = 0.0


@dataclass
class SimulationConfig:
    """Liste des entités à instancier."""

    nodes: List[NodeConfig]
    gateways: List[GatewayConfig]

    @classmethod
    def from_ini(cls, path: str) -> "SimulationConfig":
        """Charge une configuration INI ou JSON compatible FLoRa."""
        ndata, gdata, _, _ = load_config(path)
        nodes = [NodeConfig(**nd) for nd in ndata]
        gws = [GatewayConfig(**gw) for gw in gdata]
        return cls(nodes, gws)


class Node(_Node):
    """Nœud hérité du modèle Python."""

    def __init__(self, node_id: int, cfg: NodeConfig, **kwargs):
        super().__init__(node_id, cfg.x, cfg.y, cfg.sf, cfg.tx_power, **kwargs)


class Gateway(_Gateway):
    """Passerelle LoRa."""

    def __init__(self, gateway_id: int, cfg: GatewayConfig):
        super().__init__(
            gateway_id,
            cfg.x,
            cfg.y,
            cfg.altitude,
            rx_gain_dB=cfg.rx_gain_dB,
        )


class NetworkServer(_NetworkServer):
    """Serveur réseau LoRa."""

    def add_gateway(self, gw: _Gateway) -> None:
        """Register ``gw`` if not already present.

        The lightweight wrapper originally added gateways blindly which could
        result in duplicates when ``add_gateway`` was called multiple times with
        the same instance.  The internal :attr:`gateways` list is now checked to
        ensure each gateway is stored only once, mirroring the behaviour of
        :meth:`register_node` for nodes.
        """

        if gw not in self.gateways:
            self.gateways.append(gw)

    def register_node(self, node: _Node) -> None:
        if node not in self.nodes:
            self.nodes.append(node)


def build_from_config(cfg: SimulationConfig) -> tuple[List[Node], List[Gateway], NetworkServer]:
    """Instancie nœuds et passerelles selon ``cfg``."""
    nodes = [Node(i, ncfg) for i, ncfg in enumerate(cfg.nodes)]
    gws = [Gateway(i, gcfg) for i, gcfg in enumerate(cfg.gateways)]
    server = NetworkServer(process_delay=0.001)
    server.nodes = nodes
    server.gateways = gws
    return nodes, gws, server
