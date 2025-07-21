import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VERSION_4.launcher.node import Node  # noqa: E402
from VERSION_4.launcher.gateway import Gateway  # noqa: E402
from VERSION_4.launcher.channel import Channel  # noqa: E402
from VERSION_4.launcher.server import NetworkServer, JoinServer  # noqa: E402
from VERSION_4.launcher.lorawan import (
    JoinRequest,
    JoinAccept,
    compute_join_mic,
    aes_encrypt,
)  # noqa: E402


def test_otaa_join_procedure():
    node = Node(1, 0.0, 0.0, 7, 14.0, channel=Channel(), activated=False, appkey=bytes(16))
    gw = Gateway(1, 0.0, 0.0)
    js = JoinServer(net_id=0)
    js.register(node.join_eui, node.dev_eui, node.appkey)
    ns = NetworkServer(join_server=js)
    ns.gateways = [gw]
    ns.nodes = [node]

    req = node.prepare_uplink(b"")
    assert isinstance(req, JoinRequest)
    assert compute_join_mic(node.appkey, req.to_bytes()) == req.mic

    ns.receive(event_id=1, node_id=node.id, gateway_id=gw.id, frame=req)
    frame = gw.pop_downlink(node.id)
    assert isinstance(frame, JoinAccept)
    assert compute_join_mic(node.appkey, frame.to_bytes()) == frame.mic
    assert aes_encrypt(node.appkey, frame.encrypted)[:10] == frame.to_bytes()

    node.handle_downlink(frame)
    assert node.activated
    assert len(node.nwkskey) == 16
    assert len(node.appskey) == 16
    assert node.downlink_pending == 0
    assert js.get_session_keys(node.join_eui, node.dev_eui) == (
        node.nwkskey,
        node.appskey,
    )
