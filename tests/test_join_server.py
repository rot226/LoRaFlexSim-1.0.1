import pytest

from loraflexsim.launcher.join_server import JoinServer
from loraflexsim.launcher.lorawan import (
    JoinRequest,
    compute_join_mic,
    decrypt_join_accept,
    derive_session_keys,
)


def _register_device(server: JoinServer) -> tuple[int, int, bytes, JoinRequest]:
    join_eui = 0x70B3D57ED0000000
    dev_eui = 0x0004A30B001C0530
    app_key = bytes.fromhex("8A2C1F6E9D4475B0FF1133557799BBAA")
    server.register(join_eui, dev_eui, app_key)

    request = JoinRequest(join_eui, dev_eui, dev_nonce=0x2345)
    request.mic = compute_join_mic(app_key, request.to_bytes())
    return join_eui, dev_eui, app_key, request


def test_otaa_join_flow_encrypts_accept_and_derives_keys():
    server = JoinServer(net_id=0x123456)
    join_eui, dev_eui, app_key, request = _register_device(server)

    accept, nwk_skey, app_skey = server.handle_join(request)

    assert accept.encrypted is not None, "Le JoinAccept doit être chiffré"
    assert accept.mic == compute_join_mic(app_key, accept.to_bytes())

    decrypted, mic = decrypt_join_accept(app_key, accept.encrypted, len(accept.to_bytes()))
    assert mic == accept.mic
    assert decrypted.app_nonce == accept.app_nonce
    assert decrypted.net_id == accept.net_id
    assert decrypted.dev_addr == accept.dev_addr

    expected_nwk, expected_app = derive_session_keys(
        app_key, request.dev_nonce, accept.app_nonce, server.net_id
    )
    assert (nwk_skey, app_skey) == (expected_nwk, expected_app)
    assert server.get_session_keys(join_eui, dev_eui) == (nwk_skey, app_skey)
    assert accept.dev_addr != 0


def test_otaa_join_rejects_invalid_mic():
    server = JoinServer(net_id=0x001122)
    join_eui, dev_eui, app_key, request = _register_device(server)

    request.mic = b"\x00\x00\x00\x00"
    with pytest.raises(ValueError):
        server.handle_join(request)

    # Le serveur ne doit pas dériver de clés tant que le MIC est invalide.
    assert server.get_session_keys(join_eui, dev_eui) is None
