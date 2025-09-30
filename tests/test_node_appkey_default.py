import pytest
from loraflexsim.launcher.node import Node


def test_node_appkey_defaults_to_16_zero_bytes_when_none():
    n = Node(0, 0, 0, 7, 14, appkey=None)
    assert n.appkey == bytes(16)


def test_node_appkey_allows_explicit_empty_bytes():
    n = Node(0, 0, 0, 7, 14, appkey=b"")
    assert n.appkey == b""
