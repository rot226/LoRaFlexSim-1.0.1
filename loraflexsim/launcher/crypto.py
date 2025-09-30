"""Minimal AES and CMAC algorithms in pure Python.

This module provides AES-128 ECB encryption and the CMAC
algorithm used by LoRaWAN for MIC computation. The code is
intentionally straightforward and avoids any external
dependencies so it can run on restricted environments.
"""

from __future__ import annotations

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Helper functions for operations in GF(2^8)
# ---------------------------------------------------------------------------

def _xtime(a: int) -> int:
    a <<= 1
    if a & 0x100:
        a ^= 0x11B
    return a & 0xFF


def _mul(a: int, b: int) -> int:
    res = 0
    for _ in range(8):
        if b & 1:
            res ^= a
        a = _xtime(a)
        b >>= 1
    return res & 0xFF


def _gf_inv(a: int) -> int:
    if a == 0:
        return 0
    # Compute a^(254) in GF(2^8)
    r = 1
    for _ in range(254):
        r = _mul(r, a)
    return r


def _generate_sbox() -> Tuple[list[int], list[int]]:
    sbox = [0] * 256
    inv = [0] * 256
    for i in range(256):
        x = _gf_inv(i)
        y = x
        for _ in range(1, 5):
            x = ((x << 1) | (x >> 7)) & 0xFF
            y ^= x
        y ^= 0x63
        sbox[i] = y
        inv[y] = i
    return sbox, inv


_SBOX, _INV_SBOX = _generate_sbox()

# Round constants
_RCON = [
    0x00,
    0x01,
    0x02,
    0x04,
    0x08,
    0x10,
    0x20,
    0x40,
    0x80,
    0x1B,
    0x36,
]

# ---------------------------------------------------------------------------
# AES key schedule and encryption
# ---------------------------------------------------------------------------

def _sub_word(word: List[int]) -> List[int]:
    return [_SBOX[b] for b in word]


def _rot_word(word: List[int]) -> List[int]:
    return word[1:] + word[:1]


def _key_schedule(key: bytes) -> List[List[int]]:
    assert len(key) == 16
    words = [list(key[i : i + 4]) for i in range(0, 16, 4)]
    for i in range(4, 44):
        temp = words[i - 1]
        if i % 4 == 0:
            temp = _sub_word(_rot_word(temp))
            temp[0] ^= _RCON[i // 4]
        words.append([a ^ b for a, b in zip(words[i - 4], temp)])
    return [sum(words[i : i + 4], []) for i in range(0, 44, 4)]


def _add_round_key(state: List[int], round_key: List[int]) -> None:
    for i in range(16):
        state[i] ^= round_key[i]


def _sub_bytes(state: List[int]) -> None:
    for i in range(16):
        state[i] = _SBOX[state[i]]


def _shift_rows(state: List[int]) -> None:
    state[1], state[5], state[9], state[13] = state[5], state[9], state[13], state[1]
    state[2], state[6], state[10], state[14] = state[10], state[14], state[2], state[6]
    state[3], state[7], state[11], state[15] = state[15], state[3], state[7], state[11]


def _mix_columns(state: List[int]) -> None:
    for c in range(4):
        a0 = state[4 * c]
        a1 = state[4 * c + 1]
        a2 = state[4 * c + 2]
        a3 = state[4 * c + 3]
        state[4 * c] = _mul(a0, 2) ^ _mul(a1, 3) ^ a2 ^ a3
        state[4 * c + 1] = a0 ^ _mul(a1, 2) ^ _mul(a2, 3) ^ a3
        state[4 * c + 2] = a0 ^ a1 ^ _mul(a2, 2) ^ _mul(a3, 3)
        state[4 * c + 3] = _mul(a0, 3) ^ a1 ^ a2 ^ _mul(a3, 2)


def _aes_encrypt_block(block: bytes, round_keys: List[List[int]]) -> bytes:
    assert len(block) == 16
    state = list(block)
    _add_round_key(state, round_keys[0])
    for rk in round_keys[1:-1]:
        _sub_bytes(state)
        _shift_rows(state)
        _mix_columns(state)
        _add_round_key(state, rk)
    _sub_bytes(state)
    _shift_rows(state)
    _add_round_key(state, round_keys[-1])
    return bytes(state)


def aes_encrypt(key: bytes, data: bytes) -> bytes:
    """Encrypt ``data`` (multiple of 16 bytes) using AES-128 ECB."""
    assert len(data) % 16 == 0
    round_keys = _key_schedule(key)
    out = bytearray()
    for i in range(0, len(data), 16):
        out += _aes_encrypt_block(data[i : i + 16], round_keys)
    return bytes(out)


def _inv_sub_bytes(state: List[int]) -> None:
    for i in range(16):
        state[i] = _INV_SBOX[state[i]]


def _inv_shift_rows(state: List[int]) -> None:
    row1 = [state[1], state[5], state[9], state[13]]
    row2 = [state[2], state[6], state[10], state[14]]
    row3 = [state[3], state[7], state[11], state[15]]
    row1 = row1[-1:] + row1[:-1]
    row2 = row2[-2:] + row2[:-2]
    row3 = row3[-3:] + row3[:-3]
    state[1], state[5], state[9], state[13] = row1
    state[2], state[6], state[10], state[14] = row2
    state[3], state[7], state[11], state[15] = row3


def _inv_mix_columns(state: List[int]) -> None:
    for c in range(4):
        a0 = state[4 * c]
        a1 = state[4 * c + 1]
        a2 = state[4 * c + 2]
        a3 = state[4 * c + 3]
        state[4 * c] = (
            _mul(a0, 14) ^ _mul(a1, 11) ^ _mul(a2, 13) ^ _mul(a3, 9)
        )
        state[4 * c + 1] = (
            _mul(a0, 9) ^ _mul(a1, 14) ^ _mul(a2, 11) ^ _mul(a3, 13)
        )
        state[4 * c + 2] = (
            _mul(a0, 13) ^ _mul(a1, 9) ^ _mul(a2, 14) ^ _mul(a3, 11)
        )
        state[4 * c + 3] = (
            _mul(a0, 11) ^ _mul(a1, 13) ^ _mul(a2, 9) ^ _mul(a3, 14)
        )


def _aes_decrypt_block(block: bytes, round_keys: List[List[int]]) -> bytes:
    assert len(block) == 16
    state = list(block)
    _add_round_key(state, round_keys[-1])
    for rk in reversed(round_keys[1:-1]):
        _inv_shift_rows(state)
        _inv_sub_bytes(state)
        _add_round_key(state, rk)
        _inv_mix_columns(state)
    _inv_shift_rows(state)
    _inv_sub_bytes(state)
    _add_round_key(state, round_keys[0])
    return bytes(state)


def aes_decrypt(key: bytes, data: bytes) -> bytes:
    """Decrypt ``data`` (multiple of 16 bytes) using AES-128 ECB."""
    assert len(data) % 16 == 0
    round_keys = _key_schedule(key)
    out = bytearray()
    for i in range(0, len(data), 16):
        out += _aes_decrypt_block(data[i : i + 16], round_keys)
    return bytes(out)


# ---------------------------------------------------------------------------
# CMAC (RFC 4493)
# ---------------------------------------------------------------------------

_RB = 0x87


def _left_shift(b: bytes) -> bytes:
    n = int.from_bytes(b, "big") << 1
    n &= (1 << (len(b) * 8)) - 1
    return n.to_bytes(len(b), "big")


def _generate_subkeys(key: bytes) -> Tuple[bytes, bytes]:
    L = aes_encrypt(key, bytes(16))
    if L[0] & 0x80:
        k1_buf = bytearray(_left_shift(L))
        k1_buf[-1] ^= _RB
        k1: bytes = bytes(k1_buf)
    else:
        k1 = _left_shift(L)

    if k1[0] & 0x80:
        k2_buf = bytearray(_left_shift(k1))
        k2_buf[-1] ^= _RB
        k2: bytes = bytes(k2_buf)
    else:
        k2 = _left_shift(k1)

    return k1, k2


def cmac(key: bytes, msg: bytes) -> bytes:
    K1, K2 = _generate_subkeys(key)
    n = (len(msg) + 15) // 16
    if n == 0:
        n = 1
    complete = len(msg) % 16 == 0
    last = msg[(n - 1) * 16 :]
    if complete:
        last = bytes(x ^ y for x, y in zip(last, K1))
    else:
        padded = last + b"\x80" + bytes(15 - len(last))
        last = bytes(x ^ y for x, y in zip(padded, K2))
    C = bytes(16)
    round_keys = _key_schedule(key)
    for i in range(n - 1):
        blk = msg[i * 16 : i * 16 + 16]
        C = _aes_encrypt_block(bytes(x ^ y for x, y in zip(C, blk)), round_keys)
    C = _aes_encrypt_block(bytes(x ^ y for x, y in zip(C, last)), round_keys)
    return C


def encrypt_payload(
    app_skey: bytes,
    devaddr: int,
    fcnt: int,
    direction: int,
    payload: bytes,
) -> bytes:
    """Encrypt ``payload`` using the LoRaWAN payload encryption scheme."""
    blocks = []
    for i in range(1, (len(payload) + 15) // 16 + 1):
        a = (
            bytes([0x01, 0x00, 0x00, 0x00, 0x00])
            + bytes([direction & 0x01])
            + devaddr.to_bytes(4, "little")
            + fcnt.to_bytes(4, "little")
            + bytes([0x00, i])
        )
        s = aes_encrypt(app_skey, a)
        start = (i - 1) * 16
        block = payload[start : start + 16]
        blocks.append(bytes(x ^ y for x, y in zip(block, s)))
    return b"".join(blocks)[: len(payload)]


def decrypt_payload(
    app_skey: bytes,
    devaddr: int,
    fcnt: int,
    direction: int,
    payload: bytes,
) -> bytes:
    """Decrypt ``payload`` using the LoRaWAN payload encryption scheme."""
    return encrypt_payload(app_skey, devaddr, fcnt, direction, payload)

