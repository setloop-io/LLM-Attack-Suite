"""MITM tee proxy: bidirectional capture integrity, framing, caps."""

import json
import struct

import pytest

from attacker.mitm import (MitmProxy, _EchoServer, _echo_roundtrip,
                           _read_journal)

SCHEMA = {"ts", "bytes", "dir", "session_id"}


@pytest.fixture
def echo():
    with _EchoServer() as srv:
        yield srv


def make_proxy(tmp_path, echo, name="capture.jsonl", **kw):
    return MitmProxy("127.0.0.1", 0, "127.0.0.1", echo.port,
                     str(tmp_path / name), **kw)


def test_bidirectional_tee_integrity(tmp_path, echo):
    payloads = [b"first client payload", b"y" * 4096, b"third"]
    with make_proxy(tmp_path, echo) as proxy:
        assert isinstance(proxy.port, int) and proxy.port > 0
        for p in payloads:
            assert _echo_roundtrip(proxy.port, p) == p
        stats = proxy.stats()
    journal = str(tmp_path / "capture.jsonl")
    records = _read_journal(journal)
    assert records and all(set(r) == SCHEMA for r in records)
    assert all(isinstance(r["ts"], float) and isinstance(r["bytes"], int)
               and r["dir"] in ("c2s", "s2c")
               and isinstance(r["session_id"], str) for r in records)
    sent = sum(len(p) for p in payloads)
    for direction in ("c2s", "s2c"):
        assert sum(r["bytes"] for r in records if r["dir"] == direction) \
            == sent
    expected = b"".join(payloads)
    assert open(journal + ".c2s", "rb").read() == expected
    assert open(journal + ".s2c", "rb").read() == expected
    per_key = {}
    for r in records:
        per_key.setdefault((r["session_id"], r["dir"]), []).append(r["ts"])
    assert all(ts == sorted(ts) for ts in per_key.values())
    assert len({r["session_id"] for r in records}) == len(payloads)
    assert stats["records"] == len(records)
    assert stats["bytes_c2s"] == stats["bytes_s2c"] == sent
    assert stats["sessions"] == len(payloads)
    assert stats["capped"] is False and stats["frame"] == "raw"


def _wait_for(predicate, timeout=3.0):
    """Poll `predicate` until it holds (EOF/flush propagation across the
    relay threads is asynchronous); False on timeout."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_len32_frames_split_on_boundaries(tmp_path, echo):
    m1, m2 = b"frame-one", b"z" * 300
    framed = struct.pack(">I", len(m1)) + m1 + struct.pack(">I", len(m2)) + m2
    tail = b"\x00\x00\x00\x09ab"
    journal = str(tmp_path / "capture.jsonl")
    with make_proxy(tmp_path, echo, frame="len32") as proxy:
        assert _echo_roundtrip(proxy.port, framed + tail) == framed + tail
        assert _wait_for(lambda: len(_read_journal(journal)) == 6)
    records = _read_journal(journal)
    for direction in ("c2s", "s2c"):
        sizes = [r["bytes"] for r in records if r["dir"] == direction]
        assert sizes == [4 + len(m1), 4 + len(m2), len(tail)]


def test_max_bytes_cap_drops_records_but_never_relayed_bytes(tmp_path, echo):
    payload = b"cap-me " * 200
    with make_proxy(tmp_path, echo, max_bytes=100) as proxy:
        assert _echo_roundtrip(proxy.port, payload) == payload
        stats = proxy.stats()
    records = _read_journal(str(tmp_path / "capture.jsonl"))
    assert stats["capped"] is True
    assert sum(r["bytes"] for r in records) <= 100


def test_stop_is_idempotent_and_port_survives(tmp_path, echo):
    proxy = make_proxy(tmp_path, echo)
    port = proxy.port
    proxy.start()
    assert _echo_roundtrip(port, b"ping") == b"ping"
    proxy.stop()
    proxy.stop()
    assert proxy.port == port
    stats = proxy.stats()
    assert stats["records"] > 0 and stats["duration_s"] >= 0


def test_zero_traffic_capture_writes_an_empty_journal(tmp_path, echo):
    with make_proxy(tmp_path, echo) as proxy:
        pass
    assert _read_journal(str(tmp_path / "capture.jsonl")) == []
    assert proxy.stats()["records"] == 0


def test_bad_frame_is_rejected(tmp_path, echo):
    with pytest.raises(ValueError, match="frame"):
        make_proxy(tmp_path, echo, frame="len64")


def test_self_test_passes():
    from attacker import mitm
    assert mitm.self_test() == 0
