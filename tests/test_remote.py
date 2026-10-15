"""Fleet config, probing, and remote orchestration (attacker.remote)."""

import json
import socket
import threading
import time
import types

import pytest

from attacker import remote


@pytest.fixture
def loopback_listener():
    """A bound localhost socket with a drain thread (accept+close) so
    probe connections never pile up in the listen backlog."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    srv.settimeout(0.2)
    draining = threading.Event()

    def _drain():
        while not draining.is_set():
            try:
                conn, _ = srv.accept()
                conn.close()
            except socket.timeout:
                continue
            except OSError:
                break

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    yield srv
    draining.set()
    srv.close()
    t.join(timeout=2)


DEAD = {"name": "dead", "host": "192.0.2.1", "role": "cpu"}


def test_shipped_config_parses_with_the_five_fleet_nodes():
    cfg = remote.load_config()
    nodes = {n["name"]: n for n in remote.enabled_nodes(cfg)}
    assert sorted(nodes) == ["koula", "sasa", "soula", "toula", "voula"]
    assert nodes["toula"]["role"] == "gpu" and nodes["koula"]["role"] == "gpu"
    assert all(nodes[n]["role"] == "cpu" for n in ("soula", "voula", "sasa"))
    assert all(n["enabled"] and n["port"] == 22 for n in nodes.values())
    localhost = remote.get_node(cfg, "localhost")
    assert localhost["enabled"] is False


def test_config_validation_is_descriptive(tmp_path):
    base = [{"name": "a", "host": "h", "role": "gpu"}]
    for bad, frag in (
            ([{"name": "x", "host": "h", "role": "tpu"}], "role"),
            ([{"name": "x", "role": "gpu"}], "host"),
            ([{"name": "x", "host": "h", "role": "gpu", "weight": 1}],
             "unknown keys"),
            (base + [{"name": "a", "host": "h2", "role": "cpu"}],
             "duplicate")):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"nodes": bad}))
        with pytest.raises(ValueError, match=frag.split()[0]):
            remote.load_config(str(p))
    with pytest.raises(KeyError, match="no-such"):
        remote.get_node(remote.load_config(), "no-such")


def test_probe_unreachable_node_is_fast_and_clean():
    t0 = time.monotonic()
    st = remote.probe_node(DEAD, timeout=1)
    dt = time.monotonic() - t0
    assert not st.reachable and not st.ssh_ok
    assert st.error
    assert st.latency_ms is None
    assert dt < 5.0
    d = st.to_dict()
    assert d["node"] == "dead" and d["reachable"] is False
    assert "error" in d


def test_probe_localhost_socket(loopback_listener):
    port = loopback_listener.getsockname()[1]
    st = remote.probe_node({"name": "loop", "host": "127.0.0.1",
                            "port": port}, timeout=1, ssh=False)
    assert st.reachable and st.latency_ms is not None
    assert not st.ssh_ok
    assert "error" not in st.to_dict()


def test_probe_fleet_parallel_mixed(loopback_listener):
    port = loopback_listener.getsockname()[1]
    nodes = [DEAD, {"name": "loop", "host": "127.0.0.1", "port": port,
                    "role": "cpu"},
             {"name": "off", "host": "192.0.2.9", "role": "gpu",
              "enabled": False}]
    t0 = time.monotonic()
    report = remote.probe_fleet({"nodes": nodes}, timeout=1)
    dt = time.monotonic() - t0
    assert report["summary"] == {"probed": 2, "reachable": 1, "ssh_ok": 0,
                                 "unreachable": 1}
    assert report["nodes"]["dead"]["reachable"] is False
    assert report["nodes"]["loop"]["reachable"] is True
    assert "off" not in report["nodes"]
    assert dt < 5.0


def test_fetch_captures_unreachable_fails_fast(tmp_path):
    t0 = time.monotonic()
    ok, msg, files = remote.fetch_captures(DEAD, "captures",
                                           str(tmp_path / "dead"),
                                           timeout=10, probe_timeout=1)
    dt = time.monotonic() - t0
    assert not ok and files == 0 and "unreachable" in msg
    assert dt < 5.0
    assert not (tmp_path / "dead").exists()


def test_fetch_captures_success_counts_files(tmp_path, monkeypatch):
    """Reachable node: rsync/scp invocation is faked; files created by the
    'transfer' must be counted (and the pre-probe short-circuited)."""
    monkeypatch.setattr(remote, "probe_node",
                        lambda node, timeout=3, ssh=True: remote.NodeStatus(
                            node["name"], node["host"], reachable=True,
                            ssh_ok=True, latency_ms=1.0))

    def fake_run(cmd, capture_output, text, timeout):
        dest = cmd[-1]
        import os
        os.makedirs(dest, exist_ok=True)
        (open(os.path.join(dest, "wire_0000.json"), "w")).write("{}")
        (open(os.path.join(dest, "wire_0000.pt"), "w")).write("x")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    ok, msg, files = remote.fetch_captures(
        {"name": "toula", "host": "toula", "role": "gpu"}, "captures",
        str(tmp_path / "out"), timeout=10)
    assert ok and files == 2 and "toula" in msg


def test_run_remote_attack_failure_returns_rc(loopback_listener):
    """ssh launches against the non-ssh loopback peer and fails fast
    (handshake EOF), so rc != 0 without a ConnectTimeout stall."""
    port = loopback_listener.getsockname()[1]
    t0 = time.monotonic()
    rc, tail = remote.run_remote_attack(
        {"name": "loop", "host": "127.0.0.1", "port": port},
        ["--self-test"], timeout=5)
    dt = time.monotonic() - t0
    assert rc != 0 and isinstance(tail, str)
    assert dt < 10.0


def test_self_test_passes():
    assert remote.self_test() == 0
