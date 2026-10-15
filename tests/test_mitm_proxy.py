"""mitm-proxy attack driver: long-running capture cell over a live relay."""

import json
import socket
import threading
import time

import pytest

from attacker.attacks import ATTACKS, load, register_attack, validate_contract
from attacker.attacks import mitm_proxy
from attacker.evidence import verify_evidence
from attacker.mitm import _EchoServer, _echo_roundtrip, _read_journal

SCHEMA = {"ts", "bytes", "dir", "session_id"}


@pytest.fixture
def registered():
    before = set(ATTACKS)
    if "mitm-proxy" not in ATTACKS:
        register_attack("mitm-proxy", mitm_proxy)
    yield mitm_proxy
    for name in set(ATTACKS) - before:
        ATTACKS.pop(name, None)


@pytest.fixture
def echo():
    with _EchoServer() as srv:
        yield srv


def parse(*argv):
    return mitm_proxy.build_parser().parse_args(list(argv))


def free_port():
    """Reserve an ephemeral listen port for a driver run (released before
    the proxy binds it)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def drive(args, worker):
    """Run mitm_proxy.run in a thread while `worker(listen_port)` drives
    traffic; returns the driver's exit code."""
    box = {}
    t = threading.Thread(target=lambda: box.setdefault(
        "rc", mitm_proxy.run(args)), daemon=True)
    t.start()
    time.sleep(0.15)
    worker(int(args.listen))
    t.join(timeout=15)
    assert not t.is_alive(), "mitm-proxy driver did not stop after duration"
    return box.get("rc")


def test_contract_and_registration(registered):
    assert validate_contract(mitm_proxy) is mitm_proxy
    assert mitm_proxy.MODES == ("training", "inference")
    assert mitm_proxy.REQUIRES_LABELS is False
    assert ATTACKS["mitm-proxy"] is mitm_proxy or \
        load("mitm-proxy") is mitm_proxy


def test_help_works_torchless():
    with pytest.raises(SystemExit) as exc:
        parse("--help")
    assert exc.value.code == 0


def assert_echoed(port, payload):
    assert _echo_roundtrip(port, payload) == payload


def test_end_to_end_capture_with_evidence(tmp_path, echo):
    listen = free_port()
    journal = str(tmp_path / "capture.jsonl")
    output = str(tmp_path / "mitm.json")
    args = parse("--target", f"127.0.0.1:{echo.port}",
                 "--listen", str(listen), "--journal", journal,
                 "--duration", "0.6", "--output", output, "--evidence")
    payloads = [b"client payload one", b"q" * 4096, b"tail"]
    rc = drive(args, lambda port: [assert_echoed(port, p)
                                   for p in payloads])
    assert rc == 0
    records = _read_journal(journal)
    assert records and all(set(r) == SCHEMA for r in records)
    sent = sum(len(p) for p in payloads)
    art = json.load(open(output))
    cell = art["results"][0]
    for key in ("records", "bytes_c2s", "bytes_s2c", "sessions",
                "duration_s"):
        assert key in cell
    assert cell["bytes_c2s"] == sent and cell["bytes_s2c"] == sent
    assert cell["sessions"] == len(payloads)
    assert cell["records"] == len(records)
    assert cell["duration_s"] > 0 and cell["capped"] is False
    assert cell["listen_port"] == listen
    summary = art["summary"][0]
    assert summary["journal_path"] == journal
    assert open(journal + ".c2s", "rb").read() == b"".join(payloads)
    assert open(journal + ".s2c", "rb").read() == b"".join(payloads)
    ok, problems = verify_evidence(output + ".evidence.json")
    assert ok and problems == []


def assert_echoed(port, payload):
    assert _echo_roundtrip(port, payload) == payload


def test_max_bytes_cap_through_the_cli(tmp_path, echo):
    listen = free_port()
    journal = str(tmp_path / "capture.jsonl")
    args = parse("--target", f"127.0.0.1:{echo.port}",
                 "--listen", str(listen), "--journal", journal,
                 "--max-bytes", "64", "--duration", "0.5",
                 "--output", str(tmp_path / "mitm.json"))
    payload = b"overflow " * 100
    rc = drive(args, lambda port: assert_echoed(port, payload))
    assert rc == 0
    records = _read_journal(journal)
    assert sum(r["bytes"] for r in records) <= 64
    art = json.load(open(str(tmp_path / "mitm.json")))
    assert art["summary"][0]["capped"] is True
    assert art["results"][0]["bytes_c2s"] <= 64


def test_hard_misuse_exits_nonzero(tmp_path, echo):
    assert mitm_proxy.run(parse("--target", "no-port")) == 2
    assert mitm_proxy.run(parse("--target", "host:abc")) == 2
    assert mitm_proxy.run(parse("--target", "host:99999")) == 2
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    try:
        args = parse("--target", f"127.0.0.1:{echo.port}",
                     "--listen", str(blocker.getsockname()[1]),
                     "--journal", str(tmp_path / "j.jsonl"),
                     "--duration", "0.1")
        assert mitm_proxy.run(args) == 2
    finally:
        blocker.close()


def test_self_test_passes():
    assert mitm_proxy.self_test() == 0
