"""remote-fetch attack driver: fleet probe + capture pull orchestration."""

import json
import os
import time

import pytest

from attacker import remote
from attacker.attacks import ATTACKS, load, register_attack, validate_contract
from attacker.attacks import remote_fetch
from attacker.evidence import verify_evidence


@pytest.fixture
def registered():
    before = set(ATTACKS)
    if "remote-fetch" not in ATTACKS:
        register_attack("remote-fetch", remote_fetch)
    yield remote_fetch
    for name in set(ATTACKS) - before:
        ATTACKS.pop(name, None)


@pytest.fixture
def dead_config(tmp_path):
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps({"nodes": [
        {"name": "dead-a", "host": "192.0.2.1", "role": "gpu"},
        {"name": "dead-b", "host": "192.0.2.2", "role": "cpu",
         "enabled": True},
        {"name": "off", "host": "192.0.2.3", "role": "cpu",
         "enabled": False}]}))
    return str(path)


def parse(*argv):
    return remote_fetch.build_parser().parse_args(list(argv))


def test_contract_and_registration(registered):
    assert validate_contract(remote_fetch) is remote_fetch
    assert remote_fetch.MODES == ("training", "inference")
    assert remote_fetch.REQUIRES_LABELS == "none"
    assert ATTACKS["remote-fetch"] is remote_fetch or \
        load("remote-fetch") is remote_fetch


def test_help_works_torchless():
    with pytest.raises(SystemExit) as exc:
        parse("--help")
    assert exc.value.code == 0


def test_unreachable_fleet_is_a_clean_result(tmp_path, dead_config):
    output = str(tmp_path / "fetch.json")
    args = parse("--nodes-config", dead_config, "--probe-timeout", "1",
                 "--timeout", "5", "--output-dir", str(tmp_path / "pulled"),
                 "--output", output)
    t0 = time.monotonic()
    rc = remote_fetch.run(args)
    dt = time.monotonic() - t0
    assert rc == 0
    assert dt < 8.0
    art = json.load(open(output))
    summary = art["summary"][0]
    assert summary["nodes_probed"] == 2
    assert summary["nodes_reachable"] == 0
    assert summary["nodes_fetched"] == 0
    assert summary["all_unreachable"] is True
    assert summary["files_total"] == 0 and summary["bytes_total"] == 0
    for cell in art["results"]:
        assert cell["reachable"] is False and cell["fetched"] is False
        assert cell["files"] == 0 and cell["bytes"] == 0
        assert cell["experiment"] == "remote_fetch"
        assert "probe_error" in cell
    journal = (open(output + ".jsonl").read().strip().splitlines())
    assert len(journal) == 2


def test_simulated_fetch_preserves_node_layout(tmp_path, dead_config,
                                               monkeypatch):
    """Reachable node: probe/fetch faked; the layout must be
    <output-dir>/<node>/<remote_dir_basename>/ and bytes counted."""
    monkeypatch.setattr(remote, "probe_fleet", lambda config, timeout=3: {
        "nodes": {"dead-a": {"node": "dead-a", "host": "192.0.2.1",
                             "port": 22, "reachable": True, "ssh_ok": True,
                             "latency_ms": 1.0},
                  "dead-b": {"node": "dead-b", "host": "192.0.2.2",
                             "port": 22, "reachable": False,
                             "ssh_ok": False, "latency_ms": None,
                             "error": "tcp: timed out"}},
        "summary": {"probed": 2, "reachable": 1, "ssh_ok": 1,
                    "unreachable": 1}})

    def fake_fetch(node, remote_dir, local_dir, timeout=120,
                   probe_timeout=3):
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, "wire_0000.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(local_dir, "wire_0000.pt"), "wb") as f:
            f.write(b"\x00" * 10)
        return True, f"[{node['name']}] fetched 2 file(s)", 2

    monkeypatch.setattr(remote, "fetch_captures", fake_fetch)
    output = str(tmp_path / "fetch.json")
    out_dir = str(tmp_path / "pulled")
    args = parse("--nodes-config", dead_config, "--remote-dir",
                 "runs/er1/captures", "--output-dir", out_dir,
                 "--output", output, "--evidence")
    assert remote_fetch.run(args) == 0
    layout = os.path.join(out_dir, "dead-a", "captures")
    assert sorted(os.listdir(layout)) == ["wire_0000.json", "wire_0000.pt"]
    art = json.load(open(output))
    summary = art["summary"][0]
    assert summary["nodes_fetched"] == 1
    assert summary["files_total"] == 2
    assert summary["bytes_total"] == 12
    assert summary["all_unreachable"] is False
    cells = {c["node"]: c for c in art["results"]}
    assert cells["dead-a"]["fetched"] is True
    assert cells["dead-a"]["files"] == 2 and cells["dead-a"]["bytes"] == 12
    assert cells["dead-b"]["fetched"] is False
    ok, problems = verify_evidence(output + ".evidence.json")
    assert ok and problems == []


def test_hard_misuse_exits_nonzero(tmp_path, dead_config):
    assert remote_fetch.run(parse("--nodes-config", dead_config,
                                  "--node", "no-such-node")) == 2
    assert remote_fetch.run(parse("--nodes-config", dead_config,
                                  "--node", "off")) == 2
    assert remote_fetch.run(parse("--nodes-config",
                                  str(tmp_path / "missing.json"))) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert remote_fetch.run(parse("--nodes-config", str(bad))) == 2


def test_self_test_passes():
    assert remote_fetch.self_test() == 0
