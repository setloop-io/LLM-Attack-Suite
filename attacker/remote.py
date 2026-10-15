#!/usr/bin/env python3
"""Fleet config + remote orchestration core (pure stdlib, torch-free).

Deployment shape (b): the attacker orchestrates/pulls wire captures from
remote fleet nodes over SSH. Everything here degrades gracefully: the fleet
may be unreachable, ssh may be missing, DNS may not resolve — every public
function below probes with short timeouts, returns a status instead of
raising, and never hangs (every subprocess call carries a hard timeout).

Node config: attacker/nodes.json (JSON rather than TOML: equally
stdlib-only on the >=3.12 floor, and node entries are edited by hand often
enough that trailing-comma tolerance wins). Schema "llm_attacker.nodes.v1":

    {"schema": ..., "nodes": [{"name", "host", "role": "gpu"|"cpu",
                               "user"?, "port"?, "tags"?, "enabled"?}]}

Usage:
    from attacker import remote
    cfg = remote.load_config()
    report = remote.probe_fleet(cfg, timeout=3)
    ok, msg, nfiles = remote.fetch_captures(node, "captures", "out/toula/")
    rc, tail = remote.run_remote_attack(node, ["--self-test"])
"""

import concurrent.futures
import json
import os
import shlex
import shutil
import socket
import subprocess
import time

SCHEMA = "llm_attacker.nodes.v1"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "nodes.json")
VALID_ROLES = ("gpu", "cpu")
_OPTIONAL_KEYS = {"user": str, "tags": list, "enabled": bool, "port": int,
                  "note": str}


class NodeStatus:
    """Outcome of probing one fleet node. Construction never raises;
    `error` is set (and omitted from to_dict when absent) on failure."""

    def __init__(self, name, host, port=22, reachable=False, ssh_ok=False,
                 latency_ms=None, error=None):
        self.name = name
        self.host = host
        self.port = port
        self.reachable = reachable
        self.ssh_ok = ssh_ok
        self.latency_ms = latency_ms
        self.error = error

    def to_dict(self):
        d = {"node": self.name, "host": self.host, "port": self.port,
             "reachable": self.reachable, "ssh_ok": self.ssh_ok,
             "latency_ms": self.latency_ms}
        if self.error is not None:
            d["error"] = self.error
        return d


def _validate_node(node, seen):
    """Normalize one node dict; raise ValueError with a descriptive message
    on schema drift (config is attacker-authored: drift is a hard error,
    unlike probing which never raises)."""
    if not isinstance(node, dict):
        raise ValueError(f"node entry must be a JSON object, got {node!r}")
    label = node.get("name") or node.get("host") or repr(node)
    for key in ("name", "host", "role"):
        if not isinstance(node.get(key), str) or not node[key]:
            raise ValueError(f"node {label!r}: {key!r} must be a non-empty "
                             f"str")
    if node["role"] not in VALID_ROLES:
        raise ValueError(f"node {node['name']!r}: role {node['role']!r} not "
                         f"in {VALID_ROLES}")
    if node["name"] in seen:
        raise ValueError(f"duplicate node name {node['name']!r}")
    seen.add(node["name"])
    unknown = set(node) - {"name", "host", "role"} - set(_OPTIONAL_KEYS)
    if unknown:
        raise ValueError(f"node {node['name']!r}: unknown keys "
                         f"{sorted(unknown)}")
    out = {"name": node["name"], "host": node["host"], "role": node["role"],
           "user": node.get("user"), "tags": list(node.get("tags") or []),
           "enabled": bool(node.get("enabled", True)),
           "port": int(node.get("port", 22))}
    if out["user"] is not None and not isinstance(out["user"], str):
        raise ValueError(f"node {node['name']!r}: 'user' must be a str")
    if any(not isinstance(t, str) for t in out["tags"]):
        raise ValueError(f"node {node['name']!r}: 'tags' must be strs")
    if not (1 <= out["port"] <= 65535):
        raise ValueError(f"node {node['name']!r}: 'port' out of range")
    if "note" in node:
        out["note"] = node["note"]
    return out


def _validate_config(data):
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        raise ValueError("nodes config must be a JSON object with a "
                         "'nodes' list")
    seen = set()
    nodes = [_validate_node(n, seen) for n in data["nodes"]]
    return {"schema": data.get("schema", SCHEMA), "nodes": nodes}


def load_config(path=None):
    """Parse + validate a nodes config (default: the shipped nodes.json).
    Raises ValueError/OSError with a descriptive message on drift."""
    path = path or DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return _validate_config(json.load(f))


def enabled_nodes(config):
    return [n for n in config["nodes"] if n.get("enabled", True)]


def get_node(config, name):
    for n in config["nodes"]:
        if n["name"] == name:
            return n
    raise KeyError(f"unknown node {name!r}; configured: "
                   f"{', '.join(n['name'] for n in config['nodes']) or '<none>'}")


def _ssh_target(node):
    return f"{node['user']}@{node['host']}" if node.get("user") \
        else node["host"]


def _ssh_opts(connect_timeout):
    """BatchMode kills every prompt (passwords AND host-key confirms), so
    ssh can never block on stdin; unknown host keys fail cleanly with
    'Host key verification failed' instead of mutating known_hosts."""
    return ["-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={max(1, int(connect_timeout))}"]


def probe_node(node, timeout=3, ssh=True):
    """TCP reachability to the node's ssh port + (optionally) an
    `ssh ... true` handshake. Returns a NodeStatus; never raises."""
    name = node.get("name", node.get("host", "?"))
    host, port = node.get("host"), int(node.get("port", 22))
    status = NodeStatus(name, host, port)
    try:
        t0 = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
            status.reachable = True
            status.latency_ms = round(1000.0 * (time.monotonic() - t0), 3)
        except Exception as e:
            status.error = f"tcp {host}:{port}: {e}"
            return status
        if not ssh:
            return status
        if shutil.which("ssh") is None:
            status.error = "ssh not found on PATH"
            return status
        try:
            proc = subprocess.run(
                ["ssh"] + _ssh_opts(timeout) + ["-p", str(port),
                                                _ssh_target(node), "true"],
                capture_output=True, text=True, timeout=timeout + 2)
            status.ssh_ok = proc.returncode == 0
            if not status.ssh_ok:
                tail = (proc.stderr or "").strip().splitlines()
                status.error = (tail[-1] if tail else
                                f"ssh exit {proc.returncode}")
        except subprocess.TimeoutExpired:
            status.error = f"ssh timed out after {timeout + 2}s"
        except Exception as e:
            status.error = f"ssh: {e}"
        return status
    except Exception as e:
        status.error = f"probe: {e}"
        return status


def probe_fleet(config, timeout=3):
    """Probe all enabled nodes in parallel (threads). Returns
    {"nodes": {name: status dict}, "summary": {...}}; never raises."""
    nodes = enabled_nodes(config)
    out = {"nodes": {}, "summary": {"probed": len(nodes), "reachable": 0,
                                    "ssh_ok": 0, "unreachable": len(nodes)}}
    if not nodes:
        return out
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, len(nodes))) as pool:
        futures = {pool.submit(probe_node, n, timeout): n for n in nodes}
        for fut in concurrent.futures.as_completed(futures):
            node = futures[fut]
            try:
                status = fut.result()
            except Exception as e:
                status = NodeStatus(node["name"], node["host"],
                                    node.get("port", 22), error=str(e))
            out["nodes"][node["name"]] = status.to_dict()
            out["summary"]["reachable"] += int(status.reachable)
            out["summary"]["ssh_ok"] += int(status.ssh_ok)
    out["summary"]["unreachable"] = len(nodes) - out["summary"]["reachable"]
    return out


def fetch_captures(node, remote_dir, local_dir, timeout=120, probe_timeout=3):
    """Pull the CONTENTS of remote_dir on `node` into local_dir (created if
    missing) over ssh: rsync when available, scp -r fallback. Returns
    (ok, message, files_transferred). Unreachable nodes fail fast on a short
    pre-probe; every subprocess call carries a hard timeout."""
    name = node.get("name", node.get("host", "?"))
    try:
        quick = probe_node(node, timeout=probe_timeout, ssh=False)
        if not quick.reachable:
            return False, f"[{name}] unreachable: {quick.error}", 0
        os.makedirs(local_dir, exist_ok=True)
        target = _ssh_target(node)
        port = int(node.get("port", 22))
        connect = f"ConnectTimeout={max(1, min(int(timeout), 10))}"
        if shutil.which("rsync") is not None:
            ssh_cmd = f"ssh -o BatchMode=yes -o {connect} -p {port}"
            cmd = ["rsync", "-az", "--timeout", str(max(1, int(timeout))),
                   "-e", ssh_cmd, f"{target}:{remote_dir}/",
                   local_dir.rstrip("/") + "/"]
        elif shutil.which("scp") is not None:
            cmd = ["scp", "-r", "-q", "-o", "BatchMode=yes", "-o", connect,
                   "-P", str(port), f"{target}:{remote_dir}/.", local_dir]
        else:
            return False, f"[{name}] error: neither rsync nor scp on PATH", 0
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"[{name}] error: fetch timed out after {timeout}s", 0
        if proc.returncode != 0:
            lines = (proc.stderr or proc.stdout or "").strip().splitlines()
            return False, (f"[{name}] fetch failed: "
                           f"{lines[0] if lines else proc.returncode}"), 0
        files = sum(len(fs) for _, _, fs in os.walk(local_dir))
        return True, f"[{name}] fetched {files} file(s) into {local_dir}", files
    except Exception as e:
        return False, f"[{name}] error: {e}", 0


def run_remote_attack(node, attack_argv, timeout=300, remote_python="python3"):
    """Run `python3 -m attacker <attack_argv...>` on a node where the
    framework is installed; returns (rc, stdout_tail) — 124 on timeout,
    127 when ssh itself cannot launch. Never raises, never hangs."""
    remote_cmd = f"{remote_python} -m attacker " + " ".join(
        shlex.quote(str(a)) for a in attack_argv)
    ct = max(1, min(int(timeout), 10))
    cmd = ["ssh"] + _ssh_opts(ct) + ["-p", str(int(node.get("port", 22))),
                                     _ssh_target(node), remote_cmd]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
        tail_src = proc.stdout if proc.stdout.strip() else proc.stderr
        return proc.returncode, (tail_src or "")[-4000:]
    except subprocess.TimeoutExpired:
        return 124, f"error: remote attack timed out after {timeout}s"
    except Exception as e:
        return 127, f"error: could not launch ssh: {e}"


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile
    print("fleet config + remote orchestration (pure python):")
    cfg = load_config()
    names = [n["name"] for n in enabled_nodes(cfg)]
    check("shipped nodes.json parses with the five fleet nodes enabled",
          sorted(names) == ["koula", "sasa", "soula", "toula", "voula"])
    check("fleet roles are gpu/cpu pairs as deployed",
          get_node(cfg, "toula")["role"] == "gpu"
          and get_node(cfg, "koula")["role"] == "gpu"
          and get_node(cfg, "soula")["role"] == "cpu")
    check("disabled localhost example present",
          get_node(cfg, "localhost")["enabled"] is False)
    try:
        get_node(cfg, "no-such-node")
        check("unknown node lookup is a descriptive KeyError", False)
    except KeyError as e:
        check("unknown node lookup is a descriptive KeyError",
              "no-such-node" in str(e))
    for bad, frag in (({"name": "x", "host": "h", "role": "tpu"}, "role"),
                      ({"name": "x", "role": "gpu"}, "host"),
                      ({"name": "toula", "host": "h2", "role": "gpu"},
                       "duplicate")):
        try:
            _validate_config({"nodes": enabled_nodes(cfg) + [bad]})
            check(f"config drift ({frag}) raises", False)
        except ValueError:
            check(f"config drift ({frag}) raises", True)

    dead = {"name": "dead", "host": "192.0.2.1", "role": "cpu"}
    t0 = time.monotonic()
    st = probe_node(dead, timeout=1)
    dt = time.monotonic() - t0
    check("unreachable node probes as not-reachable with an error",
          not st.reachable and not st.ssh_ok and st.error)
    check(f"unreachable probe is fast ({dt:.2f}s < 5s)", dt < 5.0)

    import threading
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    srv.settimeout(0.2)
    lport = srv.getsockname()[1]
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

    drain = threading.Thread(target=_drain, daemon=True)
    drain.start()
    st = probe_node({"name": "loop", "host": "127.0.0.1", "port": lport},
                    timeout=1, ssh=False)
    check("localhost socket probes reachable with a latency",
          st.reachable and st.latency_ms is not None and not st.ssh_ok)
    report = probe_fleet({"nodes": [dead,
                                    {"name": "loop", "host": "127.0.0.1",
                                     "port": lport, "role": "cpu"}]},
                         timeout=1)
    check("probe_fleet summarizes mixed fleet",
          report["summary"]["probed"] == 2
          and report["summary"]["reachable"] == 1
          and report["summary"]["unreachable"] == 1
          and report["nodes"]["dead"]["reachable"] is False
          and report["nodes"]["loop"]["reachable"] is True)
    rc, tail = run_remote_attack({"name": "loop", "host": "127.0.0.1",
                                  "port": lport}, ["--self-test"], timeout=5)
    check("remote attack on a non-ssh peer returns rc!=0 with a tail",
          rc != 0 and isinstance(tail, str))
    draining.set()
    srv.close()
    drain.join(timeout=2)

    with tempfile.TemporaryDirectory() as d:
        t0 = time.monotonic()
        okf, msg, files = fetch_captures(dead, "captures",
                                         os.path.join(d, "dead"), timeout=10,
                                         probe_timeout=1)
        dt = time.monotonic() - t0
        check("fetch from unreachable node is a clean fast failure",
              not okf and files == 0 and "unreachable" in msg and dt < 5.0)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
