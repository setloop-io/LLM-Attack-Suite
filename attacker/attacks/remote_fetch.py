#!/usr/bin/env python3
"""remote-fetch — probe the fleet and pull wire captures from reachable
nodes over SSH (deployment shape (b): the attacker orchestrates remote
capture stores from this host).

Each enabled node in attacker/nodes.json (or --node NAME, repeatable) is
probed (TCP + ssh handshake, short timeouts, parallel); capture dirs are
pulled from reachable nodes into --output-dir preserving node layout:

    <output-dir>/<node>/<remote_dir_basename>/

An unreachable fleet is a TRUTHFUL RESULT, not a crash: the artifact is
written with all-unreachable per-node statuses and the exit code is 0.
Non-zero exit is reserved for hard misuse (bad config, unknown node).

Usage:
    python -m attacker --mode inference --attack remote-fetch \
        --remote-dir captures --output-dir pulled --output fetch.json
    python -m attacker --mode training --attack remote-fetch --node toula \
        --output fetch_toula.json --evidence
"""

import argparse
import os
import sys

from .. import artifacts, remote

EXPERIMENT_ID = "remote_fetch"
MODES = ("training", "inference")
REQUIRES_LABELS = "none"
DESCRIPTION = ("probe the fleet over SSH and pull capture dirs from "
               "reachable nodes into <node>/<dir>/ (unreachable fleet = "
               "clean all-unreachable artifact, exit 0)")

THREAT_MODEL = ("REMOTE ATTACKER (deployment shape b): the attacker "
                "orchestrates/pulls wire-capture stores from fleet nodes "
                "over SSH from a separate host. Passive with respect to "
                "the observed traffic: captures are copied, never "
                "modified. Probing is reachability-only (TCP + "
                "'ssh ... true'); no attack math runs on this path.")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes-config", default=remote.DEFAULT_CONFIG_PATH,
                    help="fleet config (default: the shipped nodes.json)")
    ap.add_argument("--node", action="append", default=None,
                    help="limit to these ENABLED node names (repeatable; "
                         "default: every enabled node)")
    ap.add_argument("--remote-dir", default="captures",
                    help="capture directory on each node (its basename "
                         "becomes the layout leaf)")
    ap.add_argument("--output-dir", default="remote_captures",
                    help="local pull destination: <output-dir>/<node>/"
                         "<remote_dir_basename>/")
    ap.add_argument("--output", default=None,
                    help="artifact JSON path; <output>.jsonl is the "
                         "per-node crash journal")
    ap.add_argument("--probe-timeout", type=float, default=3,
                    help="per-node reachability timeout (seconds)")
    ap.add_argument("--timeout", type=float, default=120,
                    help="per-node fetch timeout (seconds)")
    ap.add_argument("--evidence", action="store_true",
                    help="write a tamper-evident evidence chain alongside "
                         "the artifact (attacker.evidence)")
    return ap


def _dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _make_evidence(args):
    if not args.evidence:
        return None
    if not args.output:
        print(f"[{EXPERIMENT_ID}] --evidence needs --output; continuing "
              "without an evidence chain")
        return None
    from ..evidence import EvidenceLogger
    return EvidenceLogger(args.output, node_role="attacker/remote-fetch")


def run(args):
    try:
        config = remote.load_config(args.nodes_config)
    except (OSError, ValueError) as e:
        print(f"[{EXPERIMENT_ID}] error: cannot load nodes config: {e}",
              file=sys.stderr)
        return 2
    if args.node:
        nodes = []
        for name in args.node:
            try:
                node = remote.get_node(config, name)
            except KeyError as e:
                print(f"[{EXPERIMENT_ID}] error: {e}", file=sys.stderr)
                return 2
            if not node["enabled"]:
                print(f"[{EXPERIMENT_ID}] error: node {name!r} is disabled "
                      f"in the config", file=sys.stderr)
                return 2
            nodes.append(node)
    else:
        nodes = remote.enabled_nodes(config)

    report = remote.probe_fleet({"nodes": nodes}, timeout=args.probe_timeout)
    ev = _make_evidence(args)
    out = artifacts.make_artifact(
        "dtraining.attacker.remote_fetch.v1",
        {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
         "nodes": [n["name"] for n in nodes], "remote_dir": args.remote_dir,
         "output_dir": args.output_dir, "probe_timeout": args.probe_timeout,
         "timeout": args.timeout},
        THREAT_MODEL,
        interpretation="Per-node reachability/ssh/fetch status cells. An "
                       "all-unreachable summary means the fleet could not "
                       "be contacted from this host during the run — a "
                       "truthful measurement, not a framework failure.",
        evidence=ev)
    leaf = os.path.basename(args.remote_dir.rstrip("/")) or "captures"
    totals = {"fetched": 0, "files": 0, "bytes": 0}
    for node in nodes:
        status = report["nodes"].get(node["name"], {})
        cell = {"experiment": EXPERIMENT_ID, "node": node["name"],
                "role": node["role"], "host": node["host"],
                "reachable": status.get("reachable", False),
                "ssh_ok": status.get("ssh_ok", False),
                "latency_ms": status.get("latency_ms"),
                "fetched": False, "files": 0, "bytes": 0}
        if "error" in status:
            cell["probe_error"] = status["error"]
        if cell["reachable"]:
            local_dir = os.path.join(args.output_dir, node["name"], leaf)
            okf, msg, files = remote.fetch_captures(
                node, args.remote_dir, local_dir, timeout=args.timeout,
                probe_timeout=args.probe_timeout)
            cell["message"] = msg
            cell["fetched"] = bool(okf)
            cell["files"] = files
            if okf:
                cell["bytes"] = _dir_bytes(local_dir)
                totals["fetched"] += 1
                totals["files"] += files
                totals["bytes"] += cell["bytes"]
        print(f"[{EXPERIMENT_ID}] {node['name']}: reachable="
              f"{cell['reachable']} ssh_ok={cell['ssh_ok']} fetched="
              f"{cell['fetched']} files={cell['files']}"
              + (f" error={cell['probe_error']}"
                 if "probe_error" in cell else ""))
        if ev is not None:
            ev.event("node_status", cell)
        artifacts.append_jsonl(args.output, cell)
        out["results"].append(cell)
    summary = {"experiment": EXPERIMENT_ID, "nodes_probed": len(nodes),
               "nodes_reachable": report["summary"]["reachable"],
               "nodes_ssh_ok": report["summary"]["ssh_ok"],
               "nodes_fetched": totals["fetched"],
               "files_total": totals["files"],
               "bytes_total": totals["bytes"],
               "all_unreachable": report["summary"]["reachable"] == 0}
    out["summary"].append(summary)
    artifacts.write_artifact(args.output, out, evidence=ev)
    print(f"[{EXPERIMENT_ID}] probed={summary['nodes_probed']} reachable="
          f"{summary['nodes_reachable']} fetched={summary['nodes_fetched']} "
          f"files={summary['files_total']} bytes={summary['bytes_total']}")
    return 0


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import json
    import tempfile
    import time
    print("remote-fetch attack driver (pure python, unreachable fleet):")
    with tempfile.TemporaryDirectory() as d:
        cfg_path = os.path.join(d, "nodes.json")
        with open(cfg_path, "w") as f:
            json.dump({"schema": remote.SCHEMA, "nodes": [
                {"name": "dead-a", "host": "192.0.2.1", "role": "gpu"},
                {"name": "dead-b", "host": "192.0.2.2", "role": "cpu"}]}, f)
        output = os.path.join(d, "fetch.json")
        args = build_parser().parse_args(
            ["--nodes-config", cfg_path, "--probe-timeout", "1",
             "--timeout", "5", "--output-dir", os.path.join(d, "pulled"),
             "--output", output])
        t0 = time.monotonic()
        rc = run(args)
        dt = time.monotonic() - t0
        art = json.load(open(output))
        summary = art["summary"][0]
        check("unreachable fleet exits 0 with a clean artifact",
              rc == 0 and summary["all_unreachable"] is True
              and summary["nodes_probed"] == 2
              and summary["nodes_reachable"] == 0
              and summary["nodes_fetched"] == 0)
        check(f"unreachable fleet run is fast ({dt:.2f}s < 8s)", dt < 8.0)
        check("per-node cells carry status and zero transfers",
              all(c["reachable"] is False and c["fetched"] is False
                  and c["files"] == 0 and c["bytes"] == 0
                  for c in art["results"]))
        journal = open(output + ".jsonl").read().strip().splitlines()
        check("crash journal has one status cell per node",
              len(journal) == 2)
        bad = build_parser().parse_args(["--nodes-config", cfg_path,
                                         "--node", "no-such-node"])
        check("unknown --node is hard misuse (exit 2)", run(bad) == 2)
        disabled_cfg = os.path.join(d, "disabled.json")
        with open(disabled_cfg, "w") as f:
            json.dump({"nodes": [{"name": "off", "host": "192.0.2.3",
                                  "role": "cpu", "enabled": False}]}, f)
        off = build_parser().parse_args(["--nodes-config", disabled_cfg,
                                         "--node", "off"])
        check("--node of a disabled node is hard misuse (exit 2)",
              run(off) == 2)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
