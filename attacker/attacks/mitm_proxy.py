#!/usr/bin/env python3
"""mitm-proxy — man-in-the-middle capture cell (deployment shape (c): the
attacker sits between the local node and the cloud server).

Wraps attacker.mitm.MitmProxy as a long-running capture cell: listen
locally, relay to --target unchanged, tee BOTH directions into a capture
journal (frozen wire_shape schema: {"ts", "bytes", "dir", "session_id"},
one JSON object per line, payload sidecars <journal>.c2s/.s2c). On stop
(--duration elapsed or Ctrl-C) an artifact summarizing the capture is
written: records / bytes_c2s / bytes_s2c / sessions / duration_s + the
journal path.

Usage:
    python -m attacker --mode inference --attack mitm-proxy \
        --target 127.0.0.1:9000 --listen 8080 --journal out.jsonl \
        --duration 60 --output mitm.json --evidence
    python -m attacker --mode training --attack mitm-proxy \
        --target cloud.internal:9000 --frame len32 --max-bytes 1073741824
"""

import argparse
import sys
import time

from .. import artifacts
from ..mitm import FRAMES, MitmProxy

EXPERIMENT_ID = "mitm_proxy"
MODES = ("training", "inference")
REQUIRES_LABELS = False
DESCRIPTION = ("sit between the local node and the cloud server, relaying "
               "traffic unchanged while teeing both directions into a "
               "capture journal (frozen wire_shape schema)")

THREAT_MODEL = ("MAN-IN-THE-MIDDLE (deployment shape c): the attacker "
                "controls a relay between the local node and the cloud "
                "server and observes every byte in both directions. "
                "Passive: traffic is forwarded unchanged (the tee cap "
                "drops journal records, never relayed bytes). Label-free "
                "wire observation; captures feed shape/timing analyses.")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True,
                    help="real server as host:port (traffic is relayed "
                         "here unchanged)")
    ap.add_argument("--listen", type=int, default=0,
                    help="local listen port (0 = ephemeral; the bound "
                         "port is printed and recorded)")
    ap.add_argument("--listen-host", default="127.0.0.1",
                    help="local listen address")
    ap.add_argument("--journal", default=None,
                    help="capture journal path (default: <output>.mitm.jsonl"
                         " or mitm_capture.jsonl); payload sidecars land at "
                         "<journal>.c2s/.s2c")
    ap.add_argument("--frame", choices=FRAMES, default="raw",
                    help="tee per recv chunk (raw) or split records at "
                         "len32 frame boundaries")
    ap.add_argument("--max-bytes", type=int, default=None,
                    help="total tee cap across both directions (default: "
                         "uncapped); the relay itself is never throttled")
    ap.add_argument("--duration", type=float, default=60,
                    help="capture duration in seconds (Ctrl-C stops "
                         "earlier, sealing the capture)")
    ap.add_argument("--output", default=None,
                    help="artifact JSON path; <output>.jsonl is the "
                         "crash journal")
    ap.add_argument("--evidence", action="store_true",
                    help="write a tamper-evident evidence chain alongside "
                         "the artifact (attacker.evidence)")
    return ap


def _parse_target(text):
    host, sep, port = text.rpartition(":")
    if not sep or not host:
        raise ValueError(f"--target must be host:port, got {text!r}")
    try:
        port = int(port)
    except ValueError:
        raise ValueError(f"--target port is not an int: {text!r}")
    if not (1 <= port <= 65535):
        raise ValueError(f"--target port out of range: {port}")
    return host, port


def _make_evidence(args):
    if not args.evidence:
        return None
    if not args.output:
        print(f"[{EXPERIMENT_ID}] --evidence needs --output; continuing "
              "without an evidence chain")
        return None
    from ..evidence import EvidenceLogger
    return EvidenceLogger(args.output, node_role="attacker/mitm")


def run(args):
    try:
        host, port = _parse_target(args.target)
    except ValueError as e:
        print(f"[{EXPERIMENT_ID}] error: {e}", file=sys.stderr)
        return 2
    journal = args.journal or (args.output + ".mitm.jsonl" if args.output
                               else "mitm_capture.jsonl")
    try:
        proxy = MitmProxy(args.listen_host, args.listen, host, port,
                          journal, frame=args.frame,
                          max_bytes=args.max_bytes)
    except OSError as e:
        print(f"[{EXPERIMENT_ID}] error: cannot listen on "
              f"{args.listen_host}:{args.listen}: {e}", file=sys.stderr)
        return 2
    ev = _make_evidence(args)
    out = artifacts.make_artifact(
        "dtraining.attacker.mitm_proxy.v1",
        {"attack": EXPERIMENT_ID, "mode": getattr(args, "mode", None),
         "target": f"{host}:{port}", "listen_host": args.listen_host,
         "listen": args.listen, "journal": journal, "frame": args.frame,
         "max_bytes": args.max_bytes, "duration": args.duration},
        THREAT_MODEL,
        interpretation="One capture cell: per-direction byte/record totals "
                       "and session count over the capture window; the "
                       "journal (frozen wire_shape schema) + payload "
                       "sidecars are the analysis input.",
        evidence=ev)
    with proxy:
        print(f"[{EXPERIMENT_ID}] listening on "
              f"{args.listen_host}:{proxy.port} -> {host}:{port} for "
              f"{args.duration}s (journal {journal}, frame={args.frame})")
        t0 = time.monotonic()
        try:
            while True:
                remaining = args.duration - (time.monotonic() - t0)
                if remaining <= 0:
                    break
                time.sleep(min(0.1, remaining))
        except KeyboardInterrupt:
            print(f"[{EXPERIMENT_ID}] interrupted; sealing the capture")
        stats = proxy.stats()
        listen_port = proxy.port
    rec = {"experiment": EXPERIMENT_ID, "target": f"{host}:{port}",
           "listen_port": listen_port, **stats}
    artifacts.append_jsonl(args.output, rec)
    out["results"].append(rec)
    out["summary"].append({
        "experiment": EXPERIMENT_ID, "records": stats["records"],
        "bytes_c2s": stats["bytes_c2s"], "bytes_s2c": stats["bytes_s2c"],
        "sessions": stats["sessions"], "duration_s": stats["duration_s"],
        "capped": stats["capped"], "journal_path": journal})
    if ev is not None:
        ev.event("capture_stopped", stats)
    artifacts.write_artifact(args.output, out, evidence=ev)
    print(f"[{EXPERIMENT_ID}] captured {stats['records']} records "
          f"(c2s={stats['bytes_c2s']}B s2c={stats['bytes_s2c']}B, "
          f"{stats['sessions']} session(s), capped={stats['capped']}) "
          f"in {stats['duration_s']}s")
    return 0


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import json
    import os
    import socket
    import tempfile
    import threading

    from ..mitm import _EchoServer, _echo_roundtrip, _read_journal
    print("mitm-proxy attack driver (pure python, localhost echo pair):")
    with tempfile.TemporaryDirectory() as d, _EchoServer() as echo:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        listen_port = probe.getsockname()[1]
        probe.close()
        journal = os.path.join(d, "capture.jsonl")
        output = os.path.join(d, "mitm.json")
        args = build_parser().parse_args(
            ["--target", f"127.0.0.1:{echo.port}",
             "--listen", str(listen_port), "--journal", journal,
             "--duration", "0.6", "--output", output, "--evidence"])
        driver = threading.Thread(target=lambda: setattr(
            args, "_rc", run(args)), daemon=True)
        driver.start()
        time.sleep(0.15)
        payloads = [b"client->server payload", b"x" * 4096, b"tail"]
        echoed = [_echo_roundtrip(listen_port, p) for p in payloads]
        driver.join(timeout=10)
        rc = getattr(args, "_rc", None)
        check("echo client/server pair round-trips through the attack CLI",
              echoed == payloads and rc == 0 and not driver.is_alive())
        records = _read_journal(journal)
        sent = sum(len(p) for p in payloads)
        check("journal tees every echoed byte in both directions",
              sum(r["bytes"] for r in records if r["dir"] == "c2s") == sent
              and sum(r["bytes"] for r in records if r["dir"] == "s2c")
              == sent)
        check("journal records carry exactly the frozen schema keys",
              records and all(set(r) == {"ts", "bytes", "dir", "session_id"}
                              for r in records))
        art = json.load(open(output))
        summary = art["summary"][0]
        check("artifact summary reports the capture shape",
              summary["bytes_c2s"] == sent and summary["bytes_s2c"] == sent
              and summary["sessions"] == len(payloads)
              and summary["records"] == len(records)
              and summary["duration_s"] > 0)
        from ..evidence import verify_evidence
        good, problems = verify_evidence(output + ".evidence.json")
        check("evidence chain verifies for the mitm artifact",
              good and problems == [])
        bad = build_parser().parse_args(["--target", "no-port-here"])
        check("malformed --target is hard misuse (exit 2)", run(bad) == 2)
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
