#!/usr/bin/env python3
"""Generate the integrity regression oracle from the retained calibration CSV.
The source is a deterministic test fixture, not empirical research evidence.
Run: python3 tests/data/gen_integrity_expected.py
"""

import json
import csv
import os
import sys

DOC = os.path.join(os.path.dirname(__file__), "integrity_calibration.csv")
OUT = os.path.join(os.path.dirname(__file__), "integrity_expected.json")

EXPERIMENT_ID = {
    "protocol-integrity": "protocol_integrity",
    "malicious-client": "malicious_client",
    "key-session-abuse": "key_session_abuse",
    "supply-chain": "supply_chain",
    "resource-exhaustion": "resource_exhaustion",
    "malicious-server": "malicious_server_steering",
    "poisoning-backdoor": "poisoning_backdoor",
}


def _none(v):
    return None if v == "None" else v


def _num_or_none(v):
    return None if v == "None" else float(v)


def _row_cells(line):
    """Split one markdown table row into its 14 stripped columns."""
    parts = [p.strip() for p in line.split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def parse_appendix_a(doc_path=DOC):
    """Parse the 113-row calibration fixture into oracle cell records."""
    with open(doc_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    cells = []
    for cols in rows:
        (_num, attack, arm, world, role, axes, signal, _scope, _ckey,
         scale, thr, value, verdict, _reason) = cols[:14]
        cells.append({
            "experiment": EXPERIMENT_ID[attack],
            "arm": _none(arm),
            "world": _none(world),
            "role": "informational" if role == "None" else role,
            "axes": [] if axes == "-" else axes.split("+"),
            "signal_key": signal,
            "scale": scale,
            "integrity_threshold": _num_or_none(thr),
            "value": _num_or_none(value),
            "verdict": verdict.strip("*"),
        })
    return cells


def build(doc_path=DOC):
    cells = parse_appendix_a(doc_path)
    tally = {v: sum(1 for c in cells if c["verdict"] == v)
             for v in ("OPEN", "CLOSED", "UNMEASURED")}
    return {"schema": "llm_attacker.integrity_oracle.v1",
            "source": "tests/data/integrity_calibration.csv "
                      "(locked D1 = 1.5x)",
            "n_cells": len(cells), "tally": tally, "cells": cells}


def main():
    oracle = build()
    with open(OUT, "w") as f:
        json.dump(oracle, f, indent=2)
        f.write("\n")
    print(f"[gen] wrote {OUT}: {oracle['n_cells']} cells, tally {oracle['tally']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
