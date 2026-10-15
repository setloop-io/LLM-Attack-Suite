#!/usr/bin/env python3
"""tests/run_suite.py — multi-environment test runner + merged verdict.

Runs the pytest suite in the primary .venv and (if present) a torch venv,
merges the per-run records each env's plugin wrote (tests/last_run.json),
and writes tests/TEST_REPORT.{json,md}: per-test best outcome across envs,
coverage gaps (tests green in NO env), and the final verdict lines.

Usage: python3 tests/run_suite.py [--torch-venv PATH] [--no-run] (merge only)
Exit 0 iff no failures anywhere and every gap is acknowledged in
tests/known_gaps.json.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")
LAST_RUN = os.path.join(TESTS, "last_run.json")
KNOWN_GAPS = os.path.join(TESTS, "known_gaps.json")
OUT_JSON = os.path.join(TESTS, "TEST_REPORT.json")
OUT_MD = os.path.join(TESTS, "TEST_REPORT.md")

BEST = {"failed": 0, "passed": 1, "skipped": 2}
RANK = {"passed": 0, "skipped": 1, "failed": 2}


def run_env(py, label):
    if os.path.exists(LAST_RUN):
        os.remove(LAST_RUN)
    r = subprocess.run([py, "-m", "pytest", TESTS, "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    rec = {"tests": [], "totals": {"passed": 0, "failed": 0, "skipped": 0}}
    if os.path.exists(LAST_RUN):
        with open(LAST_RUN) as f:
            rec = json.load(f)
    rec["returncode"] = r.returncode
    rec["env_label"] = label
    return r.returncode, rec


def merge(env_records, known_gaps):
    """Per-test best outcome across envs + gap analysis. Pure data in/out."""
    known = {g["nodeid"] for g in known_gaps}
    best = {}
    for label, rec in env_records:
        for t in rec["tests"]:
            cur = best.get(t["nodeid"])
            if cur is None or RANK[t["outcome"]] < RANK[cur["outcome"]]:
                best[t["nodeid"]] = {**t, "env": label}
    green_somewhere = [n for n, t in best.items() if t["outcome"] == "passed"]
    failed_anywhere = sorted({t["nodeid"] for _, rec in env_records
                              for t in rec["tests"] if t["outcome"] == "failed"})
    failed_anywhere += [f"environment:{label}:exit={rec['returncode']}"
                        for label, rec in env_records if rec.get("returncode", 0)]
    if not best:
        failed_anywhere.append("no tests collected")
    never_ran = [n for n, t in best.items()
                 if t["outcome"] == "skipped"]
    gaps = [n for n in never_ran if n not in known]
    return {"best": best, "green_somewhere": sorted(green_somewhere),
            "failed": sorted(failed_anywhere),
            "never_ran": sorted(never_ran), "unacknowledged_gaps": sorted(gaps),
            "acknowledged_gaps": sorted(n for n in never_ran if n in known)}


def verdict_lines(merged, env_labels):
    lines = []
    tdd = "green" if not merged["failed"] else \
        f"RED ({len(merged['failed'])} failing)"
    torch_envs = [l for l in env_labels
                  if l.startswith("torch:") or "torch" in l.split("-")[0]]
    lines.append(f"TDD (torch-free): {tdd}")
    lines.append(f"TORCH TIER: {'covered in ' + ', '.join(torch_envs) if torch_envs else 'NOT RUN — no torch venv'}")
    g = merged["unacknowledged_gaps"]
    a = merged["acknowledged_gaps"]
    lines.append(f"BDD/gaps: {len(a)} acknowledged, {len(g)} UNACKNOWLEDGED")
    for n in g:
        lines.append(f"  - NEVER RAN: {n}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-venv", default=None,
                    help="path to a venv WITH torch (auto-detects "
                         "/tmp/sbvenv when omitted)")
    ap.add_argument("--no-run", action="store_true",
                    help="merge existing last_run.json files only")
    args = ap.parse_args(argv)

    venvs = []
    primary = os.path.join(ROOT, ".venv", "bin", "python")
    if os.path.exists(primary):
        has_torch = subprocess.run([primary, "-c", "import torch"], capture_output=True).returncode == 0
        venvs.append((primary, "torch:primary" if has_torch else "venv-torchfree"))
    torchvenv = args.torch_venv or ("/tmp/sbvenv"
                                    if os.path.exists("/tmp/sbvenv/bin/python")
                                    else None)
    if torchvenv:
        venvs.append((os.path.join(torchvenv, "bin", "python"),
                      f"torch:{os.path.basename(torchvenv)}"))

    env_records = []
    if not args.no_run:
        for py, label in venvs:
            print(f"[suite] running {label} ...")
            rc, rec = run_env(py, label)
            print(f"[suite] {label}: rc={rc} "
                  f"({rec['totals']['passed']}p/{rec['totals']['failed']}f"
                  f"/{rec['totals']['skipped']}s)")
            env_records.append((label, rec))
    else:
        with open(LAST_RUN) as f:
            env_records.append(("last-run", json.load(f)))

    with open(KNOWN_GAPS) as f:
        known = json.load(f)["gaps"]
    merged = merge(env_records, known)
    lines = verdict_lines(merged, [l for l, _ in env_records])
    report = {"schema": "llm_attacker.test_report.v1",
              "envs": [l for l, _ in env_records],
              "totals": {"unique_tests": len(merged["best"]),
                         "green_somewhere": len(merged["green_somewhere"]),
                         "failed": len(merged["failed"]),
                         "never_ran": len(merged["never_ran"]),
                         "unacknowledged_gaps": len(merged["unacknowledged_gaps"])},
              "verdict": lines, "merged": {n: t for n, t in merged["best"].items()}}
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=1)
    with open(OUT_MD, "w") as f:
        f.write("# Test report (multi-env)\n\n")
        f.writelines(f"- {l}\n" for l in lines)
        f.write("\n## Never ran in any env\n\n")
        for n in merged["never_ran"]:
            mark = " (acknowledged)" if n in merged["acknowledged_gaps"] \
                else " **UNACKNOWLEDGED**"
            f.write(f"- `{n}`{mark}\n")
    for l in lines:
        print(l)
    print(f"[suite] wrote {OUT_MD} (+ {OUT_JSON})")
    return 0 if not merged["failed"] and not merged["unacknowledged_gaps"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
