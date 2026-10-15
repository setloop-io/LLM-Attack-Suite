#!/usr/bin/env python3
"""benchmark_integrity — the split integrity/availability battery
(`benchmark.split-integrity-availability.v1`).

One command runs the seven integrity/availability attacks offline at the toy
tier and grades them through `integrity-gate.v1` into a report card:

    python -m attacker --benchmark-integrity full --workdir out/integ
    python -m attacker --benchmark-integrity-grade out/integ   # torch-free

Unlike benchmark.v1 this battery is CAPTURE-FREE: there is no target config
and no split-LLM capture step — each attack is a self-contained offline
experiment (`python -m attacker --mode training --attack NAME --toy`). The
card carries a null `target`.

Doctrine (mirrors benchmark.v1):
  - the battery is the VERSIONED STANDARD; changing the attack set or the axis
    map is a NEW battery version, never a silent edit;
  - the driver shells out per attack so a failed attack is journaled as an
    error artifact (never a crash);
  - grading is Aggregation B (calibration §7, D3): only `defense`-role cells
    roll into an axis grade; `baseline` cells are asserted OPEN and `control`
    cells asserted CLOSED (D4) but do NOT aggregate. Every axis carries its
    OPEN/CLOSED/UNMEASURED counts and the NAMES of every OPEN defense cell —
    a bare "AT-RISK" is not informative (calibration §7.1).

Coverage (structural) is reported by attacker/coverage.py separately;
gradeability (n_seeds>=3, a finite in-domain value) is what this card grades.
The card shows both lenses so coverage never silently outruns gradeability.

Pure stdlib, torch-free.
"""

import argparse
import json
import os
import subprocess
import sys

from . import artifacts, integrity_gate
from .scorecard import load_artifact

BATTERY_VERSION = "benchmark.split-integrity-availability.v1"
REPORTCARD_SCHEMA = "dtraining.attacker.benchmark_integrity.v1"
CELL_INDEX_NAME = "integrity_cells.json"

INTEGRITY_AXES = {
    "S10": "protocol & control-plane metadata (negotiation, epochs, keys)",
    "S13": "AI artifacts & supply chain",
    "LLM04": "Supply Chain",
    "LLM05": "Data and Model Poisoning",
    "LLM06": "Unbounded Consumption",
}

GRADE_SEVERITY = {"OPEN": 3, "AT-RISK": 2, "CLOSED": 1, "UNMEASURED": 0}


class AttackSpec:
    """One battery attack. `mode` is pinned to 'training' — the only mode all
    seven share, and the only one malicious-server & poisoning-backdoor
    support (calibration §2)."""
    __slots__ = ("attack", "experiment", "mode")

    def __init__(self, attack, experiment, mode="training"):
        self.attack = attack
        self.experiment = experiment
        self.mode = mode


BATTERY_ATTACKS = (
    AttackSpec("protocol-integrity", "protocol_integrity"),
    AttackSpec("malicious-client", "malicious_client"),
    AttackSpec("malicious-server", "malicious_server_steering"),
    AttackSpec("poisoning-backdoor", "poisoning_backdoor"),
    AttackSpec("resource-exhaustion", "resource_exhaustion"),
    AttackSpec("key-session-abuse", "key_session_abuse"),
    AttackSpec("supply-chain", "supply_chain"),
)

TIERS = {"full": {"quick": False, "cell_timeout_s": 7200},
         "quick": {"quick": True, "cell_timeout_s": 1800}}


def _adapt_passthrough(rows):
    """arm + world already present (protocol / malicious-client /
    key-session), or arm-only with world absent -> None (resource)."""
    return [dict(r) for r in rows]


def _adapt_supply_chain(rows):
    cells = []
    for r in rows:
        if r.get("arm") == "all":
            cells.append({**r, "world": "negative-control"})
        else:
            cells.append({**r, "world": "permissive"})
            cells.append({**r, "world": "enforcing"})
    return cells


def _adapt_poisoning(rows):
    cells = []
    for r in rows:
        if r.get("arm") == "zero-poison":
            cells.append({**r, "world": "zero-poison"})
        else:
            for world in ("undefended", "defended", "random-corruption"):
                cells.append({**r, "world": world})
    return cells


def _adapt_malicious_server(rows):
    cells = []
    for r in rows:
        if "crafted_minus_honest_pp" in r:
            cells.append({**r, "arm": "bwd-direction", "world": None})
        else:
            cells.append({**r, "arm": "bwd-direction"})
    return cells


_ADAPTERS = {
    "supply_chain": _adapt_supply_chain,
    "poisoning_backdoor": _adapt_poisoning,
    "malicious_server_steering": _adapt_malicious_server,
}


def normalize_artifact(art):
    """(experiment, [normalized cells]) for one artifact dict. Every summary
    row is accounted for by >=1 cell — no silent drops (calibration §1)."""
    rows = [r for r in (art.get("summary") or []) if isinstance(r, dict)]
    exp = (rows[0].get("experiment") if rows
           else (art.get("config") or {}).get("attack"))
    cells = _ADAPTERS.get(exp, _adapt_passthrough)(rows)
    for c in cells:
        c["experiment"] = exp
    return exp, cells


def _enrich_integrity(cells, gate):
    """Score every normalized cell through `gate`. The gate is data-in to the
    scoring hop — the object passed here is the object that scores (proved by
    the runtime-spy test), never a module default resolved inside."""
    scored = []
    for c in cells:
        rule = integrity_gate.match_rule(c, gate)
        result = integrity_gate.score_cell(c, cells, gate)
        scored.append({
            "experiment": c.get("experiment"), "arm": c.get("arm"),
            "world": c.get("world"), "n_seeds": c.get("n_seeds"),
            "signal_key": rule.signal_key if rule else None,
            "axes": list(rule.axes) if rule else [],
            "scale": rule.scale if rule else None,
            "control_scope": rule.control_scope if rule else None,
            "control_key": rule.control_key if rule else None,
            **result})
    return scored


def _cell_name(c):
    tail = f"/{c['world']}" if c.get("world") is not None else ""
    return f"{c['experiment']}/{c['arm']}{tail}"


def _grade_axis(axis, scored):
    """Aggregation B: only defense cells decide the axis grade; baselines and
    controls are asserted elsewhere, never aggregated (D3)."""
    dfn = [c for c in scored if c["role"] == "defense" and axis in c["axes"]]
    opens = [c for c in dfn if c["verdict"] == "OPEN"]
    closed = [c for c in dfn if c["verdict"] == "CLOSED"]
    unmeas = [c for c in dfn if c["verdict"] == "UNMEASURED"]
    if opens and closed:
        grade = "AT-RISK"
    elif opens:
        grade = "OPEN"
    elif closed:
        grade = "CLOSED"
    else:
        grade = "UNMEASURED"
    return {"axis": axis, "surface": INTEGRITY_AXES[axis], "grade": grade,
            "n_defense": len(dfn), "n_open": len(opens),
            "n_closed": len(closed), "n_unmeasured": len(unmeas),
            "open_defense_cells": sorted(_cell_name(c) for c in opens)}


def _role_assertions(scored):
    """D4: every gradeable, in-profile-axis baseline must be OPEN and control
    CLOSED. UNMEASURED cells and the informational aggregate are exempt. A
    violation is a real stop condition (calibration §6) — surfaced, never
    silently swallowed."""
    violations = []
    for c in scored:
        if c["verdict"] == "UNMEASURED" or c["role"] == "informational" \
                or not c["axes"]:
            continue
        if c["role"] == "baseline" and c["verdict"] != "OPEN":
            violations.append({"cell": _cell_name(c), "role": "baseline",
                               "expected": "OPEN", "got": c["verdict"]})
        if c["role"] == "control" and c["verdict"] != "CLOSED":
            violations.append({"cell": _cell_name(c), "role": "control",
                               "expected": "CLOSED", "got": c["verdict"]})
    return violations


def grade_integrity(scored, gate, tier=None, n_summary_rows=None, errors=None):
    """Pure: scored cells -> the report card (Aggregation B, D3)."""
    axes = {a: _grade_axis(a, scored) for a in INTEGRITY_AXES}
    measured = [a for a in axes.values() if a["grade"] != "UNMEASURED"]
    overall = (max(measured, key=lambda a: GRADE_SEVERITY[a["grade"]])["grade"]
               if measured else "UNMEASURED")
    tally = {v: sum(1 for c in scored if c["verdict"] == v)
             for v in ("OPEN", "CLOSED", "UNMEASURED")}
    violations = _role_assertions(scored)
    return {"schema": REPORTCARD_SCHEMA, "battery_version": BATTERY_VERSION,
            "tier": tier, "target": None,
            "gate": {"version": gate.version,
                     "n_seeds_required": gate.n_seeds_required,
                     "digest": integrity_gate.gate_digest(gate)},
            "axes": axes, "overall": overall, "cell_tally": tally,
            "n_cells": len(scored), "n_summary_rows": n_summary_rows,
            "assertions": {"ok": not violations, "violations": violations},
            "cells": scored, "errors": errors or [],
            "provenance": artifacts.light_provenance()}


def grade_from_artifacts(paths, gate, tier=None):
    """Load, normalize, score and grade the seven artifacts -> (card, index).
    Duplicate normalized coordinates are a fail-closed ValueError (no silent
    double count); an unreadable artifact is recorded, not raised."""
    all_cells, errors, n_rows = [], [], 0
    for p in paths:
        try:
            art = load_artifact(p)
        except (ValueError, OSError) as e:
            errors.append({"artifact": str(p), "error": str(e)})
            continue
        n_rows += len([r for r in (art.get("summary") or [])
                       if isinstance(r, dict)])
        for rec in (art.get("results") or []):
            if isinstance(rec, dict) and "error" in rec:
                errors.append({"artifact": str(p),
                               "experiment": rec.get("experiment"),
                               "error": rec["error"]})
        _, cells = normalize_artifact(art)
        all_cells.extend(cells)
    seen = {}
    for c in all_cells:
        key = (c.get("experiment"), c.get("arm"), c.get("world"))
        if key in seen:
            raise ValueError(f"duplicate normalized cell {key} — refusing to "
                             "double-count (fail-closed)")
        seen[key] = c
    scored = _enrich_integrity(all_cells, gate)
    card = grade_integrity(scored, gate, tier=tier, n_summary_rows=n_rows,
                           errors=errors)
    index = {"schema": REPORTCARD_SCHEMA, "battery_version": BATTERY_VERSION,
             "tier": tier, "target": None,
             "gate_config": integrity_gate.gate_config(gate),
             "gate_digest": integrity_gate.gate_digest(gate),
             "cells": all_cells, "artifacts": [str(p) for p in paths]}
    return card, index


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else ("-" if v is None
                                                    else str(v))


def render_integrity_md(card):
    """The integrity/availability report card, markdown."""
    g = card["gate"]
    t = card["cell_tally"]
    lines = [
        f"# Split integrity/availability report card — `{card.get('tier') or '?'}` tier",
        "",
        f"- **battery**: `{card['battery_version']}` (the versioned standard)",
        f"- **gate**: `{g['version']}` (seeds >= {g['n_seeds_required']}, "
        f"digest `{g['digest'][:12]}`)",
        f"- **target**: none (capture-free battery)",
        f"- **overall grade**: **{card['overall']}**",
        f"- **cells**: {card['n_cells']} normalized from "
        f"{card.get('n_summary_rows')} summary rows — "
        f"{t['OPEN']} OPEN, {t['CLOSED']} CLOSED, {t['UNMEASURED']} UNMEASURED",
        f"- **role assertions** (D4: baseline OPEN, control CLOSED): "
        f"{'all hold' if card['assertions']['ok'] else 'VIOLATED'}",
        "",
        "| axis | surface | grade | defense cells (O/C/U) |",
        "|---|---|---|---|"]
    for a in card["axes"].values():
        lines.append(f"| {a['axis']} | {a['surface']} | **{a['grade']}** | "
                     f"{a['n_defense']} ({a['n_open']}/{a['n_closed']}/"
                     f"{a['n_unmeasured']}) |")
    for a in card["axes"].values():
        lines += ["", f"## {a['axis']} — {a['surface']}: **{a['grade']}**", ""]
        if a["open_defense_cells"]:
            lines.append("OPEN defense cells: "
                         + ", ".join(f"`{n}`" for n in a["open_defense_cells"]))
        else:
            lines.append("_no OPEN defense cells_")
    if not card["assertions"]["ok"]:
        lines += ["", "## Role-assertion violations (STOP CONDITION)", ""]
        for v in card["assertions"]["violations"]:
            lines.append(f"- `{v['cell']}` ({v['role']}): expected "
                         f"{v['expected']}, got {v['got']}")
    if card["errors"]:
        lines += ["", "## Errors (excluded from grades)", ""]
        for e in card["errors"]:
            lines.append(f"- {e.get('experiment') or e.get('artifact')}: "
                         f"{e['error']}")
    prov = card.get("provenance") or {}
    lines += ["", "---", "",
              f"host `{prov.get('hostname', '?')}`, commit "
              f"`{prov.get('dtraining_commit', '?')}`, "
              f"{prov.get('timestamp_utc', '?')}", ""]
    return "\n".join(lines)


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_subprocess(cmd, timeout):
    try:
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True,
                              text=True, timeout=timeout)
        tail = proc.stdout if proc.stdout.strip() else proc.stderr
        return proc.returncode, (tail or "")[-2000:]
    except subprocess.TimeoutExpired:
        return 124, f"error: cell timed out after {timeout}s"
    except Exception as e:
        return 127, f"error: could not launch cell: {e}"


def _attack_argv(spec, tier, out, evidence):
    cmd = [sys.executable, "-m", "attacker", "--mode", spec.mode,
           "--attack", spec.attack, "--toy", "--output", out]
    if TIERS[tier]["quick"]:
        cmd.append("--quick")
    if evidence:
        cmd.append("--evidence")
    return cmd


def _fabricate_error_artifact(path, spec, error):
    if os.path.isfile(path):
        return
    out = artifacts.make_artifact(
        REPORTCARD_SCHEMA, {"attack": spec.experiment, "battery_cell": True},
        "integrity battery cell placeholder: the cell failed before writing "
        "its artifact; the error is the record")
    out["results"].append({"experiment": spec.experiment, "error": error})
    artifacts.write_artifact(path, out)


def run_integrity_battery(tier, workdir, runner=None,
                          gate=integrity_gate.DEFAULT_INTEGRITY_GATE,
                          evidence=False):
    """Execute the battery tier. Never raises on a failed attack: the cell is
    journaled as an error artifact and the affected axes read UNMEASURED.
    Returns 0 — the verdict lives in the card, not the exit code."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r} (valid: {sorted(TIERS)})")
    runner = runner or _run_subprocess
    timeout = TIERS[tier]["cell_timeout_s"]
    artdir = os.path.join(workdir, "artifacts")
    os.makedirs(artdir, exist_ok=True)
    paths = []
    for spec in BATTERY_ATTACKS:
        out = os.path.join(artdir, f"{spec.experiment}.json")
        print(f"[benchmark-integrity] {spec.attack} ...")
        rc, tail = runner(_attack_argv(spec, tier, out, evidence), timeout)
        if not (rc == 0 and os.path.isfile(out)):
            last = tail.strip().splitlines()[-1] if tail.strip() else ""
            _fabricate_error_artifact(out, spec, f"cell rc={rc}: {last}")
        paths.append(out)
    card, index = grade_from_artifacts(paths, gate, tier=tier)
    _write_outputs(workdir, card, index)
    print(f"[benchmark-integrity] overall {card['overall']} "
          f"({card['cell_tally']['OPEN']} OPEN / "
          f"{card['cell_tally']['CLOSED']} CLOSED / "
          f"{card['cell_tally']['UNMEASURED']} UNMEASURED) -> "
          f"{os.path.join(workdir, 'integrity_reportcard.md')}")
    return 0


def _write_outputs(workdir, card, index):
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, CELL_INDEX_NAME), "w") as f:
        json.dump(index, f, indent=2)
    with open(os.path.join(workdir, "integrity_reportcard.json"), "w") as f:
        json.dump(card, f, indent=2)
    with open(os.path.join(workdir, "integrity_reportcard.md"), "w") as f:
        f.write(render_integrity_md(card) + "\n")


def regrade_integrity(workdir):
    """`--benchmark-integrity-grade`: re-derive the card from the cell index,
    torch-free. Refuses a non-matching battery_version; reconstructs the gate
    from the index's serialized config (NOT today's module default) and
    verifies the tamper digest before scoring."""
    index_path = os.path.join(workdir, CELL_INDEX_NAME)
    if not os.path.isfile(index_path):
        raise ValueError(f"{workdir}: no {CELL_INDEX_NAME} — not a "
                         f"{BATTERY_VERSION} workdir")
    with open(index_path) as f:
        index = json.load(f)
    if index.get("battery_version") != BATTERY_VERSION:
        raise ValueError(f"{workdir}: battery_version "
                         f"{index.get('battery_version')!r} != "
                         f"{BATTERY_VERSION} — refusing to regrade")
    gate = integrity_gate.gate_from_config(index["gate_config"])
    if integrity_gate.gate_digest(gate) != index.get("gate_digest"):
        raise ValueError(f"{workdir}: gate digest mismatch — the serialized "
                         "gate config was tampered with")
    n_rows = index.get("n_summary_rows")
    scored = _enrich_integrity(index["cells"], gate)
    card = grade_integrity(scored, gate, tier=index.get("tier"),
                           n_summary_rows=n_rows)
    _write_outputs(workdir, card, index)
    per_axis = ", ".join(f"{a}={v['grade']}" for a, v in card["axes"].items())
    print(f"[benchmark-integrity-grade] {workdir}: overall "
          f"{card['overall']} ({per_axis})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m attacker --benchmark-integrity",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tier", choices=sorted(TIERS),
                    help="full = the whole battery at the toy tier; quick "
                         "passes --quick to every attack (a machinery check)")
    ap.add_argument("--workdir", required=True,
                    help="output dir (artifacts/, integrity_cells.json, "
                         "integrity_reportcard.{json,md})")
    ap.add_argument("--evidence", action="store_true",
                    help="seal every cell with an attacker.evidence chain")
    args = ap.parse_args(argv)
    return run_integrity_battery(args.tier, args.workdir,
                                 evidence=args.evidence)


def grade_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m attacker --benchmark-integrity-grade",
        description="Re-grade an existing integrity battery workdir from "
                    "integrity_cells.json (torch-free).")
    ap.add_argument("workdir", help="integrity battery workdir")
    args = ap.parse_args(argv)
    try:
        return regrade_integrity(args.workdir)
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"[benchmark-integrity-grade] {e}") from None


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile
    print("benchmark_integrity battery (pure python):")
    check("seven attacks pinned to training mode",
          len(BATTERY_ATTACKS) == 7
          and all(a.mode == "training" for a in BATTERY_ATTACKS))
    check("five axes (calibration §4)",
          set(INTEGRITY_AXES) == {"S10", "S13", "LLM04", "LLM05", "LLM06"})

    sc = _adapt_supply_chain([{"experiment": "supply_chain", "arm": "rollback",
                               "bypass_rate_permissive": 1.0,
                               "bypass_rate_enforcing": 0.0, "n_seeds": 3},
                              {"experiment": "supply_chain", "arm": "all",
                               "negative_control_false_rejections": 0,
                               "n_seeds": 3}])
    worlds = sorted((c["arm"], c["world"]) for c in sc)
    check("supply-chain splits one arm row into permissive + enforcing",
          worlds == [("all", "negative-control"), ("rollback", "enforcing"),
                     ("rollback", "permissive")])

    g = integrity_gate.DEFAULT_INTEGRITY_GATE

    def _art(exp, summary):
        return {"schema": f"dtraining.attacker.{exp}.v1",
                "config": {"attack": exp}, "threat_model": "self-test",
                "provenance": {}, "results": [], "summary": summary}

    cells_art = _art("resource_exhaustion",
                     [{"experiment": "resource_exhaustion", "arm": "retry-storm",
                       "amplification_vs_control_mean": 1.400244, "n_seeds": 3},
                      {"experiment": "resource_exhaustion", "arm": "long-context",
                       "amplification_vs_control_mean": 2.568467, "n_seeds": 3}])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "rex.json")
        with open(p, "w") as f:
            json.dump(cells_art, f)
        card, index = grade_from_artifacts([p], g, tier="full")
        rs = {c["arm"]: c for c in card["cells"]}
        check("retry-storm CLOSED at 1.5x (D1), long-context OPEN",
              rs["retry-storm"]["verdict"] == "CLOSED"
              and rs["long-context"]["verdict"] == "OPEN")
        check("LLM06 grades AT-RISK (one defense OPEN, one CLOSED)",
              card["axes"]["LLM06"]["grade"] == "AT-RISK"
              and card["axes"]["LLM06"]["open_defense_cells"]
              == ["resource_exhaustion/long-context"])
        check("every row accounted for (2 rows -> 2 cells)",
              card["n_summary_rows"] == 2 and card["n_cells"] == 2)

        seen = {}
        orig = integrity_gate.score_cell

        def spy(cell, siblings, gate):
            seen["gate"] = gate
            return orig(cell, siblings, gate)
        integrity_gate.score_cell = spy
        try:
            card2, _ = grade_from_artifacts([p], g, tier="full")
        finally:
            integrity_gate.score_cell = orig
        check("the gate passed to score_cell is the card's gate (spy)",
              seen["gate"] is g
              and card2["gate"]["digest"] == integrity_gate.gate_digest(g))

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "rex.json")
        with open(p, "w") as f:
            json.dump(cells_art, f)
        card, index = grade_from_artifacts([p], g, tier="full")
        _write_outputs(d, card, index)
        check("regrade reproduces the card (structural equality)",
              regrade_integrity(d) == 0
              and json.load(open(os.path.join(d, "integrity_reportcard.json")))
              ["cell_tally"] == card["cell_tally"])
        idx = json.load(open(os.path.join(d, CELL_INDEX_NAME)))
        idx["gate_config"]["rules"][0]["threshold"] = 999.0
        with open(os.path.join(d, CELL_INDEX_NAME), "w") as f:
            json.dump(idx, f)
        try:
            regrade_integrity(d)
            check("a hand-edited index threshold is caught by the digest", False)
        except ValueError as e:
            check("a hand-edited index threshold is caught by the digest",
                  "tamper" in str(e))
        idx["battery_version"] = "benchmark.v1"
        idx["gate_config"]["rules"][0]["threshold"] = 1.5
        with open(os.path.join(d, CELL_INDEX_NAME), "w") as f:
            json.dump(idx, f)
        try:
            regrade_integrity(d)
            check("regrade refuses a non-matching battery_version", False)
        except ValueError as e:
            check("regrade refuses a non-matching battery_version",
                  "refusing to regrade" in str(e))

    with tempfile.TemporaryDirectory() as d:
        def fake_runner(cmd, timeout):
            out = cmd[cmd.index("--output") + 1]
            exp = os.path.basename(out)[:-5]
            summ = ([{"experiment": exp, "arm": "retry-storm",
                      "amplification_vs_control_mean": 1.4, "n_seeds": 3}]
                    if exp == "resource_exhaustion" else [])
            with open(out, "w") as f:
                json.dump(_art(exp, summ), f)
            return 0, "ok"
        rc = run_integrity_battery("quick", d, runner=fake_runner)
        card = json.load(open(os.path.join(d, "integrity_reportcard.json")))
        check("fake battery returns 0, writes the card + index + md",
              rc == 0 and card["battery_version"] == BATTERY_VERSION
              and all(os.path.isfile(os.path.join(d, n)) for n in
                      (CELL_INDEX_NAME, "integrity_reportcard.json",
                       "integrity_reportcard.md")))
        check("card target is null (capture-free battery)",
              card["target"] is None)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
