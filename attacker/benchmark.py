#!/usr/bin/env python3
"""benchmark — the standard environment benchmark battery (benchmark.v1).

One command grades a target split-LLM environment against the frozen
standard battery and emits an **environment report card**:

    python -m attacker --benchmark quick --target-config target.json \
        --workdir out/bench [--evidence]
    python -m attacker --benchmark-grade out/bench   # re-grade, torch-free

The battery is the VERSIONED STANDARD (`benchmark.v1`): which defenses x
which attacks at which budget caps. Same doctrine as the frozen gate
(attacker/gate.py) — the constants below are the claim; changing any of
them is a NEW battery version, explicitly named, never a silent edit.

benchmark.v1 contents
---------------------
Target config (JSON, TargetConfig): the environment under test — model,
split boundary, corpus, device, and the label-scoring property token.
benchmark.v1 is the TRAINING-seam battery (finetune captures; the bwd
channel is what label-scoring and the tag arm read).

Defenses (the splitbench seam stack, `--defense`):
  full tier: none + the seam pack — gaussian (sigma=15.0, the
  defense-matrix campaign value), sparsify, mid-lite, at-lite, santext,
  custext, snd-lite (splitbench default knobs).
  quick tier: none + mid-lite (a small implementation-check matrix;
  no claim that this is the strongest deployed defense).

Attacks (the fixed standard subset):
  seam-inversion arm vmi   -> S1 (seam activations, fwd)   [per defense]
  label-scoring            -> S9 (property from seam grads) [per defense]
  gradient-inversion arm tag -> S2 (seam gradients, bwd)    [full only]
  gpu-memory arm scrub-check -> S5 (memory residue) — environment probe,
    run ONCE per battery (defense-independent), REAL tiers only (never
    --toy: the toy tier is a machinery check, not a security claim); on a
    GPU-less host the cell degrades to an error record and S5 grades
    UNMEASURED.

Budget caps per tier (the "budget" column of the standard):
  quick: splitbench steps capped at 20, attacks run --quick;
  full:  steps capped at 100, default attack budgets.

Grading (attacker.scorecard verdicts on the frozen gate -> axes):
  per threat axis (README.md surface map S1/S2/S5/S9):
    UNMEASURED  no gated cell (every cell errored or spoke no gate metric)
    OPEN        >=1 gated cell FAILs and none PASSes (attack beats every
                measured configuration on this axis)
    AT-RISK     both FAIL and PASS cells (some configuration is broken,
                some defense holds — best_defense names the holder)
    CLOSED      every gated cell PASSes (no attack beats the gate)
  overall grade = worst measured section (OPEN > AT-RISK > CLOSED);
  Missing or incomplete axes prevent an overall CLOSED result. A FAIL on
  the undefended ("none") baseline with a PASSing defense is AT-RISK, not
  CLOSED: the battery measures a defense MATRIX and reports the matrix.

Orchestration doctrine (repo rules): the driver shells out
(`sys.executable -m attacker ...`) per cell so a torch-less driver host
simply records per-cell errors — cells never raise, failures are
journaled into fabricated error artifacts (the scorecard's not-shown
section) and the axis grades UNMEASURED. The driver itself is pure
stdlib; grade() is a pure function over scored cell records.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from dataclasses import dataclass

from . import artifacts
from .gate import DEFAULT_GATE
from .report import CORPUS_CAVEAT
from .scorecard import (LEGACY_SCORECARD_POLICY, build_scorecard,
                        load_artifact, write_scorecard)

BATTERY_VERSION = "benchmark.v1"
REPORTCARD_SCHEMA = "dtraining.attacker.benchmark.v1"
CELL_INDEX_NAME = "benchmark_cells.json"

AXES = {
    "S1": "seam activations, forward (h* crossing to the body)",
    "S2": "seam gradients, backward (dL/dh* returned by the body)",
    "S5": "GPU memory residue (freed buffers, allocator state)",
    "S9": "labels / properties (property inference over seam signals)",
}

GRADE_SEVERITY = {"OPEN": 3, "AT-RISK": 2, "CLOSED": 1, "INCOMPLETE": 0, "UNMEASURED": 0}


@dataclass(frozen=True)
class DefenseSpec:
    """One splitbench defense cell. knobs: ((key, value), ...) pairs passed
    as --key value (benchmark.v1 pins gaussian sigma=15.0, the campaign
    value; every other defense runs its splitbench defaults)."""
    name: str
    knobs: tuple = ()
    quick: bool = False


@dataclass(frozen=True)
class AttackSpec:
    """One attack cell template of the standard battery.

    args are CLI-flag TEMPLATES formatted with {"model", "split",
    "property_token", "quick"} from the target config / tier — the
    declarative form of "which flags pin this attack to the standard"."""
    attack: str
    arm: str
    axis: str
    per_defense: bool
    quick: bool
    args: tuple = ()


BATTERY_DEFENSES = (
    DefenseSpec("none", quick=True),
    DefenseSpec("gaussian", knobs=(("sigma", 15.0),)),
    DefenseSpec("sparsify"),
    DefenseSpec("mid-lite", quick=True),
    DefenseSpec("at-lite"),
    DefenseSpec("santext"),
    DefenseSpec("custext"),
    DefenseSpec("snd-lite"),
)

BATTERY_ATTACKS = (
    AttackSpec("seam-inversion", "vmi", "S1", per_defense=True, quick=True,
               args=("--arm", "vmi", "--model", "{model}",
                     "--split", "{split}")),
    AttackSpec("label-scoring", "norm", "S9", per_defense=True, quick=True,
               args=("--property", "contains-token:{property_token}")),
    AttackSpec("gradient-inversion", "tag", "S2", per_defense=True,
               quick=False,
               args=("--arm", "tag", "--model", "{model}",
                     "--split", "{split}")),
    AttackSpec("gpu-memory", "scrub-check", "S5", per_defense=False,
               quick=True, args=("--arm", "scrub-check")),
)

TIER_BUDGETS = {
    "quick": {"steps_cap": 20, "attack_quick": True, "cell_timeout_s": 1800},
    "full": {"steps_cap": 100, "attack_quick": False, "cell_timeout_s": 7200},
}


def battery_spec():
    """The battery as a plain JSON-able dict (embedded in report cards)."""
    return {"battery_version": BATTERY_VERSION,
            "axes": dict(AXES),
            "grade_severity": dict(GRADE_SEVERITY),
            "defenses": [dataclasses.asdict(d) for d in BATTERY_DEFENSES],
            "attacks": [dataclasses.asdict(a) for a in BATTERY_ATTACKS],
            "tier_budgets": TIER_BUDGETS}


@dataclass
class TargetConfig:
    """benchmark.v1 target: which environment the battery runs against."""
    model: str = "gpt2"
    split: str = "6"
    mode: str = "finetune"
    corpus_file: str = None
    device: str = "cpu"
    dtype: str = "fp32"
    seq_len: int = 128
    batch_size: int = 4
    lr: float = 5e-5
    seed: int = 1234
    property_token: int = 220

    @classmethod
    def from_json(cls, path):
        """Load + validate; unknown keys / bad values are descriptive
        ValueErrors (BenchConfig.from_json convention)."""
        try:
            with open(path) as f:
                raw = json.load(f)
        except OSError as e:
            raise ValueError(f"cannot read target config {path}: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"target config {path} is not valid JSON: {e}") \
                from e
        if not isinstance(raw, dict):
            raise ValueError(f"target config {path} must be a JSON object, "
                             f"got {type(raw).__name__}")
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - valid)
        if unknown:
            raise ValueError(f"target config {path}: unknown keys {unknown} "
                             f"(valid: {sorted(valid)})")
        return cls(**raw).validate()

    def validate(self):
        """Descriptive ValueError on drift; returns self."""
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("TargetConfig: model must be a non-empty str "
                             "(HF name or local path)")
        if self.mode != "finetune":
            raise ValueError(f"TargetConfig: mode {self.mode!r} unsupported "
                             f"— {BATTERY_VERSION} is the training-seam "
                             "battery (finetune captures feed the bwd "
                             "attacks)")
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"TargetConfig: device {self.device!r} not in "
                             "(cpu, cuda)")
        if self.dtype not in ("fp32", "bf16"):
            raise ValueError(f"TargetConfig: dtype {self.dtype!r} not in "
                             "(fp32, bf16)")
        for name, lo in (("seq_len", 2), ("batch_size", 1), ("seed", 0),
                         ("property_token", 0)):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < lo:
                raise ValueError(f"TargetConfig: {name} must be an int "
                                 f">= {lo}, got {v!r}")
        if not isinstance(self.lr, (int, float)) or self.lr <= 0:
            raise ValueError(f"TargetConfig: lr must be > 0, got {self.lr!r}")
        if self.corpus_file is not None \
                and not isinstance(self.corpus_file, str):
            raise ValueError("TargetConfig: corpus_file must be a path str "
                             "or null")
        from .splitbench import SplitPlan
        SplitPlan.parse(self.split)
        return self


def _cell(cell_id, kind, axis, defense, attack, arm):
    return {"cell_id": cell_id, "kind": kind, "axis": axis,
            "defense": defense, "attack": attack, "arm": arm,
            "status": "pending", "artifact": None, "verdict": None,
            "excess_pp": None, "headline_metric": None,
            "headline_value": None, "error": None}


def grade(cells, target=None, tier=None, gate=DEFAULT_GATE):
    """Pure: scored cell records -> the environment report card.

    cells: list of dicts as produced by the driver (or by hand in tests):
    kind "attack" cells carry verdict/excess_pp (from the scorecard row,
    None when the cell errored or spoke no gate metric) and status
    ("ok"/"error"); kind "capture" cells carry final_loss/mean_loss (the
    utility column). The grading mapping is documented in the module
    docstring (OPEN / AT-RISK / CLOSED / UNMEASURED)."""
    axes = {}
    for axis, surface in AXES.items():
        acells = [c for c in cells
                  if c.get("kind") == "attack" and c.get("axis") == axis]
        gated = [c for c in acells if c.get("verdict") in ("PASS", "FAIL") and c.get("status") != "error"]
        fails = [c for c in gated if c["verdict"] == "FAIL"]
        passes = [c for c in gated if c["verdict"] == "PASS"]
        if not gated:
            g = "UNMEASURED"
        elif fails and passes:
            g = "AT-RISK"
        elif fails:
            g = "OPEN"
        else:
            g = "CLOSED" if len(gated) == len(acells) else "INCOMPLETE"
        worst = max((c for c in fails if c.get("excess_pp") is not None),
                    key=lambda c: c["excess_pp"], default=None)
        best = min((c for c in passes if c.get("excess_pp") is not None),
                   key=lambda c: c["excess_pp"], default=None)
        axes[axis] = {
            "axis": axis, "surface": surface, "grade": g,
            "n_cells": len(acells),
            "n_pass": len(passes), "n_fail": len(fails),
            "n_info": sum(1 for c in acells if c.get("verdict") == "INFO"),
            "n_error": sum(1 for c in acells if c.get("status") == "error"),
            "worst_cell": worst, "best_defense": best,
            "cells": acells}
    measured = [a for a in axes.values() if a["grade"] != "UNMEASURED"]
    overall = (max(measured, key=lambda a: GRADE_SEVERITY[a["grade"]])
               ["grade"] if measured else "UNMEASURED")
    observed_overall = overall
    if overall == "CLOSED" and any(a["grade"] in ("UNMEASURED", "INCOMPLETE") for a in axes.values()):
        overall = "INCOMPLETE"
    utility = {}
    for c in cells:
        if c.get("kind") == "capture" and c.get("status") == "ok" \
                and c.get("final_loss") is not None:
            utility[c["defense"]] = {"final_loss": c["final_loss"],
                                     "mean_loss": c.get("mean_loss")}
    from . import gate as _gate_mod
    return {"schema": REPORTCARD_SCHEMA,
            "battery_version": BATTERY_VERSION,
            "tier": tier,
            "target": (dataclasses.asdict(target)
                       if isinstance(target, TargetConfig) else target),
            "gate": {**dataclasses.asdict(gate),
                     "z": _gate_mod.bonferroni_z(gate)},
            "grading": ("UNMEASURED: no gated cell; OPEN: FAILs and no "
                        "PASS; AT-RISK: FAIL and PASS cells; CLOSED: all "
                        "PASS with complete supplied cells. Overall CLOSED additionally requires all axes; incomplete coverage is reported separately."),
            "axes": axes,
            "utility": utility,
            "overall": overall,
            "observed_overall": observed_overall,
            "complete": all(a["n_cells"] and a["n_pass"] + a["n_fail"] == a["n_cells"] for a in axes.values()),
            "provenance": artifacts.light_provenance()}


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else ("-" if v is None
                                                    else str(v))


def render_reportcard_md(card):
    """The environment report card, markdown."""
    t = card.get("target") or {}
    g = card["gate"]
    lines = [
        f"# Environment report card — `{card.get('tier') or '?'}` tier",
        "",
        f"- **battery**: `{card['battery_version']}` (the versioned "
        "standard; constants in attacker/benchmark.py)",
        f"- **target**: model `{t.get('model')}`, split `{t.get('split')}`, "
        f"device `{t.get('device')}`, mode `{t.get('mode')}`",
        f"- **gate**: `{g['version']}` (excess <= +{g['gate_pp']} pp over "
        f"`{g['control']}`, z={g['z']:.4f}, seeds >= {g['n_seeds_required']})",
        f"- **overall grade**: **{card['overall']}**",
        "",
        "| axis | surface | grade | cells (P/F/I/err) |",
        "|---|---|---|---|"]
    for axis in AXES:
        a = card["axes"][axis]
        lines.append(
            f"| {axis} | {a['surface']} | **{a['grade']}** | "
            f"{a['n_cells']} ({a['n_pass']}/{a['n_fail']}/{a['n_info']}/"
            f"{a['n_error']}) |")
    for axis in AXES:
        a = card["axes"][axis]
        lines += ["", f"## {axis} — {a['surface']}: **{a['grade']}**", ""]
        if not a["cells"]:
            lines.append("_(no cells)_")
            continue
        lines += ["| defense | attack | headline | excess | verdict |",
                  "|---|---|---|---|---|"]
        for c in a["cells"]:
            if c.get("status") == "error":
                lines.append(f"| {c.get('defense') or '-'} | {c['attack']} "
                             f"| - | - | ERROR: {c.get('error')} |")
                continue
            headline = ("-" if not c.get("headline_metric") else
                        f"`{c['headline_metric']}` = "
                        f"{_fmt(c.get('headline_value'))}")
            excess = ("-" if c.get("excess_pp") is None
                      else f"{c['excess_pp']:+.3f} pp")
            lines.append(f"| {c.get('defense') or '-'} | "
                         f"{c['attack']} ({c['arm']}) | {headline} | "
                         f"{excess} | {c.get('verdict') or '-'} |")
        best = a.get("best_defense")
        if best:
            lines.append(f"\nbest defense: `{best['defense']}` "
                         f"(excess {best.get('excess_pp'):+.3f} pp, PASS).")
        worst = a.get("worst_cell")
        if worst:
            lines.append(f"worst cell: `{worst['attack']}` vs "
                         f"`{worst.get('defense') or '-'}` "
                         f"(excess {worst.get('excess_pp'):+.3f} pp, FAIL).")
    if card.get("utility"):
        lines += ["", "## Utility (defense cost)", "",
                  "| defense | final_loss | mean_loss |", "|---|---|---|"]
        for defense, u in sorted(card["utility"].items()):
            lines.append(f"| {defense} | {_fmt(u.get('final_loss'))} | "
                         f"{_fmt(u.get('mean_loss'))} |")
    prov = card.get("provenance") or {}
    lines += ["", "---", "",
              f"host `{prov.get('hostname', '?')}`, commit "
              f"`{prov.get('dtraining_commit', '?')}`, "
              f"{prov.get('timestamp_utc', '?')}",
              "",
              f"> {CORPUS_CAVEAT}", ""]
    return "\n".join(lines)


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_subprocess(cmd, timeout):
    """Default runner: one cell subprocess, hard timeout, never raises.
    Returns (rc, tail)."""
    try:
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True,
                              text=True, timeout=timeout)
        tail_src = proc.stdout if proc.stdout.strip() else proc.stderr
        return proc.returncode, (tail_src or "")[-2000:]
    except subprocess.TimeoutExpired:
        return 124, f"error: cell timed out after {timeout}s"
    except Exception as e:
        return 127, f"error: could not launch cell: {e}"


def _fmt_args(template, ctx):
    return [a.format(**ctx) for a in template]


def _capture_argv(target, defense, capdir, out, steps, evidence):
    cmd = [sys.executable, "-m", "attacker.splitbench",
           "--model", target.model, "--split", target.split,
           "--mode", target.mode, "--steps", str(steps),
           "--seq-len", str(target.seq_len),
           "--batch-size", str(target.batch_size), "--lr", str(target.lr),
           "--device", target.device, "--dtype", target.dtype,
           "--seed", str(target.seed),
           "--defense", defense.name]
    for key, value in defense.knobs:
        cmd += ["--" + key.replace("_", "-"), str(value)]
    if target.corpus_file:
        cmd += ["--corpus-file", target.corpus_file]
    cmd += ["--capture-dir", capdir, "--save-canonical",
            "--output", out]
    if evidence:
        cmd.append("--evidence")
    return cmd


def _attack_argv(spec, target, ctx, capdir, out, quick, evidence):
    cmd = [sys.executable, "-m", "attacker", "--mode", "training",
           "--attack", spec.attack]
    if spec.per_defense:
        cmd += ["--capture-dir", capdir,
                "--canonical-pt", os.path.join(capdir, "canonical.pt")]
    cmd += _fmt_args(spec.args, ctx)
    if quick:
        cmd.append("--quick")
    cmd += ["--output", out]
    if evidence:
        cmd.append("--evidence")
    return cmd


def _fabricate_error_artifact(path, cell, error):
    """A failed cell still leaves a VALID artifact (error record inside):
    the scorecard lists it under not-shown, the axis grades UNMEASURED,
    and nothing raises (rotation_lifetime solve-hardening convention)."""
    if os.path.isfile(path):
        return
    out = artifacts.make_artifact(
        REPORTCARD_SCHEMA,
        {"attack": cell.get("attack") or "splitbench",
         "benchmark_cell": cell["cell_id"], "tier_cell": True},
        "benchmark cell placeholder: the cell failed before writing its "
        "artifact; the error is the record")
    out["results"].append({"experiment": cell.get("attack") or "splitbench",
                           "error": error})
    artifacts.write_artifact(path, out)


def _enrich(cells, workdir):
    """Score every ok attack cell through the scorecard machinery:
    per-cell verdict/excess/headline land back on the cell record."""
    paths = []
    for c in cells:
        p = c.get("artifact")
        if not p or not os.path.isfile(p):
            c["status"] = "error"
            c["error"] = c["error"] or "artifact missing after cell run"
            continue
        try:
            load_artifact(p)
        except (ValueError, OSError) as e:
            c["status"] = "error"
            c["error"] = f"artifact unreadable: {e}"
            continue
        paths.append(p)
    card = build_scorecard(paths, policy=LEGACY_SCORECARD_POLICY) \
        if paths else None
    if card is not None:
        write_scorecard(card, os.path.join(workdir, "scorecard.json"))
        rows = {}
        for r in card["rows"]:
            rows.setdefault(r["source"], []).append(r)
    else:
        rows = {}
    for c in cells:
        p = c.get("artifact")
        if c["status"] != "ok" or not p or p not in rows:
            continue
        cell_rows = rows[p]
        gated = [r for r in cell_rows if r["excess_pp"] is not None]
        worst = max(gated, key=lambda r: r["excess_pp"], default=None)
        row = worst or cell_rows[0]
        c["verdict"] = ("FAIL" if any(r["verdict"] == "FAIL"
                                      for r in cell_rows)
                        else row["verdict"])
        if c["verdict"] != "FAIL" and (any(r["verdict"] in ("ERROR", "INSUFFICIENT_EVIDENCE") for r in cell_rows)
                                         or any(e["source"] == p for e in card["not_shown"])):
            c["verdict"] = "INSUFFICIENT_EVIDENCE"
        c["excess_pp"] = row["excess_pp"]
        c["headline_metric"] = row["headline_metric"]
        c["headline_value"] = row["headline_value"]
        warnings = [w for r in cell_rows for w in r["warnings"]]
        if warnings:
            c["warnings"] = warnings
        if c["kind"] == "capture":
            try:
                art = load_artifact(p)
                rec = (art.get("results") or [{}])[0]
                c["final_loss"] = rec.get("final_loss")
                c["mean_loss"] = rec.get("mean_loss")
            except (ValueError, OSError):
                pass
    return card


def run_battery(tier, target, workdir, evidence=False, runner=None):
    """Execute one battery tier against a target environment.

    runner(cmd, timeout_s) -> (rc, tail): injectable for tests (the fake
    runner fabricates artifacts); default is the subprocess runner.
    Never raises: failed cells are error records. Returns 0 — the
    security verdict lives in the report card, not the exit code
    (--scorecard remains the CI enforcement point)."""
    if tier not in TIER_BUDGETS:
        raise ValueError(f"unknown tier {tier!r} (valid: "
                         f"{sorted(TIER_BUDGETS)})")
    target.validate()
    runner = runner or _run_subprocess
    budget = TIER_BUDGETS[tier]
    quick = budget["attack_quick"]
    steps = budget["steps_cap"]
    timeout = budget["cell_timeout_s"]
    os.makedirs(workdir, exist_ok=True)
    ctx = {"model": target.model, "split": target.split,
           "property_token": target.property_token}
    cells = []
    artdir = os.path.join(workdir, "artifacts")
    os.makedirs(artdir, exist_ok=True)

    defenses = [d for d in BATTERY_DEFENSES if not quick or d.quick]
    specs = [a for a in BATTERY_ATTACKS if not quick or a.quick]

    def drive(cell, cmd):
        print(f"[benchmark] cell {cell['cell_id']}: "
              f"{' '.join(os.path.basename(str(x)) for x in cmd[2:4])} ...")
        rc, tail = runner(cmd, timeout)
        if rc == 0 and cell["artifact"] and os.path.isfile(cell["artifact"]):
            cell["status"] = "ok"
        else:
            cell["status"] = "error"
            cell["error"] = (f"cell rc={rc}: {tail.strip().splitlines()[-1]}"
                             if tail.strip() else f"cell rc={rc}")
            if cell["artifact"]:
                _fabricate_error_artifact(cell["artifact"], cell,
                                          cell["error"])
        cells.append(cell)

    for defense in defenses:
        capdir = os.path.join(workdir, "captures", defense.name)
        cap_art = os.path.join(artdir, f"splitbench_{defense.name}.json")
        cap = _cell(f"capture:{defense.name}", "capture", None,
                    defense.name, "splitbench", target.mode)
        cap["artifact"] = cap_art
        drive(cap, _capture_argv(target, defense, capdir, cap_art, steps,
                                 evidence))
        for spec in [s for s in specs if s.per_defense]:
            out = os.path.join(
                artdir, f"{spec.attack}_{spec.arm}_{defense.name}.json")
            cell = _cell(f"{spec.axis}:{defense.name}:{spec.attack}",
                         "attack", spec.axis, defense.name, spec.attack,
                         spec.arm)
            cell["artifact"] = out
            if cap["status"] != "ok":
                cell["status"] = "error"
                cell["error"] = (f"skipped: capture cell failed "
                                 f"({cap['error']})")
                _fabricate_error_artifact(out, cell, cell["error"])
                cells.append(cell)
                continue
            drive(cell, _attack_argv(spec, target, ctx, capdir, out,
                                     quick, evidence))
    for spec in [s for s in specs if not s.per_defense]:
        out = os.path.join(artdir, f"{spec.attack}_{spec.arm}.json")
        cell = _cell(f"{spec.axis}:env:{spec.attack}", "attack", spec.axis,
                     None, spec.attack, spec.arm)
        cell["artifact"] = out
        drive(cell, _attack_argv(spec, target, ctx, None, out, quick,
                                 evidence))

    _enrich(cells, workdir)
    card = grade(cells, target=target, tier=tier)
    card["config"] = battery_spec()
    card["workdir"] = os.path.abspath(workdir)
    index = {"schema": REPORTCARD_SCHEMA, "battery_version": BATTERY_VERSION,
             "tier": tier, "target": dataclasses.asdict(target),
             "evidence": evidence, "cells": cells}
    for name, payload in ((CELL_INDEX_NAME, index),
                          ("reportcard.json", card)):
        p = os.path.join(workdir, name)
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
    with open(os.path.join(workdir, "reportcard.md"), "w") as f:
        f.write(render_reportcard_md(card) + "\n")
    print(f"[benchmark] report card: overall {card['overall']} "
          f"-> {os.path.join(workdir, 'reportcard.md')}")
    return 0


def regrade(workdir):
    """`--benchmark-grade`: re-derive the report card of an existing
    benchmark workdir from its cell index + artifacts (torch-free)."""
    index_path = os.path.join(workdir, CELL_INDEX_NAME)
    with open(index_path) as f:
        index = json.load(f)
    cells = index["cells"]
    for c in cells:
        if c.get("status") != "error":
            c["status"] = "ok"
        for k in ("verdict", "excess_pp", "headline_metric",
                  "headline_value"):
            c[k] = None
    _enrich(cells, workdir)
    target = TargetConfig(**index["target"]) if index.get("target") else None
    card = grade(cells, target=target, tier=index.get("tier"))
    card["config"] = battery_spec()
    card["workdir"] = os.path.abspath(workdir)
    with open(os.path.join(workdir, "reportcard.json"), "w") as f:
        json.dump(card, f, indent=2)
    with open(os.path.join(workdir, "reportcard.md"), "w") as f:
        f.write(render_reportcard_md(card) + "\n")
    print(f"[benchmark-grade] {workdir}: overall {card['overall']} "
          f"({', '.join(f'{ax}={a['grade']}' for ax, a in card['axes'].items())})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m attacker --benchmark",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tier", choices=sorted(TIER_BUDGETS),
                    help="quick = 2 defenses x 2 attacks at small budgets; "
                         "full = the whole benchmark.v1 matrix")
    ap.add_argument("--target-config", required=True,
                    help="TargetConfig JSON (model/split/corpus/device/...)")
    ap.add_argument("--workdir", required=True,
                    help="output directory (captures/, artifacts/, "
                         "scorecard.json, reportcard.{json,md}, "
                         "benchmark_cells.json)")
    ap.add_argument("--evidence", action="store_true",
                    help="seal every cell with an attacker.evidence chain")
    args = ap.parse_args(argv)
    try:
        target = TargetConfig.from_json(args.target_config)
    except ValueError as e:
        raise SystemExit(f"[benchmark] {e}") from None
    return run_battery(args.tier, target, args.workdir,
                       evidence=args.evidence)


def grade_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m attacker --benchmark-grade",
        description="Re-grade an existing benchmark workdir from "
                    "benchmark_cells.json + artifacts (torch-free).")
    ap.add_argument("workdir", help="benchmark workdir")
    args = ap.parse_args(argv)
    try:
        return regrade(args.workdir)
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"[benchmark-grade] {e}") from None


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile
    print("benchmark battery spec (pure python):")
    from .attacks import ATTACKS
    from .splitbench import _DEFENSES as SB_DEFENSES, BenchConfig
    spec = battery_spec()
    check("battery spec is JSON-able and versioned",
          json.dumps(spec) and spec["battery_version"] == BATTERY_VERSION)
    check("every battery attack is registered with a valid axis",
          all(a.attack in ATTACKS and a.axis in AXES
              for a in BATTERY_ATTACKS))
    check("every battery defense is a splitbench defense",
          all(d.name in SB_DEFENSES for d in BATTERY_DEFENSES))
    bench_fields = {f.name for f in dataclasses.fields(BenchConfig)}
    check("defense knobs are real splitbench config fields",
          all(k in bench_fields
              for d in BATTERY_DEFENSES for k, _ in d.knobs))
    check("quick tier is a strict subset of full",
          [a.attack for a in BATTERY_ATTACKS if a.quick]
          == ["seam-inversion", "label-scoring", "gpu-memory"]
          and [d.name for d in BATTERY_DEFENSES if d.quick]
          == ["none", "mid-lite"])

    print("target config validation:")
    check("defaults validate", TargetConfig().validate().model == "gpt2")
    for kw, frag in (({"mode": "inference"}, "training-seam"),
                     ({"model": ""}, "model"),
                     ({"device": "tpu"}, "device"),
                     ({"split": "0"}, "boundary"),
                     ({"property_token": -1}, "property_token")):
        try:
            TargetConfig(**kw).validate()
            check(f"target drift {kw} raises", False)
        except ValueError as e:
            check(f"target drift {kw} raises", frag in str(e))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.json")
        with open(p, "w") as f:
            json.dump({"model": "gpt2", "split": "4", "corpus_file": "w.txt"}, f)
        check("from_json round-trips", TargetConfig.from_json(p).split == "4")
        with open(p, "w") as f:
            json.dump({"modle": "gpt2"}, f)
        try:
            TargetConfig.from_json(p)
            check("unknown target key raises naming it", False)
        except ValueError as e:
            check("unknown target key raises naming it", "modle" in str(e))

    print("grade() on synthetic cells:")
    def acell(axis, defense, verdict, excess=None, status="ok"):
        c = _cell(f"{axis}:{defense}:x", "attack", axis, defense, "x", "arm")
        c["status"] = status
        c["verdict"] = verdict
        c["excess_pp"] = excess
        return c

    def capcell(defense, loss):
        c = _cell(f"capture:{defense}", "capture", None, defense,
                  "splitbench", "finetune")
        c["status"] = "ok"
        c["final_loss"] = loss
        return c

    g = grade([acell("S1", "none", "FAIL", 86.0),
               acell("S1", "mid-lite", "FAIL", 2.08),
               capcell("none", 4.0), capcell("mid-lite", 7.6)])
    check("all-FAIL axis grades OPEN, overall OPEN",
          g["axes"]["S1"]["grade"] == "OPEN" and g["overall"] == "OPEN")
    check("utility column carries final_loss per defense",
          g["utility"]["mid-lite"]["final_loss"] == 7.6)
    g = grade([acell("S1", "none", "FAIL", 86.0),
               acell("S1", "mid-lite", "PASS", 0.29)])
    check("FAIL on none + PASS defense grades AT-RISK",
          g["axes"]["S1"]["grade"] == "AT-RISK"
          and g["axes"]["S1"]["best_defense"]["defense"] == "mid-lite")
    g = grade([acell("S1", "none", "PASS", 0.4),
               acell("S1", "mid-lite", "PASS", 0.3)])
    check("all-PASS axis grades CLOSED",
          g["axes"]["S1"]["grade"] == "CLOSED" and g["overall"] == "INCOMPLETE")
    g = grade([acell("S1", "none", None, status="error"),
               acell("S5", None, "INFO")])
    check("no gated cells grades UNMEASURED (INFO does not gate)",
          g["axes"]["S1"]["grade"] == "UNMEASURED"
          and g["axes"]["S5"]["grade"] == "UNMEASURED"
          and g["overall"] == "UNMEASURED")
    g = grade([acell("S1", "none", "PASS", 0.1),
               acell("S2", "none", "FAIL", 5.0),
               acell("S5", None, None, status="error")])
    check("overall = worst measured section (S5 unmeasured ignored)",
          g["overall"] == "OPEN" and g["axes"]["S5"]["grade"] == "UNMEASURED")

    print("driver with a fake runner (no subprocess, no torch):")
    with tempfile.TemporaryDirectory() as d:
        def fake_runner(cmd, timeout):
            out = cmd[cmd.index("--output") + 1]
            if "label-scoring" in cmd and "mid-lite" in out:
                summary = [{"experiment": "norm_based_scoring", "auc": 0.51,
                            "upper95_excess_over_majority_pp": 0.3,
                            "n_seeds": 3}]
            elif "label-scoring" in cmd:
                summary = [{"experiment": "norm_based_scoring", "auc": 0.83,
                            "upper95_excess_over_majority_pp": 33.0,
                            "n_seeds": 3}]
            elif "gpu-memory" in cmd:
                return 1, "error: no hardware probe tier available"
            elif "splitbench" in cmd[2]:
                summary = [{"captures_written": 4, "final_loss": 7.6,
                            "mean_loss": 8.1, "defense": "x"}]
            else:
                summary = [{"experiment": "vmi",
                            "upper95_excess_over_majority_pp": 55.0,
                            "n_seeds": 3}]
            art = {"schema": "dtraining.attacker.fake.v1",
                   "config": {"attack": "fake", "mode": "training"},
                   "threat_model": "fake", "provenance": {},
                   "results": [dict(summary[0])], "summary": summary}
            with open(out, "w") as f:
                json.dump(art, f)
            return 0, "ok"

        rc = run_battery("quick", TargetConfig(model="gpt2"), d,
                         runner=fake_runner)
        card = json.load(open(os.path.join(d, "reportcard.json")))
        check("fake quick battery returns 0 and writes the report card",
              rc == 0 and card["schema"] == REPORTCARD_SCHEMA
              and card["battery_version"] == BATTERY_VERSION)
        check("S1 OPEN (all FAIL), S9 AT-RISK (FAIL none + PASS mid-lite), "
              "S5 UNMEASURED (error cell)",
              card["axes"]["S1"]["grade"] == "OPEN"
              and card["axes"]["S9"]["grade"] == "AT-RISK"
              and card["axes"]["S5"]["grade"] == "UNMEASURED")
        check("overall is the worst measured section (OPEN)",
              card["overall"] == "OPEN")
        check("the failed gpu-memory cell is an error record, not a crash",
              card["axes"]["S5"]["n_error"] == 1
              and card["axes"]["S5"]["cells"][0]["status"] == "error")
        check("cell index + scorecard + markdown report written",
              all(os.path.isfile(os.path.join(d, n))
                  for n in (CELL_INDEX_NAME, "scorecard.json",
                            "scorecard.txt", "reportcard.md")))
        md = open(os.path.join(d, "reportcard.md")).read()
        check("markdown card carries grades, utility, caveat",
              all(s in md for s in ("# Environment report card",
                                    "OPEN", "AT-RISK", "## Utility",
                                    "final_loss", "Corpus caveat")))
        check("the fabricated error artifact feeds the scorecard not-shown",
              json.load(open(os.path.join(d, "scorecard.json")))
              ["not_shown"][0]["attack"] == "gpu-memory")
        check("regrade reproduces the card from the cell index",
              regrade(d) == 0
              and json.load(open(os.path.join(d, "reportcard.json")))
              ["overall"] == "OPEN")

        d2 = os.path.join(d, "failcaps")
        def dead_runner(cmd, timeout):
            return 1, "error: boom"
        rc2 = run_battery("quick", TargetConfig(model="gpt2"), d2,
                          runner=dead_runner)
        card2 = json.load(open(os.path.join(d2, "reportcard.json")))
        check("a dead environment yields all-error cells, never raises",
              rc2 == 0 and card2["overall"] == "UNMEASURED"
              and all(c["status"] == "error"
                      for c in json.load(open(os.path.join(
                          d2, CELL_INDEX_NAME)))["cells"]))
        check("skipped attack cells name the capture failure",
              any("skipped: capture cell failed" in (c["error"] or "")
                  for c in json.load(open(os.path.join(
                      d2, CELL_INDEX_NAME)))["cells"]))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
