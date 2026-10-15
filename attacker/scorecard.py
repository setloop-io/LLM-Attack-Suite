#!/usr/bin/env python3
"""Scorecard: aggregate attack artifact JSONs into one verdict per defense.

Reads any number of attack artifacts (the standard schema:
schema/config/threat_model/provenance/results/summary) and produces one
scorecard.json plus a plain-text report:

  - one row per summary cell: attack, mode, labels, headline metric,
    excess-vs-control when the cell speaks the gate, PASS/FAIL/INFO
    (INFO = the cell carries no excess-over-control field, so the gate
    does not apply); stored upper bounds are re-validated against the
    requested gate's significance level and multiplicity — recomputed
    from the stored sufficient statistics or refused, never silently
    re-labelled;
  - the overall worst cell (max excess over control);
  - a "not-shown" section listing cells whose artifacts carry error
    records — failed cells are excluded from verdicts, never silently
    dropped (rotation_lifetime solve-hardening convention).

Scorecard policy is versioned separately from the numeric gate (gate.py;
frozen-gate.v2 is unchanged). Under the default scorecard-policy.v3 a PASS
is meaningful only alongside a positive control — the same attacker family
must EXCEED the +1.0 pp gate threshold on an undefended reference target
(the positive-control battery, results/attacker-positive-controls-v1/) —
otherwise the PASS could be explained by an ineffective attack. A PASS
without a backing control is downgraded to INSUFFICIENT_EVIDENCE (fail
closed); FAIL/INSUFFICIENT_EVIDENCE semantics are otherwise unchanged.
Controls are linked per report via --positive-control CONTROL.json
(repeatable; matched on attack identity, dash/underscore-normalized) or per
artifact via a "positive_control" field in the cell or artifact config (a
path, or list of paths, relative to the artifact file). A control backs a
PASS iff some cell of the control artifact scores excess > gate.gate_pp —
re-validated/recomputed under the requested gate, never re-labelled.
--scorecard-policy scorecard-policy.v2 evaluates legacy bundles without
this requirement; policy names follow the gate-versioning rule (named
versions, never silent mutation).

Pure python, torch-free. CLI (wired into python -m attacker):

    python -m attacker --scorecard ART1.json ART2.json [--output scorecard.json]
        [--positive-control CONTROL.json] [--scorecard-policy POLICY]

Exit code is 1 when any cell FAILs the gate, 2 when evidence is incomplete
(which includes a PASS downgraded for lack of a positive control) — the
scorecard is the CI enforcement point for the frozen gate.
"""

import argparse
import json
import math
import hashlib
import os
from dataclasses import asdict

from .gate import DEFAULT_GATE, GateConfig, bonferroni_z
from .gate import verdict as _gate_verdict

SCHEMA = "dtraining.attacker.scorecard.v3"

SCORECARD_POLICY = "scorecard-policy.v3"
LEGACY_SCORECARD_POLICY = "scorecard-policy.v2"
SCORECARD_POLICIES = (SCORECARD_POLICY, LEGACY_SCORECARD_POLICY)

_HEADLINE_PREFERENCE = (
    "upper95_excess_over_majority_pp", "worst_bonferroni_upper95_pct",
    "best_probe_top1_pct", "k50_crossing", "max_safe_epoch",
    "sessions_to_k90", "ica_top1_mean", "within_epoch_top1_mean",
    "token_recovery_mean", "decode_mean", "joint_mean",
    "seq_accuracy", "token_set_precision_mean", "auc",
    "alignment_search_mean", "p75", "mean", "max")

_NON_HEADLINE_KEYS = frozenset({
    "experiment", "attack", "mode", "seed", "restart", "arm", "k", "epoch",
    "n_seeds", "n_cells", "n_epochs", "n_ok", "pairs_per_epoch",
    "leak_per_epoch", "budget", "known_eval_fraction", "best_probe_model",
    "best_probe_restart", "model", "target"})


def load_artifact(path):
    """Load one artifact, raising a descriptive ValueError on schema drift."""
    with open(path) as f:
        art = json.load(f)
    if not isinstance(art, dict):
        raise ValueError(f"{path}: artifact is not a JSON object")
    missing = [k for k in ("schema", "config", "results", "summary")
               if k not in art]
    if missing:
        raise ValueError(f"{path}: artifact missing {missing} (expected the "
                         f"attacker artifact schema)")
    return art


def _registry_meta():
    """{EXPERIMENT_ID: (attack_name, REQUIRES_LABELS)} over the installed
    registry; tolerant — an unavailable registry yields an empty map."""
    try:
        from .attacks import ATTACKS, load
    except Exception:
        return {}
    meta = {}
    for name in ATTACKS:
        try:
            mod = load(name)
            meta[mod.EXPERIMENT_ID] = (name, mod.REQUIRES_LABELS)
        except Exception:
            continue
    return meta


def _labels_text(value):
    if value is True:
        return "labeled"
    if value is False:
        return "label-free"
    return str(value)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _headline(cell):
    """(metric_key, value) for the row's headline metric, (None, None) if
    the cell carries no numeric metric."""
    for key in _HEADLINE_PREFERENCE:
        if _is_num(cell.get(key)):
            return key, cell[key]
    for key in sorted(cell):
        if key not in _NON_HEADLINE_KEYS and _is_num(cell[key]):
            return key, cell[key]
    return None, None


def _z_match(a, b):
    """Bonferroni z comparison tolerant of float serialization noise; a
    real significance/multiplicity change moves z far past rel_tol."""
    return _is_num(a) and _is_num(b) and math.isclose(a, b, rel_tol=1e-12)


def _rescored_excess(cell, gate):
    """(metric_key, excess_pp, notes) for the cell under the requested
    gate. Only explicitly upper-bound metrics have gate semantics.

    A stored upper bound is valid only under the significance level and
    multiplicity it was computed with (the Bonferroni z). When the
    requested gate's z differs, recompute the bound from the stored
    sufficient statistics (recovery/control/n); when the cell does not
    store them, raise a ValueError naming the incompatibility — never
    silently re-label a stored bound."""
    z_req = bonferroni_z(gate)
    values, notes = [], []
    if _is_num(cell.get("upper95_excess_over_majority_pp")):
        n_trials = cell.get("n_trials", 9)
        z_stored = (bonferroni_z(GateConfig(alpha=0.05,
                                            n_trials=n_trials))
                    if isinstance(n_trials, int)
                    and not isinstance(n_trials, bool) and n_trials >= 1
                    else None)
        if not _z_match(z_stored, z_req):
            raise ValueError(
                "stored upper95_excess_over_majority_pp was computed under "
                f"the frozen convention (alpha=0.05, n_trials={n_trials}, "
                f"z={z_stored}), incompatible with the requested gate "
                f"{gate.version} (alpha={gate.alpha}, "
                f"n_trials={gate.n_trials}, z={z_req}); the cell stores no "
                "recovery/control/n sufficient statistics to recompute "
                "from — refusing to re-label the stored bound")
        values.append(("upper95_excess_over_majority_pp",
                       cell["upper95_excess_over_majority_pp"]))
    sub = cell.get("gate")
    if isinstance(sub, dict) and sub.get("verdict") in ("PASS", "FAIL", "INSUFFICIENT_EVIDENCE") and _is_num(sub.get("excess_pp")):
        stored_control = sub.get("control_definition")
        if stored_control is not None and stored_control != gate.control:
            raise ValueError(
                f"stored gate verdict scores against control "
                f"{stored_control!r}, incompatible with the requested gate "
                f"{gate.version} control {gate.control!r} — refusing to "
                "re-label the stored bound")
        z_stored = sub.get("z")
        if _z_match(z_stored, z_req):
            values.append(("gate.excess_pp", sub["excess_pp"]))
        else:
            recovery, control, n = (sub.get("recovery_pct"),
                                    sub.get("control_pct"), sub.get("n"))
            sufficient = (_is_num(recovery) and 0 <= recovery <= 100
                          and _is_num(control) and 0 <= control <= 100
                          and isinstance(n, int)
                          and not isinstance(n, bool) and n > 0)
            if not sufficient:
                raise ValueError(
                    f"stored gate verdict ({sub.get('gate', '?')}, "
                    f"z={z_stored}) is incompatible with the requested "
                    f"gate {gate.version} (z={z_req}) and the cell stores "
                    "no recovery_pct/control_pct/n to recompute from — "
                    "refusing to re-label the stored bound")
            regated = _gate_verdict(recovery / 100.0, control / 100.0, n,
                                    config=gate)
            values.append(("gate.excess_pp", regated["excess_pp"]))
            notes.append(f"excess recomputed under {gate.version} "
                         f"(z={z_req:.4f}); stored bound was "
                         f"{sub.get('gate', '?')} (z={z_stored})")
    best = max(values, key=lambda pair: pair[1], default=(None, None))
    return best[0], best[1], notes


def _excess(cell, gate=DEFAULT_GATE):
    """(metric_key, excess_pp) under `gate`; see _rescored_excess."""
    key, value, _notes = _rescored_excess(cell, gate)
    return key, value


def _control_identities(art):
    """Attack identities a control artifact speaks for: config.attack plus
    every cell's experiment id."""
    ids = set()
    config = art.get("config") or {}
    if config.get("attack"):
        ids.add(str(config["attack"]))
    for section in ("summary", "results"):
        for cell in art.get(section) or []:
            if isinstance(cell, dict) and cell.get("experiment"):
                ids.add(str(cell["experiment"]))
    return ids


def load_positive_control(path, gate=DEFAULT_GATE):
    """Assess one positive-control artifact: an undefended-reference run of
    the attacker. It demonstrates attack competence iff some cell's excess
    over control — re-validated/recomputed under the requested gate, never
    re-labelled — EXCEEDS gate.gate_pp (i.e. the attacker would FAIL the
    gate when nothing defends the target). Never raises: an unreadable or
    unscoreable control simply fails closed with a detail string."""
    record = {"path": str(path), "origin": None, "attacks": [],
              "max_excess_pp": None, "competent": False, "detail": None}
    try:
        art = load_artifact(path)
    except (OSError, ValueError) as e:
        record["detail"] = f"control artifact unreadable: {e}"
        return record
    record["attacks"] = sorted(_control_identities(art))
    excesses, refused = [], []
    for section in ("summary", "results"):
        for cell in art.get(section) or []:
            if not isinstance(cell, dict):
                continue
            try:
                _key, excess, _notes = _rescored_excess(cell, gate)
            except ValueError as e:
                refused.append(str(e))
                continue
            if excess is not None:
                excesses.append(excess)
    if excesses:
        best = max(excesses)
        record["max_excess_pp"] = best
        record["competent"] = best > gate.gate_pp
        if not record["competent"]:
            record["detail"] = (
                f"best excess {best:+.3f} pp does not exceed "
                f"+{gate.gate_pp} pp — the attacker did not beat control "
                "even on the undefended reference")
    elif refused:
        record["detail"] = ("no stored bound re-validates under "
                            f"{gate.version}: " + "; ".join(refused))
    else:
        record["detail"] = ("no cell carries a gate-scorable excess over "
                            "control — cannot demonstrate attack competence")
    return record


def _explicit_control_links(cell, config):
    """Control paths asserted by the artifact author: the cell's (else the
    artifact config's) positive_control field — a path or list of paths,
    resolved relative to the artifact file by the caller."""
    raw = cell.get("positive_control")
    if raw is None:
        raw = config.get("positive_control")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [p for p in raw if isinstance(p, str)]
    return []


def _norm_identity(name):
    """Dash/underscore-normalized attack identity for family matching
    (registry names use dashes, experiment ids underscores)."""
    return str(name).replace("-", "_")


def _apply_positive_control_policy(rows, contexts, report_records, cache,
                                   gate):
    """scorecard-policy.v3 pass over the built rows: a PASS stands only when
    backed by a competent positive control — explicit cell/config links
    first (author-asserted), then report-level controls matched on attack
    family. Unbacked PASSes fail closed to INSUFFICIENT_EVIDENCE with the
    reason in the row's warnings. Non-PASS rows are untouched."""
    for row, (art_path, config) in zip(rows, contexts):
        if row["verdict"] != "PASS":
            continue
        candidates, seen = [], set()
        for link in _explicit_control_links(row["cell"], config):
            resolved = (link if os.path.isabs(link) else
                        os.path.join(os.path.dirname(str(art_path)), link))
            key = os.path.abspath(resolved)
            if key not in cache:
                cache[key] = load_positive_control(resolved, gate)
                cache[key]["origin"] = "artifact"
            if key not in seen:
                seen.add(key)
                candidates.append(cache[key])
        ids = {_norm_identity(row["experiment"]),
               _norm_identity(row["attack"])}
        for rec in report_records:
            if {_norm_identity(a) for a in rec["attacks"]} & ids:
                key = os.path.abspath(rec["path"])
                if key not in seen:
                    seen.add(key)
                    candidates.append(rec)
        backing = next((r for r in candidates if r["competent"]), None)
        if backing is not None:
            row["positive_control"] = {
                "status": "backed", "by": backing["path"],
                "control_max_excess_pp": backing["max_excess_pp"]}
            continue
        if not candidates:
            reason = (
                f"PASS downgraded to INSUFFICIENT_EVIDENCE under "
                f"{SCORECARD_POLICY}: no positive control linked — a PASS "
                "is meaningful only alongside a control artifact of the "
                "same attacker family on an undefended reference target "
                f"whose excess exceeds +{gate.gate_pp} pp (link via "
                "--positive-control or the cell's positive_control field)")
        else:
            details = "; ".join(f"{r['path']}: {r['detail']}"
                                for r in candidates)
            reason = (
                f"PASS downgraded to INSUFFICIENT_EVIDENCE under "
                f"{SCORECARD_POLICY}: no linked positive control "
                "demonstrates attack competence (its excess must exceed "
                f"+{gate.gate_pp} pp on the undefended reference) — "
                + details)
        row["verdict"] = "INSUFFICIENT_EVIDENCE"
        row["warnings"].append(reason)
        row["positive_control"] = {"status": "unbacked", "detail": reason}


def _rows_for_artifact(path, art, meta, gate):
    config = art.get("config") or {}
    cells = [c for c in (art.get("summary") or []) if isinstance(c, dict)]
    cells += [c for c in art.get("results", []) if isinstance(c, dict)
              and isinstance(c.get("gate"), dict) and c not in cells]
    if not cells:
        cells = [{}]
    rows = []
    for i, cell in enumerate(cells):
        experiment = cell.get("experiment") or config.get("attack") or "?"
        name, labels = meta.get(experiment, (None, None))
        try:
            ex_key, excess, notes = _rescored_excess(cell, gate)
        except ValueError as e:
            raise ValueError(f"{path} cell {i} ({experiment}): {e}") from e
        warnings = list(notes)
        if excess is not None:
            verdict = "PASS" if excess <= gate.gate_pp else "FAIL"
            nested = cell.get("gate") if isinstance(cell.get("gate"), dict) else {}
            n_seeds = cell.get("n_seeds", nested.get("n_seeds"))
            if not isinstance(n_seeds, int) or n_seeds < gate.n_seeds_required:
                if verdict == "PASS":
                    verdict = "INSUFFICIENT_EVIDENCE"
                warnings.append(f"n_seeds={n_seeds} < required "
                                f"{gate.n_seeds_required}")
        else:
            verdict = "INFO"
        if "upper95_excess_over_majority_pp" in cell and not _is_num(cell["upper95_excess_over_majority_pp"]):
            verdict, excess, ex_key = "ERROR", None, None
            warnings.append("upper-bound metric must be finite")
        if "error" in cell:
            verdict, excess, ex_key = "ERROR", None, None
        if cell.get("evidence_tier") == "scenario":
            verdict, excess, ex_key = "INFO", None, None
        hkey, hval = _headline(cell)
        rows.append({
            "attack": name or experiment,
            "experiment": experiment,
            "mode": config.get("mode"),
            "labels": (_labels_text(labels) if labels is not None
                       else "unknown"),
            "headline_metric": hkey,
            "headline_value": hval,
            "excess_metric": ex_key,
            "excess_pp": excess,
            "verdict": verdict,
            "warnings": warnings,
            "source": str(path),
            "cell_index": i,
            "cell": cell,
        })
    return rows


def _error_cells(art):
    out = []
    for section in ("results", "summary"):
        for rec in art.get(section) or []:
            if isinstance(rec, dict) and "error" in rec:
                out.append(rec)
    return out


def build_scorecard(paths, gate=DEFAULT_GATE, positive_controls=(),
                    policy=SCORECARD_POLICY):
    """Aggregate artifact JSONs into the scorecard dict.

    policy names the scorecard behavior version (never silently mutated):
    the default scorecard-policy.v3 downgrades any PASS that lacks a
    backing positive control (see the module docstring);
    scorecard-policy.v2 reproduces the legacy behavior for old bundles.
    positive_controls are report-level control artifacts backing PASS rows
    of the same attacker family."""
    if policy not in SCORECARD_POLICIES:
        raise ValueError(f"unknown scorecard policy {policy!r} (known: "
                         f"{', '.join(SCORECARD_POLICIES)})")
    meta = _registry_meta()
    rows, not_shown, contexts = [], [], []
    unique_paths, seen = [], set()
    for path in paths:
        art = load_artifact(path)
        fingerprint = hashlib.sha256(json.dumps(art, sort_keys=True).encode()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_paths.append(path)
        new_rows = _rows_for_artifact(path, art, meta, gate)
        rows.extend(new_rows)
        contexts.extend([(path, art.get("config") or {})] * len(new_rows))
        for rec in _error_cells(art):
            not_shown.append({
                "attack": rec.get("experiment")
                          or (art.get("config") or {}).get("attack") or "?",
                "source": str(path),
                "error": rec["error"],
                "cell": {k: v for k, v in rec.items() if k != "error"}})
    paths = unique_paths
    control_records = []
    if policy == SCORECARD_POLICY:
        cache = {}
        report_records = []
        for p in positive_controls or ():
            key = os.path.abspath(str(p))
            if key not in cache:
                cache[key] = load_positive_control(p, gate)
                cache[key]["origin"] = "report"
            report_records.append(cache[key])
        _apply_positive_control_policy(rows, contexts, report_records,
                                       cache, gate)
        control_records = list(cache.values())
    gated = [r for r in rows if r["excess_pp"] is not None]
    worst = max(gated, key=lambda r: r["excess_pp"], default=None)
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    n_fail = sum(1 for r in rows if r["verdict"] == "FAIL")
    n_info = sum(1 for r in rows if r["verdict"] == "INFO")
    incomplete = bool(not_shown) or any(r["verdict"] in ("ERROR", "INSUFFICIENT_EVIDENCE") for r in rows)
    overall = "FAIL" if n_fail else ("INCOMPLETE" if incomplete else ("PASS" if n_pass else "INFO"))
    gate_info = asdict(gate)
    gate_info["z"] = bonferroni_z(gate)
    return {
        "schema": SCHEMA,
        "gate": gate_info,
        "policy": policy,
        "artifacts": [str(p) for p in paths],
        "positive_controls": control_records,
        "rows": rows,
        "worst_cell": worst,
        "not_shown": not_shown,
        "summary": {"n_artifacts": len(paths), "n_rows": len(rows),
                    "n_pass": n_pass, "n_fail": n_fail, "n_info": n_info,
                    "n_not_shown": len(not_shown), "overall": overall},
    }


def _fmt_num(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def render_report(card):
    """The plain-text report."""
    g = card["gate"]
    lines = [
        f"ATTACKER SCORECARD — gate {g['version']} (Bonferroni z={g['z']:.4f}"
        f", alpha={g['alpha']}, {g['n_trials']} trials; gate: excess "
        f"<= +{g['gate_pp']} pp over {g['control']}; seeds "
        f">= {g['n_seeds_required']}); policy "
        f"{card.get('policy', SCORECARD_POLICY)}",
        "",
        f"artifacts: {card['summary']['n_artifacts']}",
        "",
        f"{'attack':21s}{'mode':10s}{'labels':11s}{'headline':44s}"
        f"{'excess_pp':10s}{'verdict':8s}source",
    ]
    for r in card["rows"]:
        headline = ("-" if r["headline_metric"] is None else
                    f"{r['headline_metric']}={_fmt_num(r['headline_value'])}")
        excess = ("-" if r["excess_pp"] is None else f"{r['excess_pp']:+.3f}")
        verdict = r["verdict"] + ("*" if r["warnings"] else "")
        lines.append(f"{r['attack']:21s}{str(r['mode']):10s}{r['labels']:11s}"
                     f"{headline:44s}{excess:10s}{verdict:8s}{r['source']}")
        for w in r["warnings"]:
            lines.append(f"  * {w}")
    lines.append("")
    worst = card["worst_cell"]
    if worst is None:
        lines.append("WORST CELL: none (no cell reports excess-vs-control)")
    else:
        lines.append(f"WORST CELL: {worst['attack']} ({worst['source']}) "
                     f"excess {worst['excess_pp']:+.3f} pp "
                     f"-> {worst['verdict']}")
    lines.append("")
    if card.get("policy") == SCORECARD_POLICY:
        lines.append(f"POSITIVE CONTROLS ({SCORECARD_POLICY}: a PASS stands "
                     "only alongside a linked control of the same attacker "
                     "family on an undefended reference, excess "
                     f"> +{g['gate_pp']} pp):")
        records = card.get("positive_controls") or []
        if not records:
            lines.append("  none linked (unbacked PASS cells are downgraded "
                         "to INSUFFICIENT_EVIDENCE)")
        for rec in records:
            attacks = ",".join(rec["attacks"]) or "?"
            if rec["competent"]:
                status = (f"backs PASS (max excess "
                          f"{rec['max_excess_pp']:+.3f} pp)")
            else:
                status = f"does not back PASS ({rec['detail']})"
            lines.append(f"  - {rec['path']}: attacks={attacks}; {status}")
        lines.append("")
    if card["not_shown"]:
        lines.append("NOT SHOWN (error cells, excluded from verdicts):")
        for ns in card["not_shown"]:
            lines.append(f"  - {ns['attack']} [{ns['source']}]: {ns['error']}"
                         f"  cell={json.dumps(ns['cell'], sort_keys=True)}")
    else:
        lines.append("NOT SHOWN: none (no error cells)")
    s = card["summary"]
    lines += ["",
              f"OVERALL: {s['overall']} ({s['n_pass']} PASS, {s['n_fail']} "
              f"FAIL, {s['n_info']} INFO, {s['n_not_shown']} not shown)"]
    return "\n".join(lines)


def write_scorecard(card, output):
    """Write scorecard.json (atomic-ish via .tmp rename) plus the text
    report next to it."""
    tmp = output + ".tmp"
    with open(tmp, "w") as f:
        json.dump(card, f, indent=2)
    os.replace(tmp, output)
    report_path = (output[:-len(".json")] + ".txt") if \
        output.endswith(".json") else output + ".txt"
    with open(report_path, "w") as f:
        f.write(render_report(card) + "\n")
    print(f"[scorecard] wrote {output} (+ {report_path} report)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m attacker --scorecard",
        description="Aggregate attack artifact JSONs into one scorecard + "
                    "text report. Exit code is 1 when any cell FAILs the "
                    "gate (CI enforcement point).")
    ap.add_argument("artifacts", nargs="+", metavar="ARTIFACT.json",
                    help="attack artifact JSONs to aggregate")
    ap.add_argument("--positive-control", action="append", default=[],
                    metavar="CONTROL.json",
                    help="positive-control artifact: the same attacker "
                         "family on an undefended reference target; under "
                         f"{SCORECARD_POLICY} a PASS stands only when a "
                         "linked control's excess exceeds the gate "
                         "threshold. Repeatable; backs PASS rows whose "
                         "attack identity matches the control's")
    ap.add_argument("--scorecard-policy", choices=SCORECARD_POLICIES,
                    default=SCORECARD_POLICY,
                    help="scorecard behavior version; "
                         f"{LEGACY_SCORECARD_POLICY} evaluates legacy "
                         "bundles without the positive-control requirement")
    ap.add_argument("--output", default=None,
                    help="scorecard JSON path; the text report is written "
                         "next to it and always printed")
    args = ap.parse_args(argv)
    card = build_scorecard(args.artifacts,
                           positive_controls=args.positive_control,
                           policy=args.scorecard_policy)
    print(render_report(card))
    if args.output:
        write_scorecard(card, args.output)
    return 1 if card["summary"]["n_fail"] else (2 if card["summary"]["overall"] == "INCOMPLETE" else 0)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile

    def artifact(experiment, config, results, summary):
        return {"schema": f"dtraining.attacker.{experiment}.v1",
                "config": {"attack": experiment, **config},
                "threat_model": "self-test", "provenance": {},
                "results": results, "summary": summary}

    print("scorecard aggregation (pure python):")
    with tempfile.TemporaryDirectory() as d:
        paths = []
        specs = [
            artifact("latent_probe", {"mode": "training"},
                     [{"experiment": "latent_probe", "top1_pct": 4.5}],
                     [{"experiment": "latent_probe",
                       "best_probe_top1_pct": 4.5, "n_seeds": 3,
                       "upper95_excess_over_majority_pp": 0.638}]),
            artifact("some_future_attack", {"mode": "inference"},
                     [],
                     [{"experiment": "some_future_attack",
                       "upper95_excess_over_majority_pp": 1.5,
                       "n_seeds": 1}]),
            artifact("ica_bss", {"mode": "training"},
                     [{"experiment": "ica_bss", "budget": 512,
                       "error": "error: torch is required"}],
                     [{"experiment": "ica_bss", "ica_top1_mean": 0.5}]),
        ]
        for i, art in enumerate(specs):
            p = os.path.join(d, f"art{i}.json")
            with open(p, "w") as f:
                json.dump(art, f)
            paths.append(p)

        card = build_scorecard(paths, policy=LEGACY_SCORECARD_POLICY)
        verdicts = {r["experiment"]: r["verdict"] for r in card["rows"]}
        check("gated cells get PASS/FAIL, ungated get INFO (legacy policy)",
              verdicts == {"latent_probe": "PASS",
                           "some_future_attack": "FAIL",
                           "ica_bss": "INFO"})
        row = {r["experiment"]: r for r in card["rows"]}
        check("labels resolved via the registry for known attacks",
              row["latent_probe"]["labels"] == "labeled"
              and row["ica_bss"]["labels"] == "label-free"
              and row["some_future_attack"]["labels"] == "unknown")
        check("thin seed count is flagged as a warning",
              row["some_future_attack"]["warnings"]
              == ["n_seeds=1 < required 3"])
        check("worst cell is the max excess",
              card["worst_cell"]["experiment"] == "some_future_attack"
              and card["worst_cell"]["excess_pp"] == 1.5)
        check("error cells land in not-shown, excluded from verdicts",
              len(card["not_shown"]) == 1
              and card["not_shown"][0]["attack"] == "ica_bss"
              and card["not_shown"][0]["cell"] == {"experiment": "ica_bss",
                                                  "budget": 512})
        check("overall is FAIL when any cell fails",
              card["summary"]["overall"] == "FAIL")

        control_path = os.path.join(d, "control.json")
        competent = _gate_verdict(0.20, 0.04, 8192, n_seeds=3)
        with open(control_path, "w") as f:
            json.dump(artifact("latent_probe", {"mode": "training"},
                               [{"experiment": "latent_probe",
                                 "n_seeds": 3, "gate": competent}], []), f)
        unbacked = build_scorecard([paths[0]])
        check("v3: PASS without a positive control fails closed",
              unbacked["rows"][0]["verdict"] == "INSUFFICIENT_EVIDENCE"
              and unbacked["rows"][0]["positive_control"]["status"]
              == "unbacked"
              and any("no positive control linked"
                      in w for w in unbacked["rows"][0]["warnings"])
              and unbacked["summary"]["overall"] == "INCOMPLETE")
        backed = build_scorecard([paths[0]], positive_controls=[control_path])
        check("v3: PASS backed by a competent control stays PASS",
              backed["rows"][0]["verdict"] == "PASS"
              and backed["rows"][0]["positive_control"]
              == {"status": "backed", "by": control_path,
                  "control_max_excess_pp": competent["excess_pp"]}
              and backed["summary"]["overall"] == "PASS"
              and backed["positive_controls"][0]["competent"] is True)
        weak_path = os.path.join(d, "weak-control.json")
        with open(weak_path, "w") as f:
            json.dump(artifact("latent_probe", {"mode": "training"},
                               [{"experiment": "latent_probe", "n_seeds": 3,
                                 "gate": _gate_verdict(0.043, 0.04, 8192,
                                                       n_seeds=3)}], []), f)
        weak = build_scorecard([paths[0]], positive_controls=[weak_path])
        check("v3: a control that does not exceed the threshold cannot back",
              weak["rows"][0]["verdict"] == "INSUFFICIENT_EVIDENCE"
              and weak["positive_controls"][0]["competent"] is False
              and any("demonstrates attack competence"
                      in w for w in weak["rows"][0]["warnings"]))
        check("v3: report renders the positive-control section",
              "POSITIVE CONTROLS" in render_report(backed)
              and "backs PASS" in render_report(backed))

        report = render_report(card)
        check("report carries verdict rows, worst cell, not-shown, overall",
              all(s in report for s in
                  ("ATTACKER SCORECARD", "latent-probe", "WORST CELL",
                   "NOT SHOWN", "error: torch is required",
                   "OVERALL: FAIL")))

        out = os.path.join(d, "scorecard.json")
        check("main() exit code trips on a failing cell",
              main(paths + ["--output", out]) == 1)
        written = json.load(open(out))
        check("scorecard.json round-trips",
              written["schema"] == SCHEMA
              and written["summary"]["n_rows"] == 3)
        check("text report written next to the JSON",
              os.path.exists(os.path.join(d, "scorecard.txt")))
        passing = build_scorecard([paths[0], paths[2]])
        check("exit code 2 when evidence is incomplete",
              main([paths[0], paths[2]]) == 2
              and passing["summary"]["overall"] == "INCOMPLETE")

        bad = os.path.join(d, "bad.json")
        with open(bad, "w") as f:
            json.dump({"results": []}, f)
        try:
            load_artifact(bad)
            check("schema drift in an artifact is a descriptive error", False)
        except ValueError as e:
            check("schema drift in an artifact is a descriptive error",
                  "missing" in str(e))

        from .gate import GateConfig
        rec = _gate_verdict(0.10, 0.04, 8192, n_seeds=3)
        gated = os.path.join(d, "gated.json")
        with open(gated, "w") as f:
            json.dump(artifact("some_future_attack", {"mode": "training"},
                               [{"experiment": "some_future_attack",
                                 "n_seeds": 3, "gate": rec}], []), f)
        same = build_scorecard([gated])
        check("rescoring under the identical gate reproduces the stored "
              "verdict",
              same["rows"][0]["verdict"] == "FAIL"
              and same["rows"][0]["excess_pp"] == rec["excess_pp"])
        strict = GateConfig(alpha=0.01, version="audit-gate")
        rescored = build_scorecard([gated], gate=strict)
        check("a different significance level recomputes from the stored "
              "sufficient statistics",
              rescored["rows"][0]["excess_pp"] == _gate_verdict(
                  0.10, 0.04, 8192, config=strict)["excess_pp"]
              and any("recomputed" in w
                      for w in rescored["rows"][0]["warnings"]))
        v1 = GateConfig(version="frozen-gate.v1")
        check("a re-named gate with identical statistics reuses the "
              "stored bound",
              build_scorecard([gated], gate=v1)["rows"][0]["excess_pp"]
              == rec["excess_pp"])
        try:
            build_scorecard([paths[0]], gate=strict)
            check("an incompatible stored bound without sufficient "
                  "statistics is refused", False)
        except ValueError as e:
            check("an incompatible stored bound without sufficient "
                  "statistics is refused", "refusing to re-label" in str(e))

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
