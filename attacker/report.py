#!/usr/bin/env python3
"""Automatic execution reports for attack artifacts (pure stdlib, torch-free).

`render_report(artifact_path)` turns one attack artifact (the standard
schema/config/threat_model/provenance/results/summary JSON) plus its
evidence manifest (<artifact>.evidence.json, when the run was sealed with
--evidence) into one self-contained markdown report at
`<artifact>.report.md`:

  - header: schema, experiment, UTC window, host, node role (evidence
    manifest first, artifact provenance as fallback);
  - the run configuration as a table;
  - the threat-model statement (verbatim);
  - per-cell results (headline metric + excess-over-control, via the
    scorecard machinery) with error cells listed, never dropped;
  - the frozen-gate verdict interpretation (constants from
    attacker.gate.DEFAULT_GATE — the report restates the gate, it never
    redefines it);
  - evidence integrity: verify_evidence() result, the artifact's recorded
    vs recomputed sha256, run id, chain hash (degraded, noted section when
    the run carried no evidence chain);
  - footer: the reproduction command (the evidence manifest's argv) and
    the campaign corpus caveat (README.md convention: every attacker
    number ships with it).

Design rules (repo conventions): pure stdlib, torch-free, and render NEVER
raises — an unreadable artifact or a broken evidence chain degrades to a
noted section, not a traceback (report generation must be safe to bolt
onto any run, cf. evidence.py "never raises on provenance collection").

CLI (wired into python -m attacker):

    python -m attacker --report ART.json [ART2.json ...]   # writes .report.md
    python -m attacker --report ART.json --stdout          # prints instead
    python -m attacker --mode training --attack gpu-memory --toy --quick \
        --output out.json --report    # piggyback: report right after a run
"""

import argparse
import json
import os
import shlex

from . import gate as _gate_mod
from .evidence import file_sha256, verify_evidence
from .gate import DEFAULT_GATE
from .scorecard import _headline as _sc_headline, build_scorecard

CORPUS_CAVEAT = (
    "Corpus caveat: results apply only to the recorded model, corpus, attacker "
    "access and budget. Read the artifact's provenance and evidence tier; "
    "synthetic or incomplete runs do not establish deployment security.")

_VALUE_WIDTH = 72


def _fmt_value(v):
    """One-line config value for the markdown table (dicts compacted)."""
    if isinstance(v, (dict, list)):
        v = json.dumps(v, sort_keys=True)
    else:
        v = str(v)
    v = v.replace("\n", " ").replace("|", "\\|")
    return v if len(v) <= _VALUE_WIDTH else v[:_VALUE_WIDTH - 3] + "..."


def _fmt_num(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _section(fn, *args):
    """Run one section builder; a failing section degrades to a note,
    never takes the report down (the module-level never-raises rule)."""
    try:
        return fn(*args)
    except Exception as e:
        return [f"_(section unavailable: {type(e).__name__}: {e})_"]


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _header_lines(art, path, manifest):
    prov = art.get("provenance") or {}
    config = art.get("config") or {}
    experiment = config.get("attack") or config.get("target_kind") \
        or os.path.basename(str(path))
    lines = [f"# Execution report — `{experiment}`", ""]
    lines.append(f"- **schema**: `{art.get('schema', '?')}`")
    lines.append(f"- **artifact**: `{path}`")
    if manifest:
        window = f"{manifest.get('start_utc', '?')} → " \
                 f"{manifest.get('end_utc', '?')}"
        lines.append(f"- **UTC window**: {window}")
        lines.append(f"- **host**: {manifest.get('hostname', '?')} "
                     f"(platform: {manifest.get('platform', '?')})")
        lines.append(f"- **node role**: {manifest.get('node_role') or '-'}")
        lines.append(f"- **run id**: `{manifest.get('run_id', '?')}`")
    else:
        lines.append(f"- **UTC window**: "
                     f"{prov.get('timestamp_utc', '?')} (artifact "
                     "provenance; no evidence manifest)")
        lines.append(f"- **host**: {prov.get('hostname', '?')}")
        lines.append("- **node role**: - (no evidence manifest)")
    commit = prov.get("dtraining_commit")
    if not commit and manifest:
        commit = (manifest.get("git") or {}).get("commit")
    if commit:
        lines.append(f"- **code commit**: `{commit}`")
    return lines


def _config_lines(art):
    config = art.get("config") or {}
    lines = ["", "## Configuration", ""]
    if not config:
        return lines + ["_(no config block in the artifact)_"]
    lines += ["| key | value |", "|---|---|"]
    for key in sorted(config):
        lines.append(f"| `{key}` | {_fmt_value(config[key])} |")
    return lines


def _threat_model_lines(art):
    tm = art.get("threat_model")
    lines = ["", "## Threat model", ""]
    if isinstance(tm, str) and tm.strip():
        lines.append(tm.strip())
    else:
        lines.append("_(no threat_model statement in the artifact)_")
    interp = art.get("interpretation")
    if isinstance(interp, str) and interp.strip():
        lines += ["", "**Result interpretation (as recorded by the run):**",
                  "", interp.strip()]
    return lines


def _results_lines(art, card):
    lines = ["", "## Results", ""]
    rows = card["rows"]
    if not rows or all(not r["cell"] for r in rows):
        lines.append("_(no summary cells in the artifact)_")
    else:
        lines += ["| cell | headline | excess vs control | verdict |",
                  "|---|---|---|---|"]
        for r in rows:
            if not r["cell"]:
                continue
            hkey, hval = r["headline_metric"], r["headline_value"]
            if hkey is not None and hkey == r["excess_metric"]:
                hkey, hval = _sc_headline(
                    {k: v for k, v in r["cell"].items() if k != hkey})
            headline = ("-" if hkey is None else
                        f"`{hkey}` = **{_fmt_num(hval)}**")
            excess = ("-" if r["excess_pp"] is None else
                      f"{r['excess_pp']:+.3f} pp (`{r['excess_metric']}`)")
            verdict = r["verdict"] + (" ⚠" if r["warnings"] else "")
            label = r["experiment"]
            if len(rows) > 1:
                label += f" #{r['cell_index']}"
            lines.append(f"| {label} | {headline} | {excess} | {verdict} |")
        warned = [w for r in rows for w in r["warnings"]]
        for w in warned:
            lines.append(f"  - ⚠ {w}")
    ns = card["not_shown"]
    lines.append("")
    if ns:
        lines.append(f"**Error cells ({len(ns)}, excluded from verdicts, "
                     "never dropped):**")
        for rec in ns:
            lines.append(f"- `{rec['attack']}`: {rec['error']}")
    else:
        lines.append("**Error cells:** none.")
    return lines


def _gate_lines(card):
    g = DEFAULT_GATE
    z = _gate_mod.bonferroni_z(g)
    s = card["summary"]
    lines = ["", "## Gate verdicts (the frozen gate)", ""]
    lines.append(
        f"Gate `{g.version}`: a cell **PASS**es iff the Bonferroni-adjusted "
        f"Wilson upper-95 of the attack's best arm exceeds the "
        f"`{g.control}` control by at most **+{g.gate_pp} pp** "
        f"(α={g.alpha}, {g.n_trials} trials → z={z:.4f}); a verdict "
        f"requires ≥{g.n_seeds_required} seeds. **FAIL** = the attack beats "
        "the operational upper-bound threshold; this alone does not establish leakage. **PASS** = at "
        "or below the threshold with sufficient declared seeds (a zero-leak experiment still "
        "scores the control's own CI half-width — 'at floor', never "
        "'zero'); **INFO** = the cell speaks no excess-over-control "
        "metric, so the gate does not apply.")
    lines.append("")
    lines.append(f"This artifact: **{s['overall']}** — {s['n_pass']} PASS, "
                 f"{s['n_fail']} FAIL, {s['n_info']} INFO, "
                 f"{s['n_not_shown']} error cell(s) not shown.")
    worst = card["worst_cell"]
    if worst is not None:
        lines.append(f"Worst gated cell: `{worst['attack']}` excess "
                     f"{worst['excess_pp']:+.3f} pp → {worst['verdict']}.")
    return lines


def _evidence_lines(artifact_path):
    lines = ["", "## Evidence integrity", ""]
    manifest_path = str(artifact_path) + ".evidence.json"
    if not os.path.isfile(manifest_path):
        return lines + [
            f"_(degraded: no evidence manifest at `{manifest_path}` — the "
            "run was not sealed; re-run with `--evidence` for a "
            "tamper-evident chain)_",
            "",
            f"- artifact sha256 (recomputed now): `{file_sha256(artifact_path)}`",
        ]
    manifest = _load_json(manifest_path)
    ok, problems = verify_evidence(manifest_path)
    recorded = (manifest.get("artifact") or {}).get("sha256")
    recomputed = file_sha256(artifact_path) \
        if os.path.isfile(artifact_path) else None
    lines.append(f"- manifest: `{manifest_path}` "
                 f"(schema `{manifest.get('schema', '?')}`)")
    lines.append(f"- chain verification (`verify_evidence`): "
                 f"**{'OK — chain intact' if ok else 'FAILED'}** "
                 f"({manifest.get('event_count', '?')} events, chain hash "
                 f"`{str(manifest.get('chain_hash'))[:16]}…`)")
    for p in problems:
        lines.append(f"  - ✗ {p}")
    lines.append(f"- artifact sha256 recorded: `{recorded}`")
    lines.append(f"- artifact sha256 recomputed: `{recomputed}` "
                 f"{'(match)' if recorded == recomputed else '(MISMATCH)'}")
    return lines


def _footer_lines(manifest):
    lines = ["", "## Reproduction", ""]
    argv = (manifest or {}).get("argv")
    if argv:
        lines.append("```bash")
        lines.append(f"cd {shlex.quote((manifest or {}).get('cwd') or '.')}"
                     f" && {shlex.join([str(a) for a in argv])}")
        lines.append("```")
    else:
        lines.append("_(no argv recorded — re-run the command with "
                     "`--evidence` to capture the exact reproduction "
                     "command, host state, and package versions)_")
    lines += ["", "---", "", f"> {CORPUS_CAVEAT}", ""]
    return lines


def render_report(artifact_path):
    """Render the markdown execution report for one artifact. Never raises:
    an unreadable artifact still yields a (minimal, noted) report."""
    artifact_path = str(artifact_path)
    try:
        art = _load_json(artifact_path)
        if not isinstance(art, dict):
            raise ValueError("artifact is not a JSON object")
    except Exception as e:
        return (f"# Execution report — `{os.path.basename(artifact_path)}`\n\n"
                f"_(degraded: artifact unreadable: {e})_\n\n"
                f"> {CORPUS_CAVEAT}\n")
    manifest = None
    manifest_path = artifact_path + ".evidence.json"
    if os.path.isfile(manifest_path):
        try:
            manifest = _load_json(manifest_path)
        except Exception:
            manifest = None
    try:
        card = build_scorecard([artifact_path])
    except Exception as e:
        card = {"rows": [], "not_shown": [], "worst_cell": None,
                "summary": {"overall": f"unscored ({e})", "n_pass": 0,
                            "n_fail": 0, "n_info": 0, "n_not_shown": 0}}
    lines = []
    lines += _section(_header_lines, art, artifact_path, manifest)
    lines += _section(_config_lines, art)
    lines += _section(_threat_model_lines, art)
    lines += _section(_results_lines, art, card)
    lines += _section(_gate_lines, card)
    lines += _section(_evidence_lines, artifact_path)
    lines += _section(_footer_lines, manifest)
    return "\n".join(lines) + "\n"


def report_path_for(artifact_path):
    return str(artifact_path) + ".report.md"


def write_report(artifact_path):
    """Write `<artifact>.report.md`; returns the report path."""
    path = report_path_for(artifact_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_report(artifact_path))
    print(f"[report] wrote {path}")
    return path


def main(argv=None):
    """CLI for `python -m attacker --report ...` (torch-free)."""
    ap = argparse.ArgumentParser(
        prog="python -m attacker --report",
        description="Render markdown execution reports for attack "
                    "artifacts (+ evidence integrity when the run was "
                    "sealed with --evidence). Never fails on degraded "
                    "input: the report notes the degradation instead.")
    ap.add_argument("artifacts", nargs="+", metavar="ARTIFACT.json",
                    help="attack artifact JSONs to report on")
    ap.add_argument("--stdout", action="store_true",
                    help="print the report(s) instead of writing "
                         "<artifact>.report.md")
    args = ap.parse_args(argv)
    for path in args.artifacts:
        if args.stdout:
            print(render_report(path))
        else:
            write_report(path)
    return 0


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    import tempfile
    print("execution reports (pure python):")
    with tempfile.TemporaryDirectory() as d:
        art_path = os.path.join(d, "run.json")
        artifact = {
            "schema": "dtraining.attacker.selftest.v1",
            "config": {"attack": "selftest-attack", "mode": "training",
                       "budget": 64},
            "threat_model": "self-test threat model",
            "provenance": {"hostname": "selftest-host",
                           "timestamp_utc": "2026-01-01T00:00:00Z"},
            "results": [{"experiment": "selftest_attack",
                         "upper95_excess_over_majority_pp": 2.5}],
            "summary": [{"experiment": "selftest_attack",
                         "best_probe_top1_pct": 6.5,
                         "upper95_excess_over_majority_pp": 2.5,
                         "n_seeds": 3}]}
        with open(art_path, "w") as f:
            json.dump(artifact, f)

        md = render_report(art_path)
        check("report renders header/config/threat-model/results/gate",
              all(s in md for s in ("# Execution report",
                                    "selftest-attack",
                                    "## Configuration", "| `budget` | 64 |",
                                    "self-test threat model",
                                    "## Results", "best_probe_top1_pct",
                                    "+2.500 pp", "FAIL",
                                    "frozen-gate.v2", "z=2.5392")))
        check("missing evidence degrades to a noted section",
              "degraded: no evidence manifest" in md
              and "## Evidence integrity" in md)
        check("no argv recorded -> noted reproduction section",
              "no argv recorded" in md)
        check("the corpus caveat ships with the report",
              "Corpus caveat" in md and "evidence tier" in md)
        check("unsealed artifact still gets a recomputed sha256",
              "recomputed now" in md)

        from .evidence import EvidenceLogger
        sealed = os.path.join(d, "sealed.json")
        with open(sealed, "w") as f:
            json.dump(artifact, f)
        with EvidenceLogger(sealed, node_role="selftest/cpu") as ev:
            ev.event("cell_result", {"cell": 1})
        md2 = render_report(sealed)
        check("sealed run: chain verification OK + node role + argv",
              "OK — chain intact" in md2 and "selftest/cpu" in md2
              and "## Reproduction" in md2 and "```bash" in md2)
        check("recorded and recomputed sha256 match",
              "(match)" in md2 and "(MISMATCH)" not in md2)

        ev_path = sealed + ".evidence.jsonl"
        lines = open(ev_path).read().splitlines()
        rec = json.loads(lines[1])
        rec["payload"] = {"tampered": True}
        lines[1] = json.dumps(rec)
        with open(ev_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        md3 = render_report(sealed)
        check("a tampered chain is reported as FAILED with the problems",
              "FAILED" in md3 and "✗" in md3)

        check("write_report writes <artifact>.report.md",
              write_report(art_path) == art_path + ".report.md"
              and os.path.isfile(art_path + ".report.md"))

        for bad in (os.path.join(d, "missing.json"), d, __file__):
            out = render_report(bad)
            check(f"render_report({os.path.basename(bad)}) never raises "
                  "and notes the degradation",
                  isinstance(out, str) and "degraded" in out)

        check("CLI main writes reports and returns 0",
              main([art_path]) == 0)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
