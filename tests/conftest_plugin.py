"""Pytest plugin: every run writes tests/last_run.{json,md}.

Per-test record: nodeid, module, tier, outcome, skip reason, duration.
Tiers: torch-free (default) / torch (slow-marked) / bdd (slow + defense
scenario layer). The terminal summary gains one line per tier so a skip
never hides silently. Writing failures warn but never break the run.
"""

import json
import os
import platform
import time

def _out_paths(session):
    root = str(session.config.rootdir)
    d = os.path.join(root, "tests") if os.path.isdir(
        os.path.join(root, "tests")) else root
    return (os.path.join(d, "last_run.json"), os.path.join(d, "last_run.md"))


def tier_of(item):
    """Classify a test item: bdd > torch > torch-free."""
    markers = {m.name for m in item.iter_markers()}
    nid = item.nodeid.lower()
    if "slow" in markers and ("defense" in nid or "scenario" in nid):
        return "bdd"
    if "slow" in markers:
        return "torch"
    return "torch-free"


def _env_info():
    try:
        import torch
        torch_present = True
    except ImportError:
        torch_present = False
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch_present": torch_present, "ts_utc":
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


class _Recorder:
    def __init__(self):
        self.tests = []

    def log(self, item, report):
        rec = {"nodeid": item.nodeid, "module": item.nodeid.split("::")[0],
               "tier": tier_of(item), "outcome": report.outcome,
               "duration_s": round(report.duration, 4)}
        if report.outcome == "skipped" and hasattr(report, "longrepr"):
            lr = report.longrepr
            rec["reason"] = (lr[2] if isinstance(lr, tuple) and len(lr) > 2
                             else str(lr))
        if report.outcome == "failed":
            rec["error"] = str(report.longrepr)[:500]
        self.tests.append(rec)


_REC = _Recorder()
_T0 = None
_ITEMS = {}


def pytest_collection_modifyitems(items):
    _ITEMS.update({item.nodeid: item for item in items})


def pytest_runtest_logreport(report):
    if report.when == "call":
        _REC.log(_ITEMS.get(report.nodeid, _FakeItem(report)),
                 report)
    elif report.when in ("setup", "teardown") and report.outcome in ("skipped", "failed"):
        _REC.log(_ITEMS.get(report.nodeid, _FakeItem(report)),
                 report)


class _FakeItem:
    def __init__(self, report):
        self.nodeid = report.nodeid

    def iter_markers(self):
        return iter(())


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    by_tier = {}
    for t in _REC.tests:
        by_tier.setdefault(t["tier"], []).append(t)
    tr = terminalreporter
    tr.write_line("")
    tr.write_line("— test tiers —")
    for tier in ("torch-free", "torch", "bdd"):
        cells = by_tier.get(tier, [])
        if not cells:
            continue
        p = sum(1 for c in cells if c["outcome"] == "passed")
        f = sum(1 for c in cells if c["outcome"] == "failed")
        s = sum(1 for c in cells if c["outcome"] == "skipped")
        line = f"  {tier}: {p} passed, {f} failed, {s} skipped"
        if s and tier in ("torch", "bdd"):
            line += "  (run in a torch env for real coverage)"
        tr.write_line(line)


def pytest_sessionfinish(session, exitstatus):
    tests = list(_REC.tests)
    for t in tests:
        if t["outcome"] == "skipped":
            reason = t.get("reason", "")
            if "torch" in reason and t["tier"] == "torch-free":
                t["tier"] = "torch"
            if "privacy_runtime" in reason:
                t["tier"] = "bdd"
    tiers = {}
    for t in tests:
        tt = tiers.setdefault(t["tier"], {"passed": 0, "failed": 0,
                                          "skipped": 0})
        tt[t["outcome"]] = tt.get(t["outcome"], 0) + 1
    n_fail = sum(1 for t in tests if t["outcome"] == "failed")
    record = {"schema": "llm_attacker.test_run.v1", "env": _env_info(),
              "totals": {"tests": len(tests),
                         "passed": sum(1 for t in tests
                                       if t["outcome"] == "passed"),
                         "failed": n_fail,
                         "skipped": sum(1 for t in tests
                                        if t["outcome"] == "skipped")},
              "tiers": tiers, "returncode": int(exitstatus),
              "verdict": "GREEN" if n_fail == 0 and exitstatus == 0 and tests else "RED",
              "tests": tests}
    try:
        json_path, md_path = _out_paths(session)
        with open(json_path, "w") as f:
            json.dump(record, f, indent=1)
        with open(md_path, "w") as f:
            f.write(render_markdown(record))
    except OSError as e:
        print(f"[test-report] could not write last_run files: {e}")


def render_markdown(record):
    t = record["totals"]
    lines = [f"# Test run — {record['verdict']}",
             "",
             f"- env: python {record['env']['python']} "
             f"({'torch present' if record['env']['torch_present'] else 'NO torch'}), "
             f"{record['env']['ts_utc']}",
             f"- totals: {t['passed']} passed, {t['failed']} failed, "
             f"{t['skipped']} skipped of {t['tests']}",
             ""]
    lines.append("## Per tier")
    lines.append("")
    lines.append("| tier | passed | failed | skipped |")
    lines.append("|---|---|---|---|")
    for tier, tt in sorted(record["tiers"].items()):
        lines.append(f"| {tier} | {tt.get('passed', 0)} | "
                     f"{tt.get('failed', 0)} | {tt.get('skipped', 0)} |")
    skips = {}
    for test in record["tests"]:
        if test["outcome"] == "skipped":
            skips.setdefault(test.get("reason", "?"), []).append(
                test["nodeid"])
    if skips:
        lines += ["", "## Skipped (by reason)", ""]
        for reason, ids in sorted(skips.items()):
            lines.append(f"- **{len(ids)}×** {reason}")
            for nid in ids[:5]:
                lines.append(f"  - `{nid}`")
            if len(ids) > 5:
                lines.append(f"  - … and {len(ids) - 5} more")
    fails = [t for t in record["tests"] if t["outcome"] == "failed"]
    if fails:
        lines += ["", "## Failed", ""]
        for t in fails:
            lines.append(f"- `{t['nodeid']}` — {t.get('error', '')[:200]}")
    slowest = sorted(record["tests"], key=lambda t: -t["duration_s"])[:10]
    lines += ["", "## Slowest 10", ""]
    for t in slowest:
        lines.append(f"- {t['duration_s']}s `{t['nodeid']}` ({t['outcome']})")
    return "\n".join(lines) + "\n"
