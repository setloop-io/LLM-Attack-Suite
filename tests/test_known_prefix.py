"""known-prefix (E-R3, scaffold randomization vs known-prefix accumulation
under a static W): torch-free machinery checks.

The module hard-requires torch (run() calls require_torch first; the toy
world in attacker/synthetic.py is torch-only), so the numeric toy tier is
covered by the torch env. This file pins everything that stays checkable
torch-free:

  * contract / registry / threat metadata,
  * the parser (defaults ARE the documented configuration),
  * fail-closed behaviour without torch: SystemExit naming the attack, no
    files written, no traceback — in-process and through the real CLI,
  * run()'s two arms with the numerical core scripted to known values:
    the fixed_scaffold arm (the committed break) crosses the K90
    threshold at the EXACT interpolated session count; the
    jittered_scaffold arm stays flat below threshold and never brackets
    the crossing within the session budget; the undefended baseline is
    decoded with the true W; solve failures journal structured "error:"
    records and the summary degrades to baseline=None gracefully.
"""

import json
import os
import subprocess
import sys

import pytest

import attacker.synthetic as synthetic
from attacker import attack_metadata
from attacker.attacks import ATTACKS, load, validate_contract
from attacker.attacks import known_prefix as kp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_TOP1 = 100.0
FIXED_TOP1 = {24: 40.0, 48: 80.0, 72: 95.0, 96: 99.0}
JITTERED_TOP1 = {24: 5.0, 48: 6.0, 72: 7.0, 96: 8.0}
EXPECTED_K90 = 48 + (90.0 - 80.0) / (95.0 - 80.0) * (72 - 48)


class _Idx:
    """randperm result: an opaque permutation handle carrying its length."""

    def __init__(self, n):
        self.n = n

    def __getitem__(self, sl):
        start = sl.start or 0
        stop = sl.stop if sl.stop is not None else self.n
        return _Idx(stop - start)


class _Mat:
    """Tensor stand-in: shape + provenance tag; slicing/indexing returns
    another _Mat so the accumulation dataflow executes."""

    def __init__(self, n, h=64, tag=None):
        self.shape = (n, h)
        self.tag = tag

    def __getitem__(self, key):
        if isinstance(key, _Idx):
            return _Mat(key.n, self.shape[1], self.tag)
        if isinstance(key, slice):
            start = key.start or 0
            if start < 0:
                start += self.shape[0]
            stop = key.stop if key.stop is not None else self.shape[0]
            return _Mat(stop - start, self.shape[1], self.tag)
        return _Mat(1, self.shape[1], self.tag)

    def double(self):
        return self


class _W:
    """solve_w result stand-in carrying (arm, n_pairs)."""

    def __init__(self, tag):
        self.tag = tag


class _FakeTorch:
    """Only the torch surface known_prefix.run touches."""

    class Generator:
        def manual_seed(self, seed):
            return self

    @staticmethod
    def randperm(n, generator=None):
        return _Idx(n)

    @staticmethod
    def cat(mats):
        tags = {m.tag for m in mats}
        return _Mat(sum(m.shape[0] for m in mats), mats[0].shape[1],
                    tags.pop() if len(tags) == 1 else "cat")


def _world():
    def wire(h, t):
        return _Mat(h.shape[0], h.shape[1], ("wire", h.tag))

    return {"n_epochs": 1, "Ws": [_Mat(64, 64, "true_w")],
            "public_h": _Mat(4096, 64), "public_tok": object(),
            "victim_h": _Mat(512, 64), "victim_tok": object(),
            "wire": wire}


def _patch_toy(monkeypatch, solve=None):
    """Run kp.run() torch-free: no-op the torch guard, stand in for torch
    and the toy world, script the numerical core to known values. The fake
    wire tags rows by whether jitter_rows scrambled them, so the fixed and
    jittered arms stay attributable through torch.cat."""
    monkeypatch.setattr(kp, "require_torch", lambda name: None)
    monkeypatch.setattr(kp, "torch", _FakeTorch)
    monkeypatch.setattr(synthetic, "make_toy_world", lambda **kw: _world())
    monkeypatch.setattr(kp, "jitter_rows",
                        lambda h, pad_scale, seed: _Mat(*h.shape,
                                                        tag="jittered"))
    if solve is None:
        def solve(h, hw):
            arm = "jittered" if hw.tag == ("wire", "jittered") else "fixed"
            return _W((arm, hw.shape[0])), "lstsq+svd"
    monkeypatch.setattr(kp, "solve_w", solve)

    def recovery(h_wire, w_hat, ref_h, ref_tok, victim_tok):
        if w_hat.tag == "true_w":
            return BASELINE_TOP1
        arm, n = w_hat.tag
        return (FIXED_TOP1 if arm == "fixed" else JITTERED_TOP1)[n]

    monkeypatch.setattr(kp, "recovery_with_what_nn", recovery)


def _run(args_list, tmp_path, name="er3.json"):
    out = tmp_path / name
    args = kp.build_parser().parse_args([*args_list, "--output", str(out)])
    args.mode = "inference"
    assert kp.run(args) == 0
    art = json.loads(out.read_text())
    journal = [json.loads(ln) for ln in
               (tmp_path / (name + ".jsonl")).read_text().splitlines()]
    return art, journal


def test_contract_and_registry():
    assert kp.EXPERIMENT_ID == "er3_known_prefix"
    assert kp.MODES == ("training", "inference")
    assert kp.REQUIRES_LABELS == "partial (known system-prompt prefix)"
    assert "E-R3" in kp.DESCRIPTION
    assert validate_contract(kp, "known-prefix") is kp
    assert ATTACKS["known-prefix"] == "attacker.attacks.known_prefix"
    assert load("known-prefix") is kp
    md = attack_metadata.metadata_for("known-prefix")
    assert set(md["owasp_llm_2026"]) == {"LLM02", "LLM08"}
    assert md["surfaces"] == ("S1",)


def test_parser_defaults_are_the_documented_configuration():
    args = kp.build_parser().parse_args([])
    assert args.prefix_len == 32 and args.sessions == 8
    assert (args.hidden, args.seed) == (64, 1234)
    assert args.seeds == [0, 1, 2] and args.solve_seeds == [0, 1, 2]
    assert args.toy is False and args.quick is False


def test_parser_rejects_bad_prefix_len_and_help_works():
    with pytest.raises(SystemExit) as exc:
        kp.build_parser().parse_args(["--prefix-len", "long"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        kp.build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_run_fails_closed_without_torch(tmp_path):
    if kp.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "er3.json"
    args = kp.build_parser().parse_args(["--toy", "--quick", "--output",
                                         str(out)])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="requires torch"):
        kp.run(args)
    assert not out.exists()
    assert not (tmp_path / "er3.json.jsonl").exists()


def test_cli_fails_closed_without_torch(tmp_path):
    if kp.torch is not None:
        pytest.skip("torch present — the numeric toy tier runs instead")
    out = tmp_path / "er3.json"
    r = subprocess.run(
        [sys.executable, "-m", "attacker", "--mode", "inference",
         "--attack", "known-prefix", "--toy", "--quick", "--output",
         str(out)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "[er3_known_prefix] requires torch" in r.stderr
    assert "Traceback" not in r.stderr
    assert not out.exists()


def test_non_toy_run_is_a_descriptive_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(kp, "require_torch", lambda name: None)
    args = kp.build_parser().parse_args(["--output",
                                         str(tmp_path / "x.json")])
    args.mode = "inference"
    with pytest.raises(SystemExit, match="real-run driver"):
        kp.run(args)
    assert not (tmp_path / "x.json").exists()


def test_fixed_scaffold_crosses_k90_at_the_exact_interpolant(tmp_path,
                                                             monkeypatch):
    _patch_toy(monkeypatch)
    art, journal = _run(["--toy", "--quick"], tmp_path)
    assert art["schema"] == "dtraining.attacker.er3.v1"
    for k in ("config", "threat_model", "provenance", "results",
              "summary"):
        assert k in art
    assert art["config"]["sessions"] == 4
    assert art["config"]["prefix_len"] == 24
    fixed = [r for r in art["results"] if r["arm"] == "fixed_scaffold"]
    assert {(r["seed"], r["sessions"]) for r in fixed} == {
        (s, n) for s in (0, 1) for n in (1, 2, 3, 4)}
    for r in fixed:
        assert r["experiment"] == kp.EXPERIMENT_ID
        assert r["K"] == r["sessions"] * 24
        assert r["top1_mean"] == FIXED_TOP1[r["K"]]
    summ = [s for s in art["summary"] if s["arm"] == "fixed_scaffold"]
    assert {s["seed"] for s in summ} == {0, 1}
    for s in summ:
        assert s["baseline_top1"] == BASELINE_TOP1
        assert s["k90_threshold"] == 90.0
        assert s["sessions_to_k90"] == pytest.approx(EXPECTED_K90)
        assert s["crossing"]["k50_method"] == "linear_in_K"
        assert s["crossing"]["k50_bracket"] == [48, 72]
    assert len(journal) == len(art["results"])
    assert all(j["experiment"] == kp.EXPERIMENT_ID and "error" not in j
               for j in journal)


def test_jittered_scaffold_never_brackets_the_crossing(tmp_path,
                                                       monkeypatch):
    _patch_toy(monkeypatch)
    art, _ = _run(["--toy", "--quick"], tmp_path)
    jit = [r for r in art["results"] if r["arm"] == "jittered_scaffold"]
    assert len(jit) == 8
    for r in jit:
        assert r["top1_mean"] == JITTERED_TOP1[r["K"]]
    summ = [s for s in art["summary"] if s["arm"] == "jittered_scaffold"]
    assert {s["seed"] for s in summ} == {0, 1}
    for s in summ:
        assert s["sessions_to_k90"] is None
        assert s["crossing"]["k50_method"] == "not_bracketed"
        assert s["crossing"]["k50_bracket"] is None


def test_jitter_rows_is_applied_only_in_the_jittered_arm(tmp_path,
                                                         monkeypatch):
    _patch_toy(monkeypatch)
    calls = []
    monkeypatch.setattr(kp, "jitter_rows",
                        lambda h, pad_scale, seed:
                        (calls.append((h.shape[0], pad_scale, seed)),
                         _Mat(*h.shape, tag="jittered"))[1])
    _run(["--toy", "--quick"], tmp_path)
    assert len(calls) == 2 * 4
    assert all(n == 24 and ps == 1.0 for n, ps, _ in calls)
    assert sorted({sd for _, _, sd in calls}) == [1234 + 31337 + si
                                                  for si in range(4)]


def test_failed_solves_journal_errors_and_summary_degrades(tmp_path,
                                                           monkeypatch):
    _patch_toy(monkeypatch,
               solve=lambda h, hw: (None, "error: rank-deficient pairs"))
    art, journal = _run(["--toy", "--quick"], tmp_path)
    assert art["results"] == []
    errs = [j for j in journal if "error" in j]
    assert len(errs) == 16
    for j in errs:
        assert j["experiment"] == kp.EXPERIMENT_ID
        assert j["arm"] in ("fixed_scaffold", "jittered_scaffold")
        assert j["error"].startswith("error:")
        assert "rank-deficient pairs" in j["error"]
        assert "Traceback" not in j["error"]
    assert len(art["summary"]) == 4
    for s in art["summary"]:
        assert s["baseline_top1"] is None
        assert s["k90_threshold"] == 90.0
        assert s["sessions_to_k90"] is None
        assert s["crossing"]["k50_method"] == "not_bracketed"


def test_toy_run_is_deterministic(tmp_path, monkeypatch):
    _patch_toy(monkeypatch)
    a, _ = _run(["--toy", "--quick"], tmp_path, name="a.json")
    b, _ = _run(["--toy", "--quick"], tmp_path, name="b.json")
    assert a["results"] == b["results"] and a["summary"] == b["summary"]
    assert a["config"] == b["config"]
