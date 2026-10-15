"""solve_primitives (attacker/solve_primitives.py): the pure-python section
(ratchet chain, stagger schedule, partition arithmetic, crossing_k) against
frozen fixtures and independent re-derivations; the solve_w fail-closed
control flow with the numerical backend stubbed; upstream() best-effort
cross-check. The tensor numerics themselves are torch-tier (skipped here)."""

import hashlib
import math
import runpy
import types
import warnings

import pytest

from attacker import solve_primitives as sp
from attacker.solve_primitives import (FIXTURE_CROSSING_CURVE,
                                       FIXTURE_RATCHET_12345,
                                       FIXTURE_STAGGER_S4_P8,
                                       K50_THRESHOLD_PCT, RANDOM_REL_ERR,
                                       crossing_k, partition_pairs,
                                       ratchet_secret, ratchet_seed, solve_w,
                                       stagger_rotating_blocks, upstream)


def test_ratchet_seed_matches_frozen_fixture():
    assert [ratchet_seed(12345, t) for t in range(4)] == FIXTURE_RATCHET_12345


def test_ratchet_seed_is_sha256_truncated_little_endian():
    for master, epoch in ((0, 0), (12345, 0), (12345, 3), (999983, 17)):
        want = int.from_bytes(
            hashlib.sha256(f"{master}:{epoch}".encode()).digest()[:8],
            "little")
        assert ratchet_seed(master, epoch) == want


def test_ratchet_seed_properties():
    assert ratchet_seed(12345, 0) == ratchet_seed(12345, 0)
    assert ratchet_seed(12346, 0) != ratchet_seed(12345, 0)
    assert ratchet_seed(12345, 1) != ratchet_seed(12345, 0)
    assert 0 <= ratchet_seed(12345, 0) < 2 ** 64


def test_stagger_matches_frozen_fixture():
    got = {e: stagger_rotating_blocks(e, 4, 8) for e in range(16)}
    assert got == FIXTURE_STAGGER_S4_P8


def test_stagger_hand_computed_small_case():
    got = {e: stagger_rotating_blocks(e, 2, 4) for e in range(8)}
    assert got == {0: [0], 1: [], 2: [1], 3: [],
                   4: [0], 5: [], 6: [1], 7: []}


def test_stagger_each_block_rotates_once_per_period():
    for n_blocks, period in ((4, 8), (2, 4), (3, 6), (5, 10)):
        counts = {i: 0 for i in range(n_blocks)}
        for e in range(period):
            for i in stagger_rotating_blocks(e, n_blocks, period):
                counts[i] += 1
        assert counts == {i: 1 for i in range(n_blocks)}


def test_stagger_no_window_rotates_everything():
    got = FIXTURE_STAGGER_S4_P8
    assert all(len(set(got[e]) | set(got[e + 1])) < 4 for e in range(15))


def test_partition_pairs_frozen_fixture():
    assert partition_pairs(250, 4) == [63, 63, 62, 62]


def test_partition_pairs_edge_cases():
    assert partition_pairs(8, 4) == [2, 2, 2, 2]
    assert partition_pairs(2, 4) == [1, 1, 0, 0]
    assert partition_pairs(0, 3) == [0, 0, 0]
    assert partition_pairs(1, 1) == [1]


def test_partition_pairs_invariants():
    for n_pairs, n_epochs in ((4096, 4), (251, 7), (1000, 3), (17, 5)):
        got = partition_pairs(n_pairs, n_epochs)
        assert len(got) == n_epochs
        assert sum(got) == n_pairs
        assert max(got) - min(got) <= 1
        base, rem = divmod(n_pairs, n_epochs)
        assert got == [base + (1 if e < rem else 0) for e in range(n_epochs)]


def test_crossing_k_frozen_curve_known_answer():
    curve = [{"K": k, "top1_mean": t} for k, t in FIXTURE_CROSSING_CURVE]
    cross = crossing_k(curve)
    assert cross["k50_method"] == "linear_in_K"
    assert cross["k50_bracket"] == [256, 512]
    assert abs(cross["k50_interpolated"] - 389.5652) <= 0.01
    assert abs(cross["k50_interpolated"] - (256 + 3072 / 23)) < 1e-9


def test_crossing_k_descending_curve_hand_computed():
    cross = crossing_k([{"K": 64, "top1_mean": 80.0},
                        {"K": 128, "top1_mean": 30.0}])
    assert abs(cross["k50_interpolated"] - 102.4) < 1e-9
    assert cross["k50_bracket"] == [64, 128]


def test_crossing_k_exact_hit_returns_the_k():
    cross = crossing_k([{"K": 64, "top1_mean": 12.0},
                        {"K": 256, "top1_mean": 50.0},
                        {"K": 512, "top1_mean": 61.0}])
    assert cross["k50_interpolated"] == 256.0
    assert cross["k50_bracket"] == [64, 256]


def test_crossing_k_sorts_unsorted_input():
    shuffled = [{"K": 512, "top1_mean": 61.0},
                {"K": 64, "top1_mean": 12.0},
                {"K": 4096, "top1_mean": 66.0},
                {"K": 256, "top1_mean": 38.0}]
    cross = crossing_k(shuffled)
    assert cross["k50_bracket"] == [256, 512]
    assert abs(cross["k50_interpolated"] - 389.5652) <= 0.01


def test_crossing_k_not_bracketed_cases():
    flat = crossing_k([{"K": 64, "top1_mean": 3.0},
                       {"K": 4096, "top1_mean": 3.0}])
    assert flat == {"k50_interpolated": None, "k50_bracket": None,
                    "k50_method": "not_bracketed"}
    plateau = crossing_k([{"K": 64, "top1_mean": 50.0},
                          {"K": 128, "top1_mean": 50.0}])
    assert plateau["k50_method"] == "not_bracketed"
    for curve in ([{"K": 1, "top1_mean": 90.0}, {"K": 2, "top1_mean": 95.0}],
                  [{"K": 1, "top1_mean": 10.0}, {"K": 2, "top1_mean": 20.0}]):
        assert crossing_k(curve)["k50_method"] == "not_bracketed"
    assert crossing_k([])["k50_method"] == "not_bracketed"
    assert crossing_k([{"K": 64, "top1_mean": 50.0}])["k50_method"] \
        == "not_bracketed"


def test_crossing_k_custom_threshold():
    cross = crossing_k([{"K": k, "top1_mean": t}
                        for k, t in FIXTURE_CROSSING_CURVE], threshold=60.0)
    assert abs(cross["k50_interpolated"] - (256 + 22 / 23 * 256)) < 1e-9
    assert cross["k50_bracket"] == [256, 512]


def test_constants():
    assert K50_THRESHOLD_PCT == 50.0
    assert RANDOM_REL_ERR == math.sqrt(2.0)


class _FakeTensor:
    """Duck-typed stand-in carrying the contiguous-fp64 discipline calls."""
    def double(self):
        return self

    def contiguous(self):
        return self


def test_solve_w_fail_closed_on_linalg_error(monkeypatch):
    def boom(a, b):
        raise RuntimeError("LAPACK gesdd failed to converge\nverbose detail")
    monkeypatch.setattr(sp, "torch",
                        types.SimpleNamespace(linalg=types.SimpleNamespace(
                            lstsq=boom)))
    w_hat, info = solve_w(_FakeTensor(), _FakeTensor())
    assert w_hat is None
    assert info == "error: LAPACK gesdd failed to converge"
    assert "verbose detail" not in info


def test_solve_w_success_path_and_solver_tag(monkeypatch):
    monkeypatch.setattr(sp, "torch",
                        types.SimpleNamespace(linalg=types.SimpleNamespace(
                            lstsq=lambda a, b:
                            types.SimpleNamespace(solution="SOL"))))
    monkeypatch.setattr(sp, "polar", lambda m: ("polar-of", m))
    w_hat, tag = solve_w(_FakeTensor(), _FakeTensor())
    assert w_hat == ("polar-of", "SOL")
    assert tag == "lstsq+svd"


def test_solve_w_polar_falls_back_to_eigh(monkeypatch, capsys):
    monkeypatch.setattr(sp, "torch",
                        types.SimpleNamespace(linalg=types.SimpleNamespace(
                            lstsq=lambda a, b:
                            types.SimpleNamespace(solution="SOL"))))

    def bad_svd(m):
        raise RuntimeError("svd did not converge")
    monkeypatch.setattr(sp, "polar", bad_svd)
    monkeypatch.setattr(sp, "_polar_eigh", lambda m: ("eigh-polar-of", m))
    w_hat, tag = solve_w(_FakeTensor(), _FakeTensor())
    assert w_hat == ("eigh-polar-of", "SOL")
    assert tag == "lstsq+eigh_polar_fallback"
    assert "polar SVD failed" in capsys.readouterr().out


def test_solve_w_both_operands_made_contiguous_fp64(monkeypatch):
    seen = []

    class TrackedTensor:
        def double(self):
            seen.append("double")
            return self

        def contiguous(self):
            seen.append("contiguous")
            return self

    monkeypatch.setattr(sp, "torch",
                        types.SimpleNamespace(linalg=types.SimpleNamespace(
                            lstsq=lambda a, b:
                            types.SimpleNamespace(solution="SOL"))))
    monkeypatch.setattr(sp, "polar", lambda m: m)
    solve_w(TrackedTensor(), TrackedTensor())
    assert seen == ["double", "contiguous", "double", "contiguous"]


def test_ratchet_secret_passes_the_derived_epoch_seed_to_make_secret():
    calls = []
    out = ratchet_secret(16, 12345, 3,
                         make_secret=lambda hidden, seed:
                         calls.append((hidden, seed)) or "W")
    assert out == "W"
    assert calls == [(16, ratchet_seed(12345, 3))]


def test_upstream_never_raises_and_returns_callables():
    up = upstream()
    assert isinstance(up, dict)
    assert all(callable(v) for v in up.values())
    assert up == {}


def test_upstream_exposes_what_it_finds(monkeypatch):
    import os
    import sys
    fake_rl = types.ModuleType("rotation_lifetime")
    fake_rl.solve_w = lambda *a: "sw"
    fake_rl.ratchet_seed = lambda *a: "rs"
    fake_rl.partition_pairs = lambda *a: "pp"
    fake_e8 = types.ModuleType("e8_robustness")
    fake_e8.polar = lambda *a: "p"
    fake_e8.crossing_k = lambda *a: "ck"
    monkeypatch.setitem(sys.modules, "rotation_lifetime", fake_rl)
    monkeypatch.setitem(sys.modules, "e8_robustness", fake_e8)
    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    try:
        up = upstream()
    finally:
        cand = os.path.join(os.path.dirname(sp.__file__), "..",
                            "split-training")
        while cand in sys.path:
            sys.path.remove(cand)
    assert up["rotation_lifetime.solve_w"] is fake_rl.solve_w
    assert up["rotation_lifetime.ratchet_seed"] is fake_rl.ratchet_seed
    assert up["rotation_lifetime.partition_pairs"] is fake_rl.partition_pairs
    assert up["e8_robustness.polar"] is fake_e8.polar
    assert up["e8_robustness.crossing_k"] is fake_e8.crossing_k


def test_self_test_passes():
    assert sp.self_test() == 0


def test_module_runs_standalone_torchless():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("attacker.solve_primitives", run_name="__main__")
    assert exc.value.code == 0
