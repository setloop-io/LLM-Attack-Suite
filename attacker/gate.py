#!/usr/bin/env python3
"""The frozen gate, promoted from attacks/latent_probe.py hard-coding to a
shared, retargetable scoring config (README.md "The metric
definition"; RESEARCH_FIRST_CLASS_ATTACKER.md Phase D).

Verdict rule (frozen since the latent-era campaign):

  excess = Bonferroni-adjusted Wilson upper-95 of the attack's best arm,
           minus the label-free majority control, in percentage points
  PASS   iff excess <= +1.0 pp and >=3 independent seeds are declared
  INSUFFICIENT_EVIDENCE when the numeric criterion holds without enough seeds
  FAIL   means the upper-bound threshold is exceeded, not proof of leakage

Constants: alpha = 0.05, 9 trials (3 model classes x 3 restarts) ->
z = 2.5392 (the paper's z=2.62, corrected per PR-101); >= 3 seeds per
verdict. Statistical floor: even a zero-leak experiment scores the
control's own binomial CI half-width (~+0.4-0.65 pp at the campaign's
8,192-40,960 eval rows, configuration-dependent) — "at floor" is
reported, never "zero".

Pure python (math/statistics only), torch-free. Any change to these
constants is a NEW gate version, explicitly named: retarget by
constructing a GateConfig, never by mutating DEFAULT_GATE.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist

GATE_VERSION = "frozen-gate.v2"


@dataclass(frozen=True)
class GateConfig:
    """One named scoring gate. Defaults are the frozen campaign gate."""
    alpha: float = 0.05
    n_trials: int = 9
    gate_pp: float = 1.0
    n_seeds_required: int = 3
    control: str = "label_free_majority"
    version: str = GATE_VERSION


DEFAULT_GATE = GateConfig()


def bonferroni_z(config=DEFAULT_GATE):
    """One-sided normal quantile at 1 - alpha/n_trials (2.5392 for the
    frozen gate)."""
    return NormalDist().inv_cdf(1.0 - config.alpha / max(1, config.n_trials))


def wilson_upper(correct, total, config=DEFAULT_GATE):
    """Bonferroni-adjusted Wilson upper confidence bound (fraction).
    Identical math to the frozen latent_probe.wilson_upper."""
    if total <= 0:
        return float("nan")
    z = bonferroni_z(config)
    p = correct / total
    den = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total
                           + z * z / (4.0 * total * total))
    return (center + radius) / den


def floor_pp(control, n, config=DEFAULT_GATE):
    """Statistical floor in pp: the Bonferroni-Wilson half-width of the
    control itself at n eval rows — what a zero-leak experiment scores."""
    correct = int(round(control * n))
    return 100.0 * (wilson_upper(correct, n, config) - control)


def verdict(recovery, control, n, config=DEFAULT_GATE, n_seeds=None):
    """Score one cell against the gate.

    recovery, control: fractions in [0, 1] (best-arm recovery, control rate);
    n: number of eval rows; n_seeds: optional seed count behind the number.

    Returns a dict with the upper bound, excess over control in pp, the
    floor analysis, and the PASS/FAIL verdict."""
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (recovery, control)):
        raise ValueError("recovery and control must be finite fractions in [0, 1]")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    correct = int(round(recovery * n))
    upper = wilson_upper(correct, n, config)
    excess = 100.0 * (upper - control)
    floor = floor_pp(control, n, config)
    return {
        "gate": config.version,
        "z": bonferroni_z(config),
        "n": n,
        "recovery_pct": 100.0 * recovery,
        "control_pct": 100.0 * control,
        "upper95_pct": 100.0 * upper,
        "excess_pp": excess,
        "gate_pp": config.gate_pp,
        "floor_pp": floor,
        "at_floor": excess <= floor + 1e-9,
        "control_definition": config.control,
        "n_seeds_required": config.n_seeds_required,
        "n_seeds": n_seeds,
        "seeds_sufficient": (None if n_seeds is None
                             else n_seeds >= config.n_seeds_required),
        "verdict": ("FAIL" if excess > config.gate_pp else
                    "PASS" if n_seeds is not None and n_seeds >= config.n_seeds_required
                    else "INSUFFICIENT_EVIDENCE"),
    }


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("frozen gate (pure python):")
    z = bonferroni_z()
    check("Bonferroni z reproduces the frozen 2.5392 (alpha=0.05, 9 trials)",
          round(z, 4) == 2.5392)
    check("DEFAULT_GATE carries the frozen constants",
          (DEFAULT_GATE.alpha, DEFAULT_GATE.n_trials, DEFAULT_GATE.gate_pp,
           DEFAULT_GATE.n_seeds_required, DEFAULT_GATE.control)
          == (0.05, 9, 1.0, 3, "label_free_majority"))

    from .attacks import latent_probe
    check("promoted Wilson bound is numerically identical to latent_probe's",
          all(wilson_upper(c, n) == latent_probe.wilson_upper(c, n, 0.05,
                                                            trials=9)
              for c, n in ((0, 100), (50, 100), (100, 100), (316, 8192))))

    values = [wilson_upper(c, 100) for c in (0, 50, 100)]
    check("Wilson upper bound monotonic in correct",
          0.0 < values[0] < values[1] < values[2] <= 1.0)
    excesses = [verdict(r, 0.04, 8192)["excess_pp"]
                for r in (0.04, 0.05, 0.10)]
    check("excess monotonic in recovery",
          excesses[0] < excesses[1] < excesses[2])
    check("floor shrinks with n",
          floor_pp(0.04, 8192) > floor_pp(0.04, 40960) > 0.0)
    check("floor sits in the campaign band at 8,192 rows",
          0.3 < floor_pp(0.03857, 8192) < 0.8)

    at_control = verdict(0.03857, 0.03857, 8192, n_seeds=3)
    check("recovery == control is at the floor and PASSes",
          at_control["at_floor"] and at_control["verdict"] == "PASS")
    blown = verdict(0.10, 0.03857, 40960, n_seeds=1)
    check("a real break FAILs the gate",
          blown["verdict"] == "FAIL" and blown["excess_pp"] > 1.0)
    check("seed-count requirement is reported",
          blown["seeds_sufficient"] is False
          and verdict(0.04, 0.04, 8192, n_seeds=3)["seeds_sufficient"] is True)

    custom = GateConfig(alpha=0.01, n_trials=1, gate_pp=2.0,
                        version="test-gate")
    retargeted = verdict(0.05, 0.04, 8192, config=custom)
    check("gate is retargetable via GateConfig",
          retargeted["gate"] == "test-gate"
          and retargeted["z"] != z and retargeted["gate_pp"] == 2.0)

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
