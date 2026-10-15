"""Unified attacker framework for the split-inference / split-training
privacy study.

Superset of the repo's historical attack scripts (probes/e9_eval.py,
probes/er_train_eval.py, split-training/rotation_lifetime.py,
dlgpp.py, e8_robustness.py, membership_inference.py, output_inversion.py,
sipit_inversion.py, trained_inversion.py — all left untouched) plus six new
attacker capabilities (alignment-search E-R8, active-cloud, subspace-joint,
ica-bss, leak-accumulation, max-effort).

Usage:
    python -m attacker --list-attacks
    python -m attacker --self-test
    python -m attacker --mode training --attack accumulation --help
    python -m attacker --mode inference --attack alignment-search --toy ...
"""
