# Contributing to LLM-Attack Suite

LLM-Attack Suite is a modular adversarial attack suite for privacy and security
evaluation of LLM systems. Contributions are welcome. Existing software
identifiers and frozen releases retain the earlier LLM-Attacker name; see the
[naming and prior-work note](README.md) for the distinction from the
autonomous-driving framework of Mei et al.

## How to contribute

1. **Start a conversation first.** Open an issue (or email the maintainers,
   see below) describing the attack module, defence, evaluation protocol or
   fix you have in mind. This keeps new work compatible with the frozen
   experimental comparisons.
2. **Fork and branch.** Create a branch from `main` with a short descriptive
   name, and open a pull request when the work is ready for review.
3. **Keep evidence semantics intact.** Each module declares its supported
   modes, input requirements, threat metadata and execution function, and
   writes the common run record (`schema`, `config`, `threat_model`,
   `provenance`, `results`, `summary`). New modules must follow the same
   input/output contract so they can join frozen comparisons.
4. **Add or extend tests.** Run the test suite (`python tests/run_suite.py`
   or `pytest`) before opening the pull request. The synthetic qualification
   tier needs no deep-learning stack, so new modules are expected to ship with
   their deterministic fixture and self-test.
5. **No fabricated results.** Report measured outcomes, including negative
   and protected results. Do not edit retained evidence records; new versions
   of an attack set or scoring policy get new version names.

## What makes a good contribution

- New attack or defence modules under the declared threat metadata.
- Controls: a module is much stronger with a positive and a negative control.
- Documentation and reproduction notes for anything you add.
- Bug fixes with a failing test that the fix makes pass.

## Conduct and licensing

- Be respectful and constructive; security research here is dual-use, so keep
  examples and targets synthetic or author-controlled.
- By contributing, you agree that your contributions are licensed under the
  project's [MIT licence](LICENSE).

## Contact

Questions or larger proposals: open an issue, or email
`geo.politis@gmail.com` / `evanz.pappas@gmail.com`.
