# Measurement-methodology validation: 1 October 2026

The frozen selection produced **37 passes, no failures and no skips** on its
first execution. The 25 selected test functions expand to 37 parametrised test
cases. These cases verify seven specified properties of the measurement
methodology using known answers, independent numerical calculations, controlled
input transformations and injected faults. They are not 37 independent leakage
experiments or a full-suite test result.

## Observed results

| ID | Property and controlled intervention | Expected response | Observed response | Passed / cases |
|---|---|---|---|---:|
| MV1 | Known token and span answers; absent spans; wrong-position matches | Exact counts and span rates; absent spans remain missing; wrong-position matches receive no credit | All specified values matched, including 7/8 aligned tokens and zero wrong-position credit | 3/3 |
| MV2 | Independent Wilson arithmetic and numerical pins; one-token and threshold-boundary changes | Numerical agreement and the specified decision change | Bounds matched; 348/8,192 passed and 349/8,192 failed; equality passed | 4/4 |
| MV3 | Backed, absent, below-threshold and unreadable positive controls | Preserve the backed candidate pass; classify the other three as insufficient evidence | PASS for the backed case; INSUFFICIENT_EVIDENCE for all three other cases | 4/4 |
| MV4 | Change scorer-only truth; inject forbidden truth or invalid observation inputs | Byte-identical predictions after changing scorer truth; reject forbidden inputs | Prediction files were byte-identical; forbidden manifest and observation inputs were rejected | 2/2 |
| MV5 | Independent GRU recurrence/derivatives and AdamW arithmetic; exact head prefixes at three depths | Numerical agreement, causality, correct first-step decoding and nonzero embedding gradients | All specified comparisons passed; depths 1, 2 and 3 used the correct prefix with finite nonzero embedding gradients | 3/3 |
| MV6 | Constant and repeated fixed-seed intervals; duplicate perfectly correlated documents; four invalid matrices | Preserve the constant and correlated-case limits; repeat fixed-seed results; reject invalid matrices | Constant interval [0.25, 0.25]; correlated-case limits [0, 1] before and after duplication; deterministic replay and all four rejections passed | 6/6 |
| MV7 | Valid report and evidence chain; ten invalid report states resealed after mutation; three retained-hash corruptions | Accept intact cases and reject invalid states/corruption | Both intact cases accepted; all ten semantic faults and three retained-hash corruptions rejected | 15/15 |
| Total | 25 named functions; 37 parametrised cases | Every predeclared assertion satisfied | 37 passed, 0 failed, 0 skipped | 37/37 |

Each row maps to exact selected function IDs in [selection.json](selection.json).
The expanded case IDs and outcomes are retained in [run/results.json](run/results.json),
[run/junit.xml](run/junit.xml) and [run/collected-nodeids.txt](run/collected-nodeids.txt).
The unedited execution output is [run/pytest.stdout.txt](run/pytest.stdout.txt).

MV2 uses a 1e-5 absolute tolerance against the independently computed Wilson
bound with its four-decimal normal quantile, 1e-12 for full-precision pins and
1e-9 for the two campaign boundary scores. MV5's independent GRU and AdamW
comparisons use the frozen numerical criterion
`abs(candidate - reference) <= 1e-6 + 1e-5 * abs(reference)`.
The white-box prefix comparison uses PyTorch's dtype-specific
`torch.testing.assert_close` defaults; its gradient checks additionally require
finite, nonzero values and no gradients on the frozen victim parameters.

The dependence check is a controlled correlation stress case, not an estimate
of empirical confidence-interval coverage. Likewise, the positive-control row
validates the versioned scorecard eligibility rule; it does not replace measured
attack competence. The evidence row separates rejection of semantically invalid
reports even after resealing from detection of corruption against retained
hashes. These qualifications define what the interventions establish.

## Frozen source and execution

- Source commit: `6c071a68ae5134522d0a4cd3d7d0fa67acabdfa8`.
- Reviewed tag: `submission-candidate-20260922-r2`.
- Selection SHA-256: `a2ef4c6b144855e8f0cbb2112dc2f37e57488060a99124f379ecbc652c705d0e`.
- The selection and expected responses were written before this execution.
- A fresh `git archive` extracted the required source into a temporary directory;
  the working tree and its saved test reports were not used for execution.
- All 123 archived files were hashed before execution and checked afterwards;
  the archived source was unchanged.
- Python 3.14.6; PyTorch 2.14.0; Transformers 4.57.6; NumPy 2.5.3;
  pytest 9.1.1; local macOS ARM64 CPU execution.
- Pytest execution: 11.59 seconds; return code 0. This duration is an execution
  record, not a framework speed benchmark.
- No packages were installed, model weights downloaded or GPUs used. The
  neural-network checks instantiate small GPT-2/GRU fixtures from configurations.
- Offline Hugging Face settings, one-thread settings, Python hash seed and all
  effective commands are retained in [run/run.json](run/run.json) and
  [run/command.txt](run/command.txt).
- [run/packages.json](run/packages.json) records the installed distribution
  inventory. [run/source-files.sha256](run/source-files.sha256) records archived
  source hashes; its paths are relative to the pinned source snapshot.

The frozen repository's reporting plugin labels unmarked tests `torch-free`.
Several selected fixtures do use PyTorch, so that inherited marker-based tier
label is **not a statement about the actual dependencies**. The original logs
and plugin reports are preserved unchanged; use MV1–MV7 for the scientific
grouping and `run/run.json` for the effective environment.

This selection was defined by measurement properties, not selected afterwards
from the outcomes of a larger run. It excludes installation, licensing, command
syntax and registry checks. Its result makes no claim about those engineering
checks, all 64 attack modules or the entire repository test suite.

## Reproduction

Use a Python environment with the recorded dependencies and a local clone that
contains the pinned commit. From the repository root, run:

```sh
python paper/paper_b/validation/measurement-20261001/reproduce.py \
  --repository "$PWD" \
  --output /tmp/paper-b-measurement-independent-run
```

The output directory must not already exist. The script always reads the
specified commit, creates a fresh temporary source snapshot, preserves all test
outcomes and refuses to overwrite an existing result directory. It does not
install dependencies or modify the source assertions. Absolute paths, UUIDs,
timestamps and durations vary between runs; the selected properties and their
outcomes are the reproduction target.

Verify the retained supplement from this directory:

```sh
shasum -a 256 -c MANIFEST.sha256
```

`MANIFEST.sha256` covers every retained file other than itself. Keep its digest
with the manuscript release to anchor the package independently. This is a new
manuscript supplement, not an addition to the earlier immutable Hugging Face
release. Local temporary fixture weights and the source archive are not included;
the pinned source and reproduction script regenerate them.
