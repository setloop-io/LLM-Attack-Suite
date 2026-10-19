# LLM-Attack Suite v2.0

Version **v2.0** packages the branding and paper revision for **LLM-Attack Suite: A Unified Red-Team Framework for Privacy Evaluation of Split and Distributed LLM Systems**. The application was previously released as LLM-Attacker. This release changes branding, paper packaging and release metadata only: executable source and scientific evidence are unchanged, and no new experiments are reported. The Python package version remains `1.0.0`.

- [Combined paper PDF](https://huggingface.co/datasets/Setloop/LLM-Attack-Suite/blob/v2.0/paper/llm-attack-suite.pdf)
- [Self-contained arXiv source ZIP](https://huggingface.co/datasets/Setloop/LLM-Attack-Suite/resolve/v2.0/paper/llm-attack-suite-arxiv-source.zip)
- [Tagged evidence and dataset card](https://huggingface.co/datasets/Setloop/LLM-Attack-Suite/tree/v2.0)
- [Measurement-validation source and records](https://huggingface.co/datasets/Setloop/LLM-Attack-Suite/tree/v2.0/validation/measurement-20261001)

The repository is now [setloop-io/LLM-Attack-Suite](https://github.com/setloop-io/LLM-Attack-Suite), with this release at [tag v2.0](https://github.com/setloop-io/LLM-Attack-Suite/tree/v2.0). The earlier **v1.0** release remains available on [GitHub](https://github.com/setloop-io/LLM-Attack-Suite/tree/v1.0) and [Hugging Face](https://huggingface.co/datasets/Setloop/LLM-Attack-Suite/tree/v1.0) with its original filenames and contents.

**Naming and prior work.** Mei, Nie, Sun and Tian introduced **LLM-Attacker** for LLM-guided adversarial scenario generation in autonomous driving ([IEEE publication](https://doi.org/10.1109/TITS.2025.3578383)). We acknowledge that earlier use of the name and use **LLM-Attack Suite** for this application. The `llm-attacker` package and command, `attacker` imports, plugin and schema identifiers, environment-variable names and frozen evidence filenames retain their existing names for compatibility and reproducibility. These legacy identifiers refer to this suite. The Hugging Face repository has moved to `Setloop/LLM-Attack-Suite`; earlier repository links redirect to the new path. This URL correction preserves the existing v1.0 and v2.0 tags, paper bytes and evidence.

The current registry has **65 modules**. The paper's qualification matrix covers the **64-module reviewed snapshot**; the additional `secure-compute-audit` module is outside that matrix. Registered capability, synthetic checks and real-model attack effectiveness are reported separately.

The combined manuscript contains the concise paper followed by the full supplementary material, with one bibliography. The earlier-title preprint is available as [arXiv:2610.22722](https://arxiv.org/abs/2610.22722). The revised manuscript is prepared as a replacement for arXiv v2 and has not yet been submitted; the repository's v2.0 release does not update arXiv automatically. The selected arXiv distribution licence is arXiv's perpetual, non-exclusive licence.

Access at this release: the code repository is public, and Hugging Face files require login and manual approval. Repository access controls are independent of the source licence.

Before submitting the arXiv replacement, confirm prior AI systems and their tasks and the status/overlap of related papers, then complete the acknowledgement. EuroS&P also needs verified anonymous reviewer access to code and complete evidence. These submission actions remain open; this repository release does not complete them.

## Reproduce the frozen measurement validation

The evidence release includes the exact source archive used by the frozen selection, checked against all 123 retained source hashes. Its archive-based runner does not require the former Git history. After installing the research/test dependencies, download just this small package:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Setloop/LLM-Attack-Suite",
    repo_type="dataset", revision="v2.0",
    allow_patterns=["validation/measurement-20261001/**"],
    local_dir="evidence",
)
```

```bash
python evidence/validation/measurement-20261001/reproduce_from_snapshot.py \
  --output validation-output
```

The retained run has 37 passed cases and no failures or skips. A release-preparation rerun of that same frozen selection also passed all 37 cases; this checks measurement machinery and is not an additional attack-effectiveness experiment. These are retained v1.0 validation results, not a new v2.0 run. Full source snapshots for the reviewed framework and experiment-specific source archives are included with the evidence. Historical source identifiers inside frozen records are provenance labels; use the v2.0 release paths and unchanged source archives for reproduction.


LLM-Attack Suite is a modular collection of scripts and attack modules for
adversarial privacy and security evaluation of **LLM systems**. It covers
model, serving, protocol, application and agent attack surfaces. The paper
focuses on privacy evaluation of split and distributed deployments, where
parties exchange intermediate activations and gradients ("the wire"). The
framework executes attacks under declared observation contracts and retains
their configurations, predictions and measurements for independent checking.

It covers three surface families:

- **Split-model privacy**: seam/gradient inversion, membership inference,
  accumulated-observation and key-lifecycle attacks against rotated/obfuscated
  wire tensors.
- **Serving and protocol security**: traffic/timing side channels, GPU residue,
  tenant isolation, protocol integrity, supply chain, secure-compute admission.
- **Application and agent security**: prompt injection, RAG abuse, output-sink
  injection, and OWASP-style agentic batteries (goal hijack, tool misuse,
  confused deputy, memory poisoning, rogue lifecycle actions, and more).

> **Scope and honesty.** Availability in the attack registry indicates an
> *implementation*, not demonstrated real-target scientific coverage. Many
> modules run against synthetic worlds, fixtures, or local capture rigs; some
> require optional SDKs, specific captures, or hardware. A PASS/FAIL gate
> verdict is an operational threshold check, not a privacy certificate. Read
> each module's docstring, threat assumptions and actual input path —
> before interpreting a result. Local centroid/signature matching and
> direction-based gradient decoders are *adaptations*, not reproductions of
> the full published VMA, TAG, or LAMP algorithms.

## Requirements

- Python 3.12+
- The base package has **no mandatory runtime dependencies** and runs
  torch-less; model experiments require PyTorch, Transformers, NumPy, SciPy.

Optional extras:

| Extra | Contents | Needed for |
|---|---|---|
| `research` | torch, transformers, numpy, scipy | Real model runs (`attacker.experiment`, `splitbench`, torch-tier attacks) |
| `torch` | torch only | Minimal torch tier |
| `bisr-metrics` | rouge, nltk | `bisr-baseline` scoring metrics |
| `gpu` | pynvml | GPU telemetry in reports |
| `dev` | pytest | Running the test suite |

## Install and verify

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,research]'

python -m attacker --self-test        # torch-less framework self-check
python -m attacker --list-attacks     # registry: 65 modules
python -m pytest tests/ -q            # test suite
```

`requirements-research.lock` records an exact CPython 3.14/macOS arm64
validation environment for the research stack (install the project itself
separately with `pip install -e .`). Use compatible platform wheels on other
hosts and retain that run's package manifest.

GPU and external-defense integration tests require their corresponding
hardware/SDK and may skip locally; a skipped test is an unverified path.
`tests/run_suite.py` can merge per-environment reports and preserves a failure
in any environment even if the same test passes elsewhere.

## Quickstart

List the registry and run a synthetic (toy) attack end to end:

```bash
python -m attacker --list-attacks --mode training

python -m attacker --mode training --attack seam-inversion \
  --toy --quick --output results/toy.json

python -m attacker --scorecard results/toy.json --output results/scorecard.json
python -m attacker --coverage  results/toy.json --output results/coverage.json
python -m attacker --report    results/toy.json --stdout
```

Every attack module exposes torch-less `--help`:

```bash
python -m attacker --mode inference --attack seam-inversion --help
```

## CLI overview

```
python -m attacker --list-attacks [--mode training]
python -m attacker --self-test
python -m attacker --mode MODE --attack NAME [attack flags...]
python -m attacker --scorecard ART.json [ART2.json ...] [--output SC.json]
python -m attacker --coverage ART.json [ART2.json ...] [--output M.json]
python -m attacker --report ART.json [ART2.json ...] [--stdout]
python -m attacker --benchmark quick|full --target-config C --workdir D
python -m attacker --benchmark-grade WORKDIR       (re-grade, torch-free)
python -m attacker --recon --model M | --node N | --endpoint URL
```

- `--mode` selects the wire-capture schema (`attacker/captures.py`) and the
  available attack set: `training` adds gradient/membership/active-update
  attacks, `inference` adds serving/output-side attacks.
- `--scorecard` aggregates attack artifact JSONs into one scorecard plus a
  text report; exit code 1 when any cell fails the gate.
- `--coverage` reports which implementations actually ran (execution evidence,
  not a validated risk score).
- `--report` renders a markdown execution report; it also piggybacks on attack
  runs (`--attack NAME ... --output out.json --report` writes
  `out.json.report.md`).
- `--benchmark` runs the standard environment battery
  (`attacker/benchmark.py`, schema `benchmark.v1`) against a target config and
  writes an environment report card. `--benchmark-grade` re-grades an existing
  workdir without torch.
- `--recon` fingerprints a model, fleet node, or inference endpoint before
  attack selection (`attacker/recon.py`; all probes are timeout-guarded).

Supporting entry points:

```bash
python -m attacker.splitbench --help          # split-model capture rig
python -m attacker.defense_bundles --help     # external defense-artifact adapter
python -m attacker.security_assumptions --help
```

## The end-to-end experiment engine

`python -m attacker.experiment` is the dedicated protocol runner: actual
transformer forwards, split backpropagation, capture loading, attack decoding,
and scoring. A tiny random GPT-2 and synthetic token documents exercise the
complete pipeline without downloads:

```bash
python -m attacker.experiment --smoke --output results/smoke-v01
```

The defaults execute 30 cells: 3 seeds × 2 defenses × 5 mode/attack
combinations. Training runs VMI ridge inversion, WBI soft relaxation, and joint
forward/backward beam inversion; inference runs VMI and WBI against private
prefill activations. `--adaptive` enables noise-aware calibration for the
Gaussian defense. Each output directory must be new — reruns cannot silently
mix old and new cells. A `COMPLETE` smoke run establishes that the selected
path executed, **not** that a defense passed a security test.

For a real run, provide an immutable local pretrained GPT-2-family snapshot
and a curated JSONL corpus (one `{"id": ..., "text": ...}` per line, unique
IDs, no duplicate text, every document at least `--seq-len` tokens; minimum
nine documents):

```bash
python -m attacker.experiment \
  --model /absolute/path/to/pinned-gpt2-snapshot \
  --corpus /absolute/path/to/curated-documents.jsonl \
  --output results/run1 \
  --split 6 --seq-len 64 --train-steps 200 --decode-steps 16 \
  --attack-steps 200 --calibration-seqs 128 \
  --joint-beam 4 --joint-candidates 8 \
  --seeds 11 22 33 44 55 --sigma 0.1 --device cuda
```

The corpus is deterministically partitioned by document into public
calibration, private victim-training, and held-out utility partitions. The
public head/embedding snapshot is frozen **before** victim training; attack
arms never receive the victim's final private head. Supported protocol
defenses: `none` and additive `gaussian` (other `splitbench` defenses can be
exercised separately outside this protocol).

### Outputs and statistics

`experiment.json` records the protocol version, complete invocation, evidence
tier, per-seed/defense/mode/arm records, summary statistics, resource
measurements, and hashes/sizes of all component files. `partitions.json` binds
document IDs to text/token hashes. Per-cell directories contain wire tensors,
sidecars, canonical truth, and attack journals these include private scoring
truth and recovered tokens, so choose storage appropriate for the corpus.

Measurements include position-aligned token accuracy, exact sequence recovery,
token-set precision/recall, a public-calibration majority control, a seeded
document-derangement (shuffled-pair) diagnostic, paired attack-minus-control
and defense-minus-undefended effects, held-out next-token cross entropy and
perplexity before/after training, wall times, peak RSS/CUDA memory, and full
provenance hashes.

Summary confidence intervals use **seed-level** Student intervals with a
Bonferroni correction across the declared comparison families; tokens within a
sequence are never treated as independent replicates. Fewer than three seeds
produces `INSUFFICIENT_EVIDENCE`. Intervals are conditional on the fixed
corpus and do not estimate generalization across datasets or organizations.
A cell failure produces `FAILED`, preserves partial records, and exits
nonzero.

### Evidence verification

Every run seals an evidence chain (v2 binds artifacts, events, and manifest
metadata):

```bash
python -m attacker.experiment --verify results/smoke-v01
```

```python
import json
from pathlib import Path
from attacker.evidence import verify_evidence, file_sha256

root = Path('results/smoke-v01')
assert verify_evidence(root / 'experiment.json.evidence.json')[0]
run = json.loads((root / 'experiment.json').read_text())
assert run['status'] == 'COMPLETE'
for entry in run['files']:
    assert file_sha256(root / entry['path']) == entry['sha256']
```

The manifest's `manifest_sha256` can be stored independently and supplied as
the second argument to `verify_evidence`. Hashes detect changes against
retained anchors; they are **not** digital signatures or proof that an
experiment ran on the claimed host — someone able to rewrite the whole bundle
can recompute its hashes. Publish a separately anchored manifest digest with
any released bundle.

## Attack registry

65 modules, lazily imported. `--list-attacks` shows modes and label
requirements; each module's docstring carries its threat model and input
contract. Counts include collection/orchestration utilities; they are not 65
independently validated vulnerabilities.

**Token, label and latent reconstruction**

| Module | Modes | Summary |
|---|---|---|
| `seam-inversion` | T, I | VMI/WBI inversion of input tokens from forward seam activations; ridge + soft-relaxation arms, majority-control gates |
| `gradient-inversion` | T | DLG-style optimization on the training boundary; iDLG labels, direction/assignment arms (adaptations) |
| `output-inversion` | I | Decoder on cloud output features for served-token recovery |
| `vec2text-inversion` | T, I | Learned encoder-decoder generating victim text from seam rows, optional iterative correction |
| `latent-probe` | T | Coordinate and rotation-invariant token probes on release bundles |
| `latent-sensitivity` | T | Capture/chaff-identification/gauge/known-plaintext compromise sweeps |
| `latent-matching` | T | Centroid vocabulary + position-free signature matching |
| `label-scoring` | T | Gradient-norm property inference (AUC, TPR at low FPR) |
| `batch-label-recon` | T | First-batch token reconstruction from body/seam gradients |
| `ica-bss` | T, I | FastICA blind source separation of rotated mixtures (synthetic) |

**Rotation, key and accumulated-observation attacks**

| Module | Modes | Summary |
|---|---|---|
| `accumulation` | T, I | Known-plaintext accumulation within one key epoch; polar solve for the rotation |
| `known-prefix` | T, I | Known scaffold/prefix accumulation; jitter sensitivity |
| `stale-key` | T, I | Recovery from disclosed older rotation keys + optional fresh pairs |
| `seed-disclosure` | T | Reconstruct the initial encoder from a protocol-disclosed seed |
| `sharded` | T, I | Blockwise solves against sharded/staggered rotation schedules |
| `wire-eval` | T, I | Per-epoch forward/backward wire recovery with cross-epoch pooling control |
| `alignment-search` | T, I | Edit-distance/DTW realignment of jittered captures before solving |
| `subspace-joint` | T, I | Shared-subspace recovery across rotation epochs |
| `leak-accumulation` | T, I | Persistent partial key-row leakage accumulating across epochs |

**Membership and property inference**

| Module | Modes | Summary |
|---|---|---|
| `membership` | T | Logistic probes on document-level boundary features |
| `lira-membership` | T | LiRA/RMIA-style scoring on supplied reference statistics |

**Active participant attacks and model extraction**

| Module | Modes | Summary |
|---|---|---|
| `active-cloud` | T, I | Malicious-cloud crafted returned tensors/updates (offline planner) |
| `malicious-server` | T | Crafted body computation amplifying token signal in gradients |
| `malicious-client` | T, I | Crafted activations, poisoning, sybil collusion in simulated worlds |
| `body-extraction` | T, I | Query-driven student distillation of the server body (functional, not weight, recovery) |

**Hardware, timing and traffic side channels**

| Module | Modes | Summary |
|---|---|---|
| `gpu-memory` | T, I | Allocator residue, scrub checks, unified-memory contention timing |
| `kv-timing` | I | Prefix-cache/speculative-decoding timing membership channels |
| `wire-shape` | T, I | Token-count and phase inference from packet sizes/timestamps |
| `router-side-channel` | T, I | MoE routing/queue metadata leakage and linkability |

**Protocol, isolation, integrity and availability**

| Module | Modes | Summary |
|---|---|---|
| `protocol-integrity` | T, I | Replay/reorder/swap/desync/tamper battery on split frames |
| `multi-tenant-isolation` | T, I | Session, batching, cache, KV and stream isolation fixtures |
| `artifact-telemetry-leak` | T, I | Canary sweeps and permission/retention audits of persistent stores |
| `key-session-abuse` | T, I | Nonce reuse, rollback, downgrade, stale-key collusion lifecycle battery |
| `secure-compute-audit` | T, I | Digest-bound TEE/FHE admission battery with reference and process adapters |
| `resource-exhaustion` | T, I | Queue/cache/cost/memory/fairness stress in a simulated serving stack |
| `poisoning-backdoor` | T | Data/gradient/adapter/checkpoint/RAG poisoning with zero-poison controls |
| `supply-chain` | T, I | Artifact substitution/tamper/rollback/provenance verification battery |

**Reasoning exposure**

| Module | Modes | Summary |
|---|---|---|
| `reasoning-replay` | I | Portability/extraction/secret-scanning of provider reasoning blobs |
| `trace-inversion` | I | Hidden-trace synthesis from (problem, summary, answer) with verifier search |

**Generative-AI application security**

| Module | Modes | Summary |
|---|---|---|
| `prompt-injection` | I | Direct/indirect/encoded/adaptive injection scored by policy violations |
| `hidden-context-exposure` | I | Extraction of hidden application context, exact vs semantic recovery |
| `rag-vector-abuse` | I | Cross-tenant retrieval, poisoning, inversion, retention, citation abuse |
| `output-sink-injection` | I | Model-output-to-sink battery (HTML/SQL/shell/template/URL/path/code) |
| `misinformation-integrity` | I | Grounding/citation/calibration/staleness against an authoritative corpus |

**Agent security (OWASP ASI batteries)**

| Module | Modes | Summary |
|---|---|---|
| `tool-misuse` | I | Dangerous parameters, scope confusion, amplified effects |
| `agent-goal-hijack` | I | Goal redirection through untrusted tool/retrieval content |
| `human-agent-trust` | I | Displayed-approval vs executed-action mismatch |
| `memory-context-poisoning` | I | Persistent-memory poison redirecting a later session |
| `agentic-supply-chain` | I | Tampered tool/connector component changes behavior |
| `identity-privilege-abuse` | I | Forged principal/role/cross-session authority at the tool gateway |
| `cascading-failure` | I | Unvalidated values propagating through multi-hop pipelines |
| `unexpected-code-execution` | I | Unapproved or approval-swapped code digests at the gateway |
| `rogue-agent` | I | Unauthorized spawning, self-modification, persistence (simulated) |
| `insecure-agent-communication` | I | Spoofing/tampering/replay on a simulated agent bus |
| `cross-agent-deputy` | I | Confused-deputy authority forwarding between agents |

**Collection, orchestration, composition, and defense batteries**

| Module | Modes | Summary |
|---|---|---|
| `remote-fetch` | T, I | SSH capture collection from reachable fleet nodes |
| `mitm-proxy` | T, I | Bidirectional relay teeing traffic into capture journals |
| `max-effort` | T, I | Re-run a target attack across seeds/initializations/budgets |
| `privacy-composition` | T, I | Compose per-surface artifacts into advantage-vs-budget projections |
| `replica-integrity` | T, I | Replica collusion, K=2 abort, shared-key disclosure blast radius |
| `view-privacy` | T, I | Mixup de-mixing, coordinate-view collusion, tripwire accumulation |
| `representation-privacy` | T, I | Fresh coordinate/norm/Gram probes; phase-aware membership |
| `gradient-hardening` | T | Clipping/abort/noise/quantization replay against gradient decoding |
| `bisr-baseline` | T, I | Pilot-scale BiSR/SIP adaptation (development-only; not the published setting) |
| `two-seam-delegation` | T, I | Chained inversion across two seams (A/island/B delegation) |

`T` = training, `I` = inference. **Training** means applicability while a
victim learns (backward seam gradients and updates exist); **inference** means
applicability while a victim answers requests. An attacker training its own
decoder against inference observations does not make the victim scenario a
training attack.

## The splitbench capture rig

`attacker/splitbench.py` splits GPT-2, Mistral, and Qwen3 models at a
configured layer and drives local capture. Training sends the body a detached
wire tensor, receives its gradient, and backpropagates it through the client
head before the optimizer update; inference performs full re-forwards without
a production KV-cache stack. These are **local capture rigs** — they do not
establish behavior across separate hosts/processes or production serving.

Available `--defense` mechanisms: `none`, `gaussian`, `sparsify`, `mid-lite`,
`at-lite`, `santext`, `custext`, `snd-lite`, `trained-bottleneck` (learned MLP
bottleneck with a MINE-style penalty, GPU-only by default, optional
`--tb-cotrain`), `pca-compress` (compression-only control), `secret-mixup`,
`replica-median`, `quant`. These are local mechanisms/analogues, not full
published-method reproductions.

### Wire capture format

A wire capture is `wire_NNNNN.pt` plus a JSON sidecar:

| Mode | Required coordinates | Phases |
|---|---|---|
| Training | `session_id`, `mb_id`, `step`, `epoch` | `fwd`, `bwd` |
| Inference | `session_id`, `request_seq`, `position`, `epoch` | `prefill`, `decode` |

Training canonical tensors align with forward captures in step/microbatch
order. Inference `canonical.json` supports variable-length prompts and records
generated tokens; equal-length prompts additionally produce `canonical.pt`
(consumed by seam inversion). Surface providers validate capture schemas. The
live agent harness (`attacker/agent_harness.py`) also supports model-driven
planner experiments; toy outcomes and live planner outcomes are separate
evidence tiers and must stay separate.

### External defense artifacts

Real exported artifacts from a deployed defense framework enter the batteries
through one adapter contract: the `llm_attacker.defense_bundle.v1` envelope
(`attacker/defense_bundles.py`, CLI `python -m attacker.defense_bundles
validate|materialize`). The envelope adds provenance, component resolution,
and sha256 pinning around the batteries' frozen payload schemas. Adapter
acceptance means **format conformance only** — fixture-level positives do not
falsify deployed defenses.

## Scoring, gates and coverage semantics

- **`frozen-gate.v2`** preserves the historical numeric rule: the one-sided
  Bonferroni–Wilson upper bound minus majority control must be at most +1
  percentage point (`alpha=.05`, nine trials, `z=2.5391848`), and refuses PASS
  without at least three declared seeds. **FAIL means the upper bound crosses
  the operational threshold; it does not prove leakage** — a small sample can
  fail at zero observed attack advantage.
- **Scorecard v3** (schema `dtraining.attacker.scorecard.v3`, default policy
  `scorecard-policy.v3`) fails closed on vacuous PASSes: a PASS stands only
  alongside a linked positive-control artifact — the same attack family
  exceeding the +1.0 pp gate threshold on an undefended reference target —
  otherwise the PASS is downgraded to `INSUFFICIENT_EVIDENCE` (exit 2), since
  it could be explained by an ineffective attack. Link controls with
  `--positive-control CONTROL.json` (matched on attack identity) or a
  `positive_control` field in the artifact. `--scorecard-policy
  scorecard-policy.v2` evaluates legacy bundles without the requirement.
- **CLI exit codes**: 1 = any threshold failure, 2 = incomplete evidence,
  0 = passing or informational-only output. Always also inspect the overall
  status field.
- **Coverage** reports only which implementations ran; it never combines toy
  and real results into target-specific assurance, and benchmark reports
  cannot report overall `CLOSED` from a passing subset.
- **Composition** of marginal artifact scores (`privacy-composition`) is a
  *scenario projection*, not measured joint leakage or a validated confidence
  bound; duplicate inputs do not supply independent evidence.
- Hardware byte/timing observations remain descriptive until independent
  trials support suitable inference.

There is deliberately **no universal risk score and no maturity model** — the
tool refuses to emit any aggregate scalar. Keep exposure, impact, and evidence
quality as separate fields.

## Extending the framework

Third-party attack modules register via entry points and are discovered at
runtime:

```toml
[project.entry-points."llm_attacker.attacks"]
my-attack = "my_pkg.my_module"
```

Rules for attack modules (CI-enforced):

- Lazily imported; `--help` must work **torch-less** — torch may only be
  touched inside `run()`.
- Declare modes (`training`/`inference`) and label requirements honestly.
- Emit the artifact schema consumed by `--scorecard`/`--coverage`/`--report`.

`attacker/security_assumptions.py` records transport/deployment security
conditions (e.g. TLS 1.3) as **operator assertions** with
`deployment_verified=false`; they never count as a passed attack or a privacy
certificate.

## Testing

```bash
python -m pytest tests/ -q          # full suite
python -m pytest tests/ -q -m slow  # end-to-end scenarios (need torch/SDKs)
python tests/run_suite.py --help    # merge per-environment reports
```

`tests/data/` contains regression fixtures — test inputs, not empirical
research results. `tests/last_run.json` records individual outcomes per
environment.

## Repository layout

```
attacker/               the framework package
  __main__.py           CLI dispatcher (python -m attacker)
  attacks/              65 lazily-registered attack modules
  experiment.py         end-to-end protocol engine
  splitbench.py         split-model capture rig (GPT-2 / Mistral / Qwen3)
  scorecard.py          gate evaluation and scorecards
  coverage.py           execution coverage
  report.py             markdown report rendering
  benchmark.py          environment battery (benchmark.v1)
  recon.py              model/node/endpoint fingerprinting
  evidence.py           hash-chained evidence manifests (v2)
  defense_bundles.py    external defense-artifact adapter
  agent_harness.py      live agent harness (model-driven planners)
  remote.py, nodes.json fleet capture collection
tests/                  pytest suite + regression fixtures
pyproject.toml          packaging, extras, entry points
requirements-research.lock
                        exact research-stack validation environment
```

Research artifacts produced *by* this tool (experiment outputs, evidence
bundles, corpora) are written to ignored directories (`results/`, `captures/`,
`evidence/`) and are kept outside the published source package.

## Licence and citation

Framework code and release packaging are distributed under the [MIT licence](LICENSE). Third-party model, corpus and dependency terms remain applicable. The manuscript's arXiv distribution choice is stated above.

Please cite the paper title and authors Georgios Politis and Evangelos Pappas when using the framework or evidence. An arXiv identifier will be added after assignment. The paper reports conditional measurements under declared observations, knowledge, corpora and budgets; a passing score is not a general privacy guarantee.
