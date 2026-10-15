# Confidential-compute pre-cloud audit

This is the minimum shared TEE/FHE/hybrid security execution for the three
papers. It prepares and grades a finite audit; it does not authorise, provision,
or contact a cloud resource by itself.

## What is now executable

`secure-compute-audit` produces one hash-bound case plan and one result record
per case. The common TEE plan covers:

- one valid attestation/key-release path and seven invalid paths: wrong
  measurement, modified quote, stale collateral, replayed nonce, wrong-machine
  relay, endpoint-key substitution, and channel-binding mismatch;
- one clean protocol control, plus replay, reorder, duplicate, truncate,
  cross-session swap, downgrade, tamper, stale-step/result, and request/result
  substitution on the live connection;
- one correct-computation control, plus stale/substituted weights, skipped
  operators, scaled/zeroed gradients, replayed outputs, and incorrect results;
- paired `external_attacker` and `trusted_diagnostic` canaries for forward,
  backward, update, CPU/GPU staging, IPC, VRAM, logs, checkpoints, dumps,
  environment, and telemetry; and
- packet shape, timing, performance counters, access traces, and crash/restart
  host observations. Unavailable traces are `UNASSESSED`, never passes.

A host-trace case marked `PASS` means the declared observation was collected;
it does **not** mean that the trace contains no exploitable leakage. Any leakage
claim needs its own frozen classifier, null, positive control and threshold.

The FHE and hybrid profiles add malformed, modified, replayed and wrong-context
ciphertexts, extra-slot and cross-tenant-context cases, production-feedback
blinding, and an intentionally leaky positive-control oracle.

The versioned gate requires the valid release to work, releases zero secrets on
every invalid path, accepts zero protocol or computation mutations, exposes no
boundary canary in the external view, observes every trusted positive-control
canary, and detects the leaky FHE control without detecting a production oracle.
There is no percentage tolerance for secret release.

## Freeze a plan without running a target

The canonical pre-order TEE plan generated from the command below is committed
at `docs/secure-compute-tee-plan.json`. The hardware packet binds its exact
file SHA-256; regenerating a different plan requires review and a new binding.

```bash
python -m attacker --mode training --attack secure-compute-audit \
  --profile tee \
  --plan-only --plan-output secure-compute-plan.json
```

Start from
[`secure-compute-deployment-manifest.example.json`](secure-compute-deployment-manifest.example.json).
The CLI validates its required fields, zero-tolerance gates, disabled debug
mode, workload/corpus digests, budget, and profile before hashing it. The
adapter section additionally freezes the provider, factory entry point,
implementation revision, reviewed command argv, and credential mode. The
template deliberately has `ready_for_execution: false`; change it to `true`
only after review. Before an instance exists, omit `--deployment-manifest`: the
result is the immutable pre-order plan template and carries `null` only for the
future deployment binding. Review and commit its file SHA-256 with the hardware
packet. Before arrival, freeze the static deployment manifest and have Paper 4
materialise that one field against its exact bytes. Arrival and admission then
bind that already-frozen deployment digest; they never rewrite it.
The case inventory, order, expectations, required flags and controls cannot
change.

The static execution deployment manifest binds the contract, security-evidence
adapter, expected hardware, firmware, verifier policy, workload identity, seed
list, budget and stop rules. It deliberately does not bind a future instance
ID, arrival, admission, runtime record, active-security result, lifecycle launch
or campaign receipt. Those later bytes are joined by Paper 4's signed acyclic
lifecycle chain. The example values are deliberately non-executable
placeholders. Changing any case or manifest field changes its bound digest.

For a local machinery check only:

```bash
python -m attacker --mode training --attack secure-compute-audit \
  --profile hybrid --toy --output /tmp/secure-compute-toy.json
```

The fixture deliberately marks host traces unavailable. Its overall result is
therefore `UNASSESSED` and the command exits 2; it is not cloud or scientific
evidence.

### Local reference process adapter

The repository includes a real implementation of the JSONL process boundary,
backed by a deterministic no-hardware target. Its `defended` fixture must pass
every required case; its deliberately `vulnerable` fixture must fail every
negative case while preserving the valid-release, clean-protocol,
correct-computation, trusted-canary, host-observation and leaky-oracle controls.
It never contacts a network service.

```bash
python -m attacker --mode training --attack secure-compute-audit \
  --profile tee --require-host-traces \
  --driver-command 'python -m attacker.secure_compute_reference_adapter \
    --fixture defended --evidence-dir /tmp/cc-reference-evidence' \
  --evidence-bundle-dir /tmp/cc-reference-evidence \
  --deployment-manifest frozen-deployment.json \
  --output /tmp/cc-reference-audit.json
```

Use `--fixture vulnerable` only to qualify audit sensitivity. Each fixture
writes one evidence manifest per case and separate category-required evidence
JSON files. Repeated executions are byte-deterministic for the same case plan.
These synthetic records qualify machinery; they are not hardware evidence.

## Provider process contract

`attacker.secure_compute_process_adapter` is the provider-neutral executable
host. A deployment module exports a factory
`create_target(runtime) -> SecureComputeTarget`; the returned target implements
`execute(case)`. The host, rather than the provider module, owns strict JSONL
framing, frozen-manifest/profile checks, request binding, result validation, and
pre-emission evidence-bundle verification. It imports no cloud SDK and handles
no credential itself.

The inherited command/environment interface is exactly:

| Control | Flag | Environment variable | Meaning |
|---|---|---|---|
| Target factory | `--target-factory` | `LLM_ATTACKER_CC_TARGET_FACTORY` | Frozen `module:callable` provider entry point |
| Provider | `--provider` | `LLM_ATTACKER_CC_PROVIDER` | Manifest-bound provider ID, for example `azure` or `gcp` |
| Deployment | `--deployment-manifest` | `LLM_ATTACKER_CC_DEPLOYMENT_MANIFEST` | Reviewed deployment JSON path |
| Evidence root | `--evidence-dir` | `LLM_ATTACKER_CC_EVIDENCE_DIR` | Access-controlled local or mounted evidence directory |

Flags override inherited variables. All four controls are mandatory and
non-secret. Provider and factory must exactly match the deployment manifest's
`adapter` section; the other two paths select the manifest and evidence bundle
whose contents are digest-bound. Provider credentials must use the provider
wrapper's native managed/workload identity or credential discovery and must
never be placed in these controls, the command line, stdout, results, or the
deployment manifest. Provider logs go to stderr; stdout is JSONL only.

For example, after implementing and reviewing an Azure wrapper factory:

```bash
export LLM_ATTACKER_CC_TARGET_FACTORY='organisation.azure_cc:create_target'
export LLM_ATTACKER_CC_PROVIDER='azure'
export LLM_ATTACKER_CC_DEPLOYMENT_MANIFEST="$PWD/frozen-deployment.json"
export LLM_ATTACKER_CC_EVIDENCE_DIR='/mounted/cc-evidence'

python -m attacker --mode training --attack secure-compute-audit \
  --profile tee \
  --driver-command 'python -m attacker.secure_compute_process_adapter' \
  --deployment-manifest "$LLM_ATTACKER_CC_DEPLOYMENT_MANIFEST" \
  --evidence-bundle-dir "$LLM_ATTACKER_CC_EVIDENCE_DIR" \
  --plan-output admitted-plan.json --output cc-admission.json --evidence
```

The same command uses a GCP wrapper by changing the provider and factory in the
reviewed manifest and inherited environment. `--driver-command` is parsed and
executed without a shell, so do not put shell assignments or secret-bearing
arguments inside it. A service launcher, scheduler, or already-exported parent
environment supplies identity and controls.

A provider wrapper has this import-level shape:

```python
from attacker.secure_compute_process_adapter import SecureComputeTarget

class ProviderTarget(SecureComputeTarget):
    def __init__(self, runtime):
        self.runtime = runtime
        # Initialise the selected provider clients using managed identity.

    def execute(self, case):
        # Exercise the real operation, write bound evidence under
        # runtime.evidence_dir, and return llm_attacker.secure_compute_result.v1.
        raise NotImplementedError

def create_target(runtime):
    return ProviderTarget(runtime)
```

The template is deliberately incomplete: returning synthetic answers from this
factory would only qualify software plumbing, not the selected platform.

An alternative standalone executable may implement the same one-input/
one-output JSONL contract directly. Invoke it explicitly:

```bash
python -m attacker --mode training --attack secure-compute-audit \
  --profile tee --driver-command './vendor_adapter --frozen-config cc.json' \
  --deployment-manifest frozen-deployment.json \
  --evidence-bundle-dir mounted-evidence-bundle \
  --plan-output admitted-plan.json --output cc-admission.json --evidence
```

The framework executes the command without a shell. The plan and every case
bind the SHA-256 of `--deployment-manifest`. Each input uses schema
`llm_attacker.secure_compute_case.v1`; each output must use
`llm_attacker.secure_compute_result.v1`. Each result echoes the complete-case,
plan and deployment-manifest bindings and binds a per-case evidence manifest.
The returned `request_id`, `session_id`, and `step_id` must match exactly.
Missing, duplicate, extra, malformed, errored, replayed, or mis-bound records
fail closed. Evidence fields must be exactly those required by that category;
real execution and re-grading resolve and hash every manifest and evidence file.

The provider wrapper still owns these platform-specific operations:

1. Create fresh nonces, canary values, sessions, and any sacrificial secrets.
2. Apply the requested mutation against the real attester, verifier,
   key-release service, protected channel, or compute backend.
3. Record only booleans, stable error codes, identifiers, and non-secret
   measurements in `details`. Put category-required artifact SHA-256 values in
   `evidence`; the gate rejects missing or malformed digests. Never emit keys,
   canary values, plaintext,
   credentials, quote secrets, or raw trusted diagnostic tensors.
4. Keep raw quotes, collateral, firmware measurements, packet captures, and
   trusted diagnostic material in the access-controlled evidence bundle and
   bind them by digest from the deployment manifest.
5. Emit `status="unavailable"` with a stable `error_code` when a trace cannot
   be observed. Use `--require-host-traces` only when the frozen threat model
   says those traces are mandatory; they then fail rather than remain
   `UNASSESSED`.

For Azure, the unfinished wrapper work is the selected confidential VM/GPU
provisioning and CC mode, Azure identity and endpoint configuration, acquisition
and verification of the platform/accelerator attestation evidence, policy-bound
release of a sacrificial key, real mutation injection, protected workload
execution, and permitted host/accelerator trace collection. For GCP, the same
work remains against the selected Confidential VM/Space or accelerator stack,
Google identity and project/region endpoints, its attestation-token verification
and KMS/key-release policy, mutation hooks, execution, and trace facilities.
The exact service/SKU must be frozen before either wrapper can be completed.

In both providers, the wrapper must map provider errors to stable non-secret
codes; retain quotes/tokens, collateral, verifier decisions, packet captures,
and trusted diagnostics under the evidence root; implement every declared
mutation rather than infer its answer; and cooperate with the separately frozen
external lifecycle controller. Cloud provisioning, provider-side TTL/budget
enforcement, deletion and raw absence queries belong to that controller, not
this JSONL audit adapter. The generic host cannot prove that an SDK call reached genuine
hardware or that a file's contents are truthful. That assurance comes from the
selected provider evidence, its independently reviewed verifier/key-release
policy, and (where available) a measurement that binds the frozen adapter
revision. Hashes provide integrity and replay binding, not hardware provenance.

An offline result transcript can be re-graded without target access:

```bash
python -m attacker --mode training --attack secure-compute-audit \
  --profile tee --results vendor-results.jsonl \
  --deployment-manifest frozen-deployment.json \
  --evidence-bundle-dir mounted-evidence-bundle \
  --plan-output reconstructed-plan.json --output cc-admission.json --evidence
```

The profile and `--require-host-traces` setting must be identical to those used
for execution. Re-grading a transcript against a different generated plan will
fail identifier and completeness checks.

## Minimum execution policy

Run this audit during the short engineering/admission rental, before any
scientific seed. Reuse its immutable evidence bundle across the papers, while
stating that it is one shared cohort rather than three independent
replications. A `FAIL` blocks scientific execution. An `UNASSESSED` result must
be resolved or explicitly placed outside the frozen claim and threat model.
The CLI exits 0 only for `PASS`, 1 for `FAIL`, and 2 for `UNASSESSED`.

This audit does not replace utility, correctness, performance, returned-gradient
reconstruction, history accumulation, or the trained/frozen/removed/all-local
scientific controls. Those consume the admitted deployment afterward; they do
not require repeating these security mutations for every seed.
