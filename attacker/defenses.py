#!/usr/bin/env python3
"""Reference defenses — worked targets so any user gets a PASS and a FAIL
out of the box (RESEARCH_FIRST_CLASS_ATTACKER.md Phase D item 12).

make_defense(name) returns a generator that builds a synthetic-but-real
trusted bundle in exactly the schema latent-probe / latent-sensitivity
consume (attacker/attacks/latent_probe.py:61-65):

    train_wire    [train_blocks, tokens, D]  released train views (fp32)
    train_tokens  [train_blocks, tokens]     trusted labels (int64)
    eval_wire     [eval_blocks, tokens, D]
    eval_tokens   [eval_blocks, tokens]
    train_order   [train_blocks, tokens]     true pre-permutation index of
    eval_order    [eval_blocks, tokens]      each released row (identity when
                                             the release map does not permute);
                                             permutation-order-recovery scoring
                                             ground truth, cf. the labels

Canonical rows are clustered per token (h = signal_scale * dir[token] +
noise, cf. attacker/synthetic.py's toy world); each released block is one
frame of `tokens` rows. The defenses differ only in the release map:

  plain-split  identity — raw activations on the wire (no protection).
               Expected: FAIL the frozen gate badly (coordinate probes
               decode directly).
  rotation     one fresh orthogonal key per released block from the
               ratchet chain (the historical v2 behavior). Coordinate
               identity is scrambled, but rotations preserve row norms
               and the Gram matrix EXACTLY, so the invariant arms still
               decode. Expected: FAIL.
  dp-noise     canonical + iid Gaussian noise, configurable sigma (the
               Gaussian mechanism). Expected: FAIL at the default
               utility-preserving sigma; PASS needs sigma on the order of
               the cluster separation, which destroys utility — the
               honest DP story, measured by the scenario runner.
  composite    per-block fresh orthogonal gauge + token permutation +
               moderate noise (approximating v13). Expected: PASS at the
               shipped defaults. Candid note: norms and the SORTED Gram
               are exactly gauge- and permutation-invariant, so the
               load-bearing ingredient against the invariant arms is the
               NOISE; rotation+permutation strip the coordinate and
               row-position channels. Whether a given composite config
               PASSes is measured by run_scenarios, never asserted.

The VFLAIR-LLM (arXiv:2508.03097) analogue seam-defense pack (sparsify,
mid-lite, at-lite, santext, custext, snd-lite) is registered in
SEAM_DEFENSE_PACK: these defenses transform the LIVE seam / input stream
of a split run (implemented in attacker/splitbench.py, `--defense X`),
not a synthetic bundle, so they are measured by splitbench + the seam
attacks rather than the bundle BDD runner below — which keeps iterating
exactly the four reference defenses above (describe_scenarios() with the
default include_seam_pack=False is pinned). Each pack entry documents
the VFLAIR mechanism it analogues and the honest gap to the full method;
validate_defense_config recognizes the pack names and dispatches to
validate_seam_defense_config.

Torch: this MODULE imports torch-free and every pure-python helper
(describe_scenarios, validate_defense_config, make_defense metadata)
works without it. Bundle GENERATION requires torch — acceptable, because
the probes the bundles feed require torch anyway — and says so clearly
when torch is missing. run_scenarios additionally shells out to
`python -m attacker --mode training --attack latent-probe`, which needs
the defense SDK (privacy_runtime) on the host; subprocess failures are
recorded as honest error rows, never hidden.

BDD usage (on a torch host):
    python -c "from attacker.defenses import run_scenarios; \\
               run_scenarios('out/defense_matrix')"
"""

import json
import math
import os
import subprocess
import sys

try:
    import torch
except ImportError:
    torch = None

from . import artifacts, gate

SCHEMA = "llm_attacker.reference_bundle.v1"
SCORECARD_SCHEMA = "llm_attacker.defense_scorecard.v1"

TORCH_ERROR = ("attacker.defenses: bundle generation requires torch (the "
               "latent probes the bundles feed require torch anyway). "
               "Install torch or use a venv (e.g. split-training/venv-cpu); "
               "describe_scenarios()/validate_defense_config() work "
               "torch-less.")

SHAPE_DEFAULTS = {
    "hidden": 32,
    "tokens": 8,
    "train_blocks": 48,
    "eval_blocks": 24,
    "vocab": 8,
    "signal_scale": 4.0,
}

DEFENSES = {
    "plain-split": {
        "description": "raw canonical activations released (no protection)",
        "expected_verdict": "FAIL",
        "why": "coordinate probes decode clustered raw activations far "
               "above the majority control",
        "config": {},
    },
    "rotation": {
        "description": "one fresh orthogonal key per released block "
                       "(ratchet chain; the historical v2 behavior)",
        "expected_verdict": "FAIL",
        "why": "rotation preserves row norms and the Gram matrix exactly, "
               "so the invariant/graph probe arms still decode",
        "config": {},
    },
    "dp-noise": {
        "description": "Gaussian mechanism: canonical + iid N(0, sigma^2)",
        "expected_verdict": "FAIL",
        "why": "at the default utility-preserving sigma the probe still "
               "decodes; PASS needs sigma ~ cluster separation, which "
               "destroys utility",
        "config": {"sigma": 0.25},
    },
    "composite": {
        "description": "per-block fresh orthogonal gauge + token "
                       "permutation + noise (approximating v13)",
        "expected_verdict": "PASS",
        "why": "each released block is its own gauged, permuted, noisy "
               "view: coordinate identity and row position are stripped "
               "and the norm/Gram invariants are noise-blurred",
        "config": {"sigma": 1.0, "permute": True},
    },
}


SEAM_DEFENSE_PACK = {
    "sparsify": {
        "vflair": "GradientSparsification",
        "description": "keep only the top-`sparsity` fraction of "
                       "magnitude entries per seam row (rest zeroed)",
        "honesty": "analogue applied to the fwd seam ACTIVATION — "
                   "VFLAIR sparsifies gradients; same magnitude-pruning "
                   "mechanism, different tensor",
        "config": {"sparsity": 0.1},
    },
    "mid-lite": {
        "vflair": "MID (mutual-information defense)",
        "description": "fixed seeded random Gaussian projection of the "
                       "seam d -> bottleneck_dim, embedded back in "
                       "d-space (zeros elsewhere)",
        "honesty": "NOT learned MID: no mutual-information objective is "
                   "optimized; the bottleneck is a fixed random linear "
                   "map, so the wire content is rank-limited by "
                   "construction but not information-minimized",
        "config": {"bottleneck_dim": 8},
    },
    "at-lite": {
        "vflair": "adversarial training (AT)",
        "description": "shift the seam by -at_eps * v1, v1 the online "
                       "EMA estimate of the seam covariance's top PCA "
                       "direction",
        "honesty": "NOT adversarial training: no min-max optimization; "
                   "a per-step constant shift along the dominant "
                   "invertible direction (an attacker who knows the "
                   "public estimate can subtract it back — the estimate "
                   "tracks a moving fine-tuned seam, which is the "
                   "mechanism under test)",
        "config": {"at_eps": 0.1},
    },
    "santext": {
        "vflair": "SanText",
        "description": "replace each input token with probability "
                       "santext_p by a uniform draw from its santext_k "
                       "nearest embedding neighbors (cosine over wte), "
                       "before embedding",
        "honesty": "embedding-cosine neighborhoods (no differential-"
                   "privacy accounting of the full SanText mechanism); "
                   "canonical.pt keeps the ORIGINAL ids — the attacker's "
                   "ground truth is the victim's original text while the "
                   "wire carries sanitized content",
        "config": {"santext_p": 0.1, "santext_k": 8},
    },
    "custext": {
        "vflair": "CusText",
        "description": "deterministic per-token custom mapping into the "
                       "token's santext_k cosine neighbors, derived from "
                       "a (seed, token)-keyed RNG",
        "honesty": "toy-scale analogue: the mapping is seeded-RNG choice "
                   "within embedding neighbors, not CusText's trained "
                   "customization; it is a fixed substitution cipher "
                   "over the vocab (frequency/structure leaks are part "
                   "of what the attacks measure)",
        "config": {"santext_k": 8},
    },
    "snd-lite": {
        "vflair": "split-and-denoise (SND)",
        "description": "split the seam into two additive shares and "
                       "denoise one through a fixed seeded random "
                       "smoothing matrix: h_wire = s1 @ S + s2",
        "honesty": "REDUCED analogue: our threat model has ONE observed "
                   "channel, so both shares cross the same seam and the "
                   "original scheme's separate-transport assumption is "
                   "NOT modeled — the mechanism reduces to structured "
                   "randomized smoothing of the seam",
        "config": {"snd_sigma": 1.0},
    },
}


def describe_scenarios(include_seam_pack=False):
    """The living BDD matrix as plain data (pure python): one scenario per
    reference defense against latent-probe, with the expected gate verdict
    and why. Pure function — covered by the torch-free test layer.

    With include_seam_pack=True the VFLAIR-analogue seam-defense pack is
    APPENDED (attack "seam-inversion", surface "splitbench-seam",
    expected_verdict None): the pack is measured via splitbench runs,
    never asserted by this bundle BDD matrix — run_scenarios iterates the
    default four."""
    scenarios = []
    for name, meta in DEFENSES.items():
        scenarios.append({
            "defense": name,
            "attack": "latent-probe",
            "expected_verdict": meta["expected_verdict"],
            "why": meta["why"],
            "description": meta["description"],
            "config": {**SHAPE_DEFAULTS, **meta["config"]},
        })
    if include_seam_pack:
        for name, meta in SEAM_DEFENSE_PACK.items():
            scenarios.append({
                "defense": name,
                "attack": "seam-inversion",
                "surface": "splitbench-seam",
                "expected_verdict": None,
                "why": meta["honesty"],
                "description": f"{meta['description']} — VFLAIR "
                               f"{meta['vflair']} analogue; measured via "
                               f"splitbench --defense {name}",
                "config": dict(meta["config"]),
            })
    return scenarios


def validate_seam_defense_config(name, config):
    """Pure-python validation of a seam-defense-pack config (the same
    ranges splitbench's BenchConfig enforces); returns a list of
    problems ([] = valid), never raises. Unknown keys are reported, not
    coerced."""
    if name not in SEAM_DEFENSE_PACK:
        return [f"unknown seam defense {name!r} (valid: "
                f"{sorted(SEAM_DEFENSE_PACK)})"]
    allowed = SEAM_DEFENSE_PACK[name]["config"]
    problems = []
    for key in config:
        if key not in allowed:
            problems.append(f"unknown config key {key!r} for {name!r} "
                            f"(valid: {sorted(allowed)})")

    def is_num(key):
        return isinstance(config.get(key), (int, float)) \
            and not isinstance(config.get(key), bool)

    def is_int(key):
        return isinstance(config.get(key), int) \
            and not isinstance(config.get(key), bool)

    def bad(key, cond, msg):
        if key in config and cond:
            problems.append(f"{key} {msg} (got {config[key]!r})")
    bad("sparsity", not is_num("sparsity")
        or not 0.0 < config["sparsity"] <= 1.0,
        "must be a number in (0, 1] (fraction KEPT)")
    bad("bottleneck_dim", not is_int("bottleneck_dim")
        or config["bottleneck_dim"] < 1, "must be an int >= 1")
    bad("at_eps", not is_num("at_eps") or config["at_eps"] < 0,
        "must be a number >= 0")
    bad("santext_p", not is_num("santext_p")
        or not 0.0 <= config["santext_p"] <= 1.0,
        "must be a probability in [0, 1]")
    bad("santext_k", not is_int("santext_k") or config["santext_k"] < 1,
        "must be an int >= 1")
    bad("snd_sigma", not is_num("snd_sigma") or config["snd_sigma"] < 0,
        "must be a number >= 0")
    return problems


def validate_defense_config(name, config):
    """Pure-python config validation; returns a list of problems ([] =
    valid), never raises. Unknown defense names and unknown/mistyped keys
    are reported, not coerced. Seam-defense-pack names (SEAM_DEFENSE_PACK)
    dispatch to validate_seam_defense_config."""
    if name in SEAM_DEFENSE_PACK:
        return validate_seam_defense_config(name, config)
    if name not in DEFENSES:
        return [f"unknown defense {name!r} (valid: {sorted(DEFENSES)}; "
                f"seam pack: {sorted(SEAM_DEFENSE_PACK)})"]
    allowed = {**SHAPE_DEFAULTS, **DEFENSES[name]["config"]}
    problems = []
    for key, value in config.items():
        if key not in allowed:
            problems.append(f"unknown config key {key!r} for {name!r} "
                            f"(valid: {sorted(allowed)})")
    def bad(key, cond, msg):
        if key in config and cond:
            problems.append(f"{key} {msg} (got {config[key]!r})")
    for key in ("hidden", "tokens", "train_blocks", "eval_blocks", "vocab"):
        bad(key, not isinstance(config.get(key), int)
            or isinstance(config.get(key), bool), "must be an int")
    bad("hidden", isinstance(config.get("hidden"), int)
        and config["hidden"] < 4, "must be >= 4")
    bad("tokens", isinstance(config.get("tokens"), int)
        and config["tokens"] < 2, "must be >= 2")
    bad("train_blocks", isinstance(config.get("train_blocks"), int)
        and config["train_blocks"] < 2, "must be >= 2")
    bad("eval_blocks", isinstance(config.get("eval_blocks"), int)
        and config["eval_blocks"] < 2, "must be >= 2")
    bad("vocab", isinstance(config.get("vocab"), int)
        and config["vocab"] < 2, "must be >= 2")
    bad("signal_scale", not isinstance(config.get("signal_scale"),
                                       (int, float))
        or isinstance(config.get("signal_scale"), bool)
        or config["signal_scale"] <= 0, "must be a positive number")
    if "sigma" in allowed:
        bad("sigma", not isinstance(config.get("sigma"), (int, float))
            or isinstance(config.get("sigma"), bool)
            or config["sigma"] < 0, "must be a number >= 0")
    if "permute" in allowed:
        bad("permute", not isinstance(config.get("permute"), bool),
            "must be a bool")
    return problems


class ReferenceDefense:
    """One reference defense: a bundle generator with validated config.

    generate(seed=...) -> the trusted bundle dict (torch tensors). Torch
    is touched only inside generate(); the object itself is torch-free."""

    def __init__(self, name, **overrides):
        if name not in DEFENSES:
            raise ValueError(f"unknown defense {name!r} (valid: "
                             f"{sorted(DEFENSES)})")
        problems = validate_defense_config(name, overrides)
        if problems:
            raise ValueError(f"invalid config for {name!r}: "
                             + "; ".join(problems))
        self.name = name
        self.meta = DEFENSES[name]
        self.config = {**SHAPE_DEFAULTS, **self.meta["config"],
                       **overrides}

    @property
    def expected_verdict(self):
        return self.meta["expected_verdict"]

    def _canonical(self, n_blocks, tok_dirs, rng_seed):
        """Clustered canonical rows: h = scale * dir[token] + noise."""
        g = torch.Generator().manual_seed(rng_seed)
        seq, vocab = self.config["tokens"], self.config["vocab"]
        dim = self.config["hidden"]
        tokens = torch.randint(0, vocab, (n_blocks, seq), generator=g)
        rows = (self.config["signal_scale"] * tok_dirs[tokens]
                + torch.randn(n_blocks, seq, dim, generator=g))
        return rows.float(), tokens

    def _release(self, canon, master_seed):
        """The defense's release map, per block (fresh key per block from
        the ratchet chain — the repo's rotation primitive).

        Returns (released, order): order[b, s] is the TRUE pre-permutation
        index of the row released at slot s of block b — identity when the
        defense does not permute — recorded as scoring ground truth for the
        permutation-order-recovery metric (attacker/attacks/latent_matching.py),
        the same way the trusted labels are recorded for token recovery."""
        from .solve_primitives import ratchet_secret
        n_blocks, seq, dim = canon.shape
        out = canon.clone()
        order = torch.arange(seq).expand(n_blocks, seq).clone()
        for b in range(n_blocks):
            if self.name == "plain-split":
                continue
            if self.name in ("rotation", "composite"):
                key = ratchet_secret(dim, master_seed, b)
                out[b] = (canon[b] @ key).float()
                if self.name == "composite" and self.config["permute"]:
                    g = torch.Generator().manual_seed(
                        master_seed + 7919 * b)
                    perm = torch.randperm(seq, generator=g)
                    out[b] = out[b][perm]
                    order[b] = perm
            if self.name in ("dp-noise", "composite"):
                sigma = float(self.config["sigma"])
                if sigma > 0.0:
                    g = torch.Generator().manual_seed(
                        master_seed + 104729 * (b + 1))
                    out[b] = out[b] + sigma * torch.randn(
                        seq, dim, generator=g)
        return out, order

    def generate(self, seed=1234):
        """Build the trusted bundle (requires torch — see TORCH_ERROR)."""
        if torch is None:
            raise SystemExit(TORCH_ERROR)
        g = torch.Generator().manual_seed(seed)
        tok_dirs = torch.randn(self.config["vocab"], self.config["hidden"],
                               generator=g)
        tok_dirs = tok_dirs / tok_dirs.norm(dim=1, keepdim=True)
        train_x, train_y = self._canonical(self.config["train_blocks"],
                                           tok_dirs, seed + 101)
        eval_x, eval_y = self._canonical(self.config["eval_blocks"],
                                         tok_dirs, seed + 202)
        train_wire, train_order = self._release(train_x, seed + 1000)
        eval_wire, eval_order = self._release(eval_x, seed + 2000)
        return {
            "schema": SCHEMA,
            "defense": self.name,
            "config": dict(self.config),
            "expected_verdict": self.expected_verdict,
            "train_wire": train_wire,
            "train_tokens": train_y,
            "eval_wire": eval_wire,
            "eval_tokens": eval_y,
            "train_order": train_order,
            "eval_order": eval_order,
        }


def make_defense(name, **overrides):
    """Registry entry point: make_defense("rotation", hidden=64) -> a
    ReferenceDefense bundle generator. Unknown names and invalid configs
    are descriptive errors."""
    return ReferenceDefense(name, **overrides)


def run_scenarios(output_dir, shape=None, epochs=2, restarts=1,
                  batch_size=256, lr=1e-3, seed=1234, timeout=900):
    """The living BDD suite: {4 reference defenses} x {latent-probe}, end
    to end. Each scenario generates its bundle, shells out to
    `sys.executable -m attacker --mode training --attack latent-probe
    --bundle ...`, and collects a scorecard row scored against the frozen
    gate. Requires torch (bundle generation); the probe subprocess
    additionally needs privacy_runtime on the host — subprocess failures
    are recorded as error rows with the stderr tail, never hidden.

    Returns the list of rows; also writes <output_dir>/scorecard.json."""
    if torch is None:
        raise SystemExit(TORCH_ERROR)
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for scenario in describe_scenarios():
        name = scenario["defense"]
        config = dict(scenario["config"])
        if shape:
            config.update(shape)
        row = {"defense": name, "attack": scenario["attack"],
               "expected_verdict": scenario["expected_verdict"],
               "config": config, "status": "ok"}
        problems = validate_defense_config(name, config)
        if problems:
            row.update(status="error", error="; ".join(problems),
                       expectation_met=None)
            rows.append(row)
            continue
        bundle_path = os.path.join(output_dir, f"{name}.pt")
        out_json = os.path.join(output_dir, f"{name}.latent_probe.json")
        try:
            bundle = make_defense(name, **config).generate(seed)
            torch.save(bundle, bundle_path)
        except Exception as e:
            row.update(status="error", error=f"bundle generation: {e}",
                       expectation_met=None)
            rows.append(row)
            continue
        cmd = [sys.executable, "-m", "attacker", "--mode", "training",
               "--attack", "latent-probe", "--bundle", bundle_path,
               "--output", out_json, "--epochs", str(epochs),
               "--restarts", str(restarts), "--batch-size",
               str(batch_size), "--lr", str(lr)]
        print(f"[bdd] {name}: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
        row["rc"] = proc.returncode
        if proc.returncode != 0 or not os.path.isfile(out_json):
            row.update(status="error",
                       error="latent-probe subprocess failed",
                       stderr_tail=proc.stderr.strip().splitlines()[-3:],
                       expectation_met=None)
            rows.append(row)
            print(f"[bdd] {name}: subprocess rc={proc.returncode}; "
                  "recorded as an error row")
            continue
        with open(out_json) as f:
            artifact = json.load(f)
        summary = artifact["summary"][0]
        n_eval = (artifact["config"]["eval_blocks"]
                  * artifact["config"]["sequence_length"])
        verdict = gate.verdict(summary["best_probe_top1_pct"] / 100.0,
                               summary["label_free_majority_pct"] / 100.0,
                               n_eval)
        row.update({
            "verdict": verdict["verdict"],
            "expectation_met": verdict["verdict"]
            == scenario["expected_verdict"],
            "best_probe_top1_pct": summary["best_probe_top1_pct"],
            "label_free_majority_pct": summary["label_free_majority_pct"],
            "excess_pp": verdict["excess_pp"],
            "at_floor": verdict["at_floor"],
            "artifact": out_json,
        })
        print(f"[bdd] {name}: verdict={row['verdict']} "
              f"(expected {row['expected_verdict']}) "
              f"excess={row['excess_pp']:.3f} pp")
        rows.append(row)
    scorecard = {"schema": SCORECARD_SCHEMA,
                 "config": {"epochs": epochs, "restarts": restarts,
                            "batch_size": batch_size, "lr": lr,
                            "seed": seed, "gate": gate.GATE_VERSION},
                 "results": rows,
                 "summary": [{
                     "n_scenarios": len(rows),
                     "n_ok": sum(r["status"] == "ok" for r in rows),
                     "n_expectation_met": sum(
                         r.get("expectation_met") is True for r in rows),
                     "verdicts": {r["defense"]: r.get("verdict")
                                  for r in rows}}]}
    artifacts.write_artifact(os.path.join(output_dir, "scorecard.json"),
                             scorecard)
    return rows


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("reference defenses (pure-python layer):")
    scenarios = describe_scenarios()
    check("four reference defenses, all against latent-probe",
          [s["defense"] for s in scenarios]
          == ["plain-split", "rotation", "dp-noise", "composite"]
          and all(s["attack"] == "latent-probe" for s in scenarios))
    check("expectations give a PASS and a FAIL out of the box",
          [s["expected_verdict"] for s in scenarios]
          == ["FAIL", "FAIL", "FAIL", "PASS"])
    check("scenario configs validate clean",
          all(validate_defense_config(s["defense"], s["config"]) == []
              for s in scenarios))
    check("validation rejects an unknown defense / key / bad sigma",
          "unknown defense" in validate_defense_config("nope", {})[0]
          and validate_defense_config("dp-noise", {"sgima": 1.0})
          and validate_defense_config("dp-noise", {"sigma": -1.0}))
    try:
        make_defense("nope")
        check("make_defense unknown name is a descriptive error", False)
    except ValueError as e:
        check("make_defense unknown name is a descriptive error",
              "nope" in str(e) and "valid" in str(e))
    if torch is None:
        try:
            make_defense("rotation").generate()
            check("torch-free host: generate() says so clearly", False)
        except SystemExit as e:
            check("torch-free host: generate() says so clearly",
                  "requires torch" in str(e))
    else:
        bundle = make_defense("composite").generate(0)
        check("bundle schema keys match latent_probe's reads",
              all(k in bundle for k in ("train_wire", "train_tokens",
                                        "eval_wire", "eval_tokens"))
              and bundle["train_wire"].ndim == 3)
        width = bundle["eval_order"].shape[1]
        check("order ground truth: composite permutes, plain is identity",
              bool((bundle["eval_order"].sort(-1).values
                    == torch.arange(width)).all())
              and bool((make_defense("plain-split").generate(0)["eval_order"]
                        == torch.arange(width)).all()))

    print("VFLAIR-analogue seam-defense pack (registry layer):")
    pack = describe_scenarios(include_seam_pack=True)
    check("seam pack appends 6 entries after the unchanged 4",
          [s["defense"] for s in pack]
          == ["plain-split", "rotation", "dp-noise", "composite",
              "sparsify", "mid-lite", "at-lite", "santext", "custext",
              "snd-lite"] and pack[:4] == scenarios)
    check("pack scenarios target seam-inversion on the splitbench "
          "surface, verdicts measured not asserted",
          all(s["attack"] == "seam-inversion"
              and s["surface"] == "splitbench-seam"
              and s["expected_verdict"] is None
              and s["why"] and s["description"] for s in pack[4:]))
    check("pack configs validate clean via the shared entry point",
          all(validate_defense_config(s["defense"], s["config"]) == []
              for s in pack[4:]))
    check("pack validation rejects typos and out-of-range knobs",
          validate_defense_config("sparsify", {"sprsity": 0.2})
          and validate_defense_config("sparsify", {"sparsity": 0.0})
          and validate_defense_config("sparsify", {"sparsity": 1.5})
          and validate_defense_config("mid-lite", {"bottleneck_dim": 0})
          and validate_defense_config("at-lite", {"at_eps": -1.0})
          and validate_defense_config("santext", {"santext_p": 1.5})
          and validate_defense_config("custext", {"santext_k": 0})
          and validate_defense_config("snd-lite", {"snd_sigma": -0.5}))
    check("unknown defense error names both registries",
          "unknown defense" in validate_defense_config("nope", {})[0]
          and "seam pack" in validate_defense_config("nope", {})[0])
    check("validate_seam_defense_config rejects unknown names too",
          "unknown seam defense"
          in validate_seam_defense_config("mid", {})[0])

    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
