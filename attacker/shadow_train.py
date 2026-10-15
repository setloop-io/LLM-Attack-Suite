#!/usr/bin/env python3
"""Internal shadow-model trainer for membership-inference reference statistics.

The membership attacks (attacker/attacks/lira_membership.py and the
private_phase arm of attacker/attacks/representation_privacy.py) score
per-example statistics against reference distributions that LIRA-style
attacks estimate from SHADOW MODELS: attacker-owned models trained with
and without each candidate example. This module trains those shadows
internally on a synthetic corpus with a planted, class-dependent signal,
so the attacks can generate their own reference statistics instead of
requiring operator-supplied ref_in/ref_out NLLs.

LIRA shadow-model design (build_lira_records)
---------------------------------------------
A fixed synthetic corpus of n_examples token-id sequences with binary
labels is generated deterministically (make_synthetic_corpus). Half the
examples are designated MEMBERS of the target model's training set;
non-members are never trained on (by the target or by the leave-one-out
shadows). The TARGET model is trained on all members. Shadow model j
(j = 0 .. n_shadows-1) is trained on a random half of the members
(subset drawn with seed cfg.seed + 7919*j), the standard leave-one-out
LIRA construction: the shadows NOT containing a record give its OUT
reference distribution (for non-members that is every shadow). After the
random draw, membership counts are deterministically repaired so each
member is OUT of at least 2 shadows (and IN of at least 2 when
n_shadows >= 4), which the per-record schema's >=2 reference minimum
requires. Two further "all-in" reference models (seeds cfg.seed + 31 /
+ 37) are trained on the ENTIRE corpus: they are the attacker's online
IN-hypothesis references — for each candidate record the attacker owns
models trained WITH it — so every record carries ref_in and the whole
population scores on the per-example Gaussian model. Members'
ref_in additionally includes the leave-one-out shadows trained with
them. The per-example statistic is the negative log-likelihood under
the target model, matching the offline record schema of
lira_membership.load_records: {example_id, split, statistic, ref_in,
ref_out}.

Private-phase bundle (build_phase_bundle)
-----------------------------------------
The representation_privacy.v1 private_phase arm needs attacker-owned
reference models and per-record IN/OUT NLLs for public_only,
private_training_member and never_seen records, stratified by public
exposure. In addition to the leave-one-out shadows, two "all-in"
reference models (different seeds) are trained on every private and
public example; their NLLs are each record's ref_in ("models trained
WITH the example"), while ref_out comes from the leave-one-out shadows
that never saw it. Calibration/evaluation rows carry released features
(the target model's mean token embedding of public examples) so the
probe arms run on the same bundle.

Torch is imported lazily inside the training entry points: importing
this module, ShadowConfig, make_synthetic_corpus and --help are
torch-free (repo convention, attacker/attacks/common.require_torch).
Seeding: torch training under torch.random.fork_rng + torch.manual_seed;
corpus/subset draws use pure-python random.Random. No global RNG state.

Usage:
    python -m attacker.shadow_train --output records.jsonl \
        [--phase-bundle bundle.json] [--n-shadows K] [--steps S] [--seed N]
    python -m attacker.shadow_train --output records.jsonl \
        --corpus-jsonl corpus.jsonl --model /path/to/gpt2 \
        [--device cuda] [--seq-len 128] [--lm-lr 5e-5]
"""

import argparse
import json
import math
import random
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ShadowConfig:
    n_shadows: int = 4
    steps: int = 200
    lr: float = 0.01
    hidden: int = 64
    dim: int = 32
    vocab: int = 64
    seq_len: int = 16
    n_examples: int = 64
    n_public: int = 256
    seed: int = 1234
    arch: str = "mlp"


def make_synthetic_corpus(cfg, seed):
    """Deterministic token-id corpus with planted class-dependent signal.

    Two disjoint signal token sets (one per class) are drawn from the
    vocabulary. Each example carries a per-example difficulty theta_i ~
    N(0,1) (mirroring the difficulty heterogeneity of
    lira_membership.make_toy_records): the fraction of tokens drawn from
    its class's signal set is 0.1 + 0.3 * easiness_i, so easy examples
    carry more shared signal and hard ones are nearly noise. A third of
    the positions additionally carry a per-example KEY token
    (key_i + label) mod vocab: a random shift cipher the model can only
    resolve by memorizing the example, so training on an example always
    helps — easy non-members cannot be fully fit by generalization
    alone. Remaining tokens are uniform over the vocabulary (which can
    hit signal tokens by chance, keeping the signal soft). Exactly half
    the examples are designated members (a seeded sample of the
    indices).

    Returns {"ids": [[int] * seq_len] * n, "labels": [int] * n,
             "members": sorted member indices, "difficulty": [float] * n}.
    """
    rng = random.Random(seed)
    n_signal = max(2, cfg.vocab // 16)
    pool = rng.sample(range(cfg.vocab), 2 * n_signal)
    signal = {0: pool[:n_signal], 1: pool[n_signal:]}
    ids, labels, difficulty = [], [], []
    for _ in range(cfg.n_examples):
        theta = rng.gauss(0.0, 1.0)
        label = rng.randrange(2)
        easy = 0.5 * (1.0 + math.tanh(theta))
        p_signal = 0.1 + 0.3 * easy
        key = rng.randrange(cfg.vocab)
        row = []
        for _ in range(cfg.seq_len):
            u = rng.random()
            if u < 1.0 / 3.0:
                row.append((key + label) % cfg.vocab)
            elif u < 1.0 / 3.0 + (2.0 / 3.0) * p_signal:
                row.append(rng.choice(signal[label]))
            else:
                row.append(rng.randrange(cfg.vocab))
        ids.append(row)
        labels.append(label)
        difficulty.append(theta)
    members = sorted(rng.sample(range(cfg.n_examples),
                                cfg.n_examples // 2))
    return {"ids": ids, "labels": labels, "members": members,
            "difficulty": difficulty}


def _build_model(cfg):
    if cfg.arch != "mlp":
        raise SystemExit(
            f"[shadow-train] unknown arch {cfg.arch!r}: only \"mlp\" is "
            "implemented (EmbeddingBag(vocab, dim, mode=\"mean\") -> "
            "Linear(dim, hidden) -> ReLU -> Linear(hidden, 2))")
    import torch
    return torch.nn.Sequential(
        torch.nn.EmbeddingBag(cfg.vocab, cfg.dim, mode="mean"),
        torch.nn.Linear(cfg.dim, cfg.hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg.hidden, 2))


def train_shadow(cfg, member_subset, seed):
    """Train one shadow classifier on the given corpus example indices.

    Embedding-bag -> Linear->ReLU->Linear MLP, full-batch Adam at cfg.lr
    for cfg.steps updates, CPU float32, under torch.random.fork_rng +
    torch.manual_seed (the representation_privacy.mlp_fit_predict idiom:
    no global RNG state leaks in or out). The corpus is the deterministic
    private corpus (make_synthetic_corpus(cfg, cfg.seed)), so any two
    calls with the same (member_subset, seed) train the same model.
    """
    import torch
    corpus = make_synthetic_corpus(cfg, cfg.seed)
    ids = torch.tensor([corpus["ids"][i] for i in member_subset],
                       dtype=torch.long)
    labels = torch.tensor([corpus["labels"][i] for i in member_subset],
                          dtype=torch.long)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _build_model(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        for _ in range(cfg.steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(ids), labels)
            loss.backward()
            optimizer.step()
    return model


def train_on_corpus(cfg, corpus, subset, seed):
    """train_shadow on an explicit corpus (the public corpus of the
    phase bundle); same fork_rng/manual_seed discipline."""
    import torch
    ids = torch.tensor([corpus["ids"][i] for i in subset], dtype=torch.long)
    labels = torch.tensor([corpus["labels"][i] for i in subset],
                          dtype=torch.long)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _build_model(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        for _ in range(cfg.steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(ids), labels)
            loss.backward()
            optimizer.step()
    return model


def example_nll(model, ids, labels):
    """Per-example negative log-likelihoods (a plain list of floats)."""
    import torch
    with torch.no_grad():
        t_ids = torch.tensor(ids, dtype=torch.long)
        t_labels = torch.tensor(labels, dtype=torch.long)
        nll = torch.nn.functional.cross_entropy(
            model(t_ids), t_labels, reduction="none")
    return [float(v) for v in nll]


def _shadow_subsets(cfg, members):
    """Per-shadow random halves of the members (seed cfg.seed + 7919*j),
    deterministically repaired so every member is IN >=2 and OUT >=2
    shadows when n_shadows >= 4 (the record schema needs >=2 references
    per hypothesis). With n_shadows < 4 that guarantee is impossible and
    no repair is attempted (documented; RMIA only needs ref_out)."""
    subsets = []
    for j in range(cfg.n_shadows):
        rng = random.Random(cfg.seed + 7919 * j)
        subsets.append(set(rng.sample(list(members), len(members) // 2)))
    if cfg.n_shadows >= 4:
        for m in members:
            while sum(m in s for s in subsets) < 2:
                j = min((j for j in range(cfg.n_shadows)
                         if m not in subsets[j]),
                        key=lambda j: (len(subsets[j]), j))
                subsets[j].add(m)
            while sum(m in s for s in subsets) > cfg.n_shadows - 2:
                j = max((j for j in range(cfg.n_shadows)
                         if m in subsets[j]),
                        key=lambda j: (len(subsets[j]), -j))
                subsets[j].discard(m)
    return [sorted(s) for s in subsets]


def build_lira_records(cfg):
    """Train the target + shadow ensemble and emit lira_membership
    offline records: one dict per example with keys example_id, split
    ("in" = target member, "out" = never trained on), statistic (the
    per-example NLL under the target model), ref_in (members only: NLLs
    under the shadows trained WITH the example, omitted if fewer than 2)
    and ref_out (NLLs under the shadows trained WITHOUT it, omitted if
    fewer than 2). See the module docstring for the design."""
    corpus = make_synthetic_corpus(cfg, cfg.seed)
    members = corpus["members"]
    target = train_shadow(cfg, members, cfg.seed - 1)
    target_nll = example_nll(target, corpus["ids"], corpus["labels"])
    subsets = _shadow_subsets(cfg, members)
    shadow_nlls = []
    for j, subset in enumerate(subsets):
        model = train_shadow(cfg, subset, cfg.seed + 7919 * j)
        shadow_nlls.append(example_nll(model, corpus["ids"],
                                       corpus["labels"]))
    all_in_nlls = []
    for ref_seed in (cfg.seed + 31, cfg.seed + 37):
        model = train_shadow(cfg, list(range(cfg.n_examples)), ref_seed)
        all_in_nlls.append(example_nll(model, corpus["ids"],
                                       corpus["labels"]))
    member_set = set(members)
    records = []
    for i in range(cfg.n_examples):
        rec = {"example_id": f"shadow-{i:06d}",
               "split": "in" if i in member_set else "out",
               "statistic": target_nll[i]}
        ref_in = [nll[i] for nll in all_in_nlls] + \
                 [shadow_nlls[j][i] for j in range(cfg.n_shadows)
                  if i in subsets[j]]
        ref_out = [shadow_nlls[j][i] for j in range(cfg.n_shadows)
                   if i not in subsets[j]]
        if len(ref_in) >= 2:
            rec["ref_in"] = ref_in
        if len(ref_out) >= 2:
            rec["ref_out"] = ref_out
        records.append(rec)
    return records


def load_lm_corpus(path, seq_len, tokenizer=None):
    """Load a real-text corpus for the LM shadow path.

    JSONL, one row per document: {"id": ..., "ids": [int, ...]}
    (pre-tokenized) or {"id": ..., "text": ...} (needs a tokenizer).
    Every document must carry at least seq_len tokens; rows are truncated
    to seq_len. Returns [{"id": str, "ids": [int] * seq_len}]."""
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids = rec.get("ids")
            if ids is None:
                if tokenizer is None:
                    raise SystemExit(
                        f"[shadow-train] {path}:{lineno}: row carries "
                        "'text' but no 'ids'; pass --model so the "
                        "tokenizer can encode it")
                ids = tokenizer.encode(rec["text"],
                                       add_special_tokens=False)
            if len(ids) < seq_len:
                raise SystemExit(
                    f"[shadow-train] {path}:{lineno}: document "
                    f"{rec.get('id')!r} has {len(ids)} tokens < seq_len "
                    f"{seq_len}")
            rows.append({"id": str(rec.get("id", f"doc-{lineno:06d}")),
                         "ids": [int(t) for t in ids[:seq_len]]})
    if len(rows) < 4:
        raise SystemExit(f"[shadow-train] {path}: need >=4 documents "
                         f"(got {len(rows)})")
    return rows


def _train_lm(cfg, corpus_ids, subset, seed, model_path, device, lr):
    """Fine-tune a fresh pretrained causal LM on the given corpus rows.

    Full-batch Adam for cfg.steps updates under the fork_rng/manual_seed
    discipline of train_shadow (CUDA RNG included when device is cuda).
    Every call reloads the pretrained weights, so (subset, seed) fully
    determines the shadow."""
    import torch
    from transformers import AutoModelForCausalLM
    ids = torch.tensor([corpus_ids[i] for i in subset], dtype=torch.long,
                       device=device)
    devices = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(cfg.steps):
            optimizer.zero_grad()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            optimizer.step()
    model.eval()
    return model


def lm_example_nll(model, corpus_ids, device):
    """Mean next-token negative log-likelihood per document (floats)."""
    import torch
    with torch.no_grad():
        t = torch.tensor(corpus_ids, dtype=torch.long, device=device)
        logits = model(t).logits[:, :-1]
        targets = t[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            reduction="none").reshape(t.shape[0], -1).mean(dim=1)
    return [float(v) for v in nll]


def build_lira_records_lm(cfg, rows, model_path, device="cpu", lr=5e-5):
    """build_lira_records over a real-text corpus with causal-LM shadows.

    Half the documents are members of the target's fine-tune set (seeded
    draw); the target and every shadow are fresh copies of the pretrained
    model at model_path fine-tuned with the same recipe. The statistic is
    the mean next-token NLL under the target; ref_in/ref_out come from
    the leave-one-out shadows and two all-in references, exactly as in
    build_lira_records. example_id is the corpus document id."""
    corpus_ids = [r["ids"] for r in rows]
    n = len(rows)
    members = sorted(random.Random(cfg.seed).sample(range(n), n // 2))
    subsets = _shadow_subsets(cfg, members)
    target = _train_lm(cfg, corpus_ids, members, cfg.seed - 1,
                       model_path, device, lr)
    target_nll = lm_example_nll(target, corpus_ids, device)
    del target
    shadow_nlls = []
    for j, subset in enumerate(subsets):
        model = _train_lm(cfg, corpus_ids, subset, cfg.seed + 7919 * j,
                          model_path, device, lr)
        shadow_nlls.append(lm_example_nll(model, corpus_ids, device))
        del model
    all_in_nlls = []
    for ref_seed in (cfg.seed + 31, cfg.seed + 37):
        model = _train_lm(cfg, corpus_ids, list(range(n)), ref_seed,
                          model_path, device, lr)
        all_in_nlls.append(lm_example_nll(model, corpus_ids, device))
        del model
    member_set = set(members)
    records = []
    for i in range(n):
        rec = {"example_id": rows[i]["id"],
               "split": "in" if i in member_set else "out",
               "statistic": target_nll[i]}
        ref_in = [nll[i] for nll in all_in_nlls] + \
                 [shadow_nlls[j][i] for j in range(cfg.n_shadows)
                  if i in subsets[j]]
        ref_out = [shadow_nlls[j][i] for j in range(cfg.n_shadows)
                   if i not in subsets[j]]
        if len(ref_in) >= 2:
            rec["ref_in"] = ref_in
        if len(ref_out) >= 2:
            rec["ref_out"] = ref_out
        records.append(rec)
    return records


def build_phase_bundle(cfg):
    """Full representation_privacy.v1 bundle from the internal shadow
    ensemble: provenance, calibration/evaluation rows over a synthetic
    public corpus (released = the target model's mean token embedding,
    [1, dim] per document), phase_provenance (n_shadows + 3 attacker-owned
    reference model IDs, references_attacker_owned=True, statistic
    "negative_log_likelihood") and phase_records. Half the members are
    marked public_exposed (mirroring a private record that also appeared
    in public data) so both exposure strata contain members and matched
    non-members, as validate_phase requires. Phase truth is evaluation
    metadata only; the reference NLLs are fitted without it."""
    corpus = make_synthetic_corpus(cfg, cfg.seed)
    public_cfg = replace(cfg, n_examples=cfg.n_public)
    public = make_synthetic_corpus(public_cfg, cfg.seed + 101)
    members = corpus["members"]
    member_set = set(members)
    subsets = _shadow_subsets(cfg, members)
    target = train_shadow(cfg, members, cfg.seed - 1)
    target_nll = example_nll(target, corpus["ids"], corpus["labels"])
    target_public_nll = example_nll(target, public["ids"],
                                    public["labels"])
    ref_ids, ref_nlls, ref_public_nlls = [], [], []
    for j, subset in enumerate(subsets):
        model = train_shadow(cfg, subset, cfg.seed + 7919 * j)
        ref_ids.append(f"shadow-loo-{j}")
        ref_nlls.append(example_nll(model, corpus["ids"],
                                    corpus["labels"]))
        ref_public_nlls.append(example_nll(model, public["ids"],
                                           public["labels"]))
    all_in_nlls, all_in_public_nlls = [], []
    for k, ref_seed in enumerate((cfg.seed + 31, cfg.seed + 37)):
        model = train_on_corpus(
            cfg, {"ids": corpus["ids"] + public["ids"],
                  "labels": corpus["labels"] + public["labels"]},
            list(range(cfg.n_examples + cfg.n_public)), ref_seed)
        ref_ids.append(f"shadow-allin-{k}")
        all_in_nlls.append(example_nll(model, corpus["ids"],
                                       corpus["labels"]))
        all_in_public_nlls.append(example_nll(model, public["ids"],
                                              public["labels"]))

    def embedding_rows(corp, count, prefix):
        import torch
        with torch.no_grad():
            t_ids = torch.tensor(corp["ids"][:count], dtype=torch.long)
            emb = target[0](t_ids)
        rows = []
        for i in range(count):
            rows.append({"document_id": f"{prefix}-{i:06d}",
                         "released": [emb[i].tolist()],
                         "label": corp["labels"][i]})
        return rows

    half = cfg.n_public // 2
    bundle = {
        "schema": "representation_privacy.v1",
        "provenance": {
            "source": "internal shadow_train synthetic public corpus",
            "defender": "no trained defender (internal shadow ensemble)",
            "release_stage": "synthetic shadow features"},
        "calibration": embedding_rows(public, half, "calib"),
        "evaluation": embedding_rows(
            {"ids": public["ids"][half:], "labels": public["labels"][half:]},
            cfg.n_public - half, "eval"),
        "phase_provenance": {
            "public_dataset": "shadow-train synthetic public corpus",
            "private_dataset": "shadow-train synthetic private corpus",
            "target_model": "shadow-target",
            "reference_models": ref_ids,
            "references_attacker_owned": True,
            "statistic": "negative_log_likelihood"},
        "phase_records": []}

    def ref_out_for(i):
        return [ref_nlls[j][i] for j in range(cfg.n_shadows)
                if i not in subsets[j]]

    for i in range(cfg.n_examples):
        phase = ("private_training_member" if i in member_set
                 else "never_seen")
        bundle["phase_records"].append({
            "document_id": f"phase-private-{i:06d}", "phase": phase,
            "public_exposed": bool(i in member_set and
                                   members.index(i) % 2 == 0),
            "statistic": target_nll[i],
            "ref_in": [nll[i] for nll in all_in_nlls],
            "ref_out": ref_out_for(i)})
    for i in range(8):
        bundle["phase_records"].append({
            "document_id": f"phase-public-{i:06d}", "phase": "public_only",
            "public_exposed": True,
            "statistic": target_public_nll[i],
            "ref_in": [nll[i] for nll in all_in_public_nlls],
            "ref_out": [nll[i] for nll in ref_public_nlls]})
    return bundle


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, metavar="JSONL",
                    help="LIRA records output (one JSON record per line)")
    ap.add_argument("--phase-bundle", default=None, metavar="JSON",
                    help="also write the representation_privacy.v1 phase "
                         "bundle to this path")
    ap.add_argument("--n-shadows", type=int, default=ShadowConfig.n_shadows,
                    help="leave-one-out shadow models (>=2)")
    ap.add_argument("--steps", type=int, default=ShadowConfig.steps,
                    help="full-batch Adam updates per model")
    ap.add_argument("--seed", type=int, default=ShadowConfig.seed,
                    help="base seed (corpus, target and shadow draws)")
    ap.add_argument("--corpus-jsonl", default=None, metavar="JSONL",
                    help="real-text corpus for the LM shadow path "
                         "({id, ids} or {id, text} per line); replaces "
                         "the synthetic MLP corpus")
    ap.add_argument("--model", default=None, metavar="PATH",
                    help="pretrained causal-LM path for --corpus-jsonl "
                         "(shadows are fresh fine-tunes of this model)")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                    help="--corpus-jsonl training device")
    ap.add_argument("--lm-lr", type=float, default=5e-5,
                    help="--corpus-jsonl Adam learning rate")
    ap.add_argument("--seq-len", type=int, default=128,
                    help="--corpus-jsonl tokens per document (truncate)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    from .attacks.common import require_torch
    require_torch("shadow-train")
    if args.n_shadows < 2:
        raise SystemExit("[shadow-train] --n-shadows must be >=2 (the "
                         "per-record schema needs >=2 reference NLLs)")
    cfg = ShadowConfig(n_shadows=args.n_shadows, steps=args.steps,
                       seed=args.seed)
    if args.corpus_jsonl:
        if not args.model:
            raise SystemExit("[shadow-train] --corpus-jsonl requires "
                             "--model (the pretrained LM every shadow "
                             "fine-tunes from)")
        cfg = replace(cfg, seq_len=args.seq_len)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        rows = load_lm_corpus(args.corpus_jsonl, args.seq_len, tokenizer)
        records = build_lira_records_lm(cfg, rows, args.model,
                                        device=args.device, lr=args.lm_lr)
    else:
        records = build_lira_records(cfg)
    with open(args.output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[shadow-train] wrote {len(records)} records "
          f"({sum(r['split'] == 'in' for r in records)} members) "
          f"to {args.output}")
    if args.phase_bundle:
        bundle = build_phase_bundle(cfg)
        with open(args.phase_bundle, "w") as f:
            json.dump(bundle, f)
        print(f"[shadow-train] wrote phase bundle "
              f"({len(bundle['phase_records'])} phase records, "
              f"{len(bundle['phase_provenance']['reference_models'])} "
              f"attacker-owned reference models) to {args.phase_bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
