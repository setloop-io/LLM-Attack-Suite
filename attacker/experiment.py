"""Reproducible inference and split-training experiment; see root README."""
import argparse
import collections
import copy
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from .evidence import EvidenceLogger, file_sha256, verify_evidence

SCHEMA = "llm_attacker.experiment.v1"


def interval(values, comparisons=1):
    """Two-sided seed-level Student interval, Bonferroni within this run.

    Conditional on this fixed corpus. Tokens are not independent trials.
    No interval is claimed with fewer than three independent run seeds.
    """
    mean = statistics.mean(values) if values else None
    if len(values) < 3:
        return {"mean": mean, "lower": None, "upper": None,
                "n_seeds": len(values), "status": "INSUFFICIENT_EVIDENCE"}
    from scipy.stats import t
    half = float(t.ppf(1 - .05 / (2 * comparisons), len(values)-1)) * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lower": mean-half, "upper": mean+half,
            "n_seeds": len(values), "status": "ESTIMATED"}


def summarize(records):
    groups = collections.defaultdict(list)
    for rec in records:
        groups[(rec["mode"], rec["defense"], rec["arm"])].append(rec)
    baseline = {(r["mode"], r["arm"], r["seed"]): r for r in records if r["defense"] == "none"}
    out = []
    comparisons = max(1, 3 * len(groups))
    for (mode, defense, arm), cells in sorted(groups.items()):
        delta = [100*(r["accuracy"]-r["control_accuracy"]) for r in cells]
        paired = [100*(r["accuracy"]-baseline[(mode, arm, r["seed"])]["accuracy"])
                  for r in cells if (mode, arm, r["seed"]) in baseline]
        out.append({"mode": mode, "defense": defense, "arm": arm,
                    "attack_minus_control_pp": interval(delta, comparisons),
                    "defense_minus_undefended_pp": interval(paired, comparisons),
                    "attack_minus_shuffled_pair_pp": interval([100*(r["accuracy"]-r["shuffled_pair_accuracy"]) for r in cells if r.get("shuffled_pair_accuracy") is not None], comparisons),
                    "accuracy_mean": statistics.mean(r["accuracy"] for r in cells),
                    "n_seed_runs": len(cells), "comparison_family_size": comparisons,
                    "inference_scope": "seed variation conditional on fixed documents; no population or universal-security claim"})
    return out


def load_documents(path):
    docs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(docs) < 9 or any(not isinstance(d, dict) or not isinstance(d.get("id"), str)
                          or not isinstance(d.get("text"), str) or not d["text"].strip() for d in docs):
        raise ValueError("corpus needs >=9 JSONL documents with string id and nonempty text")
    if len({d["id"] for d in docs}) != len(docs) or len({d["text"].strip() for d in docs}) != len(docs):
        raise ValueError("duplicate document IDs or exact texts; deduplicate before splitting")
    return docs


def partition(docs, seed):
    docs = sorted(docs, key=lambda d: d["id"])
    random.Random(seed).shuffle(docs)
    n = len(docs)//3
    return {"calibration": docs[:n], "victim_train": docs[n:2*n], "utility_eval": docs[2*n:]}


def utility(split, cfg, batches):
    import torch
    import torch.nn.functional as F
    from . import splitbench as sb
    split.model.eval()
    torch.manual_seed(cfg.seed + 70001)
    state = sb.make_defense_state(cfg, split)
    losses = []
    with torch.no_grad():
        for ids in batches:
            ids = ids.to(split.device)
            h = sb._apply_seam_defense(split.head_forward(sb._apply_input_defense(ids, cfg, state)), cfg, state)
            logits = sb._server_rest_forward(split, h, cfg, state)
            losses.append(float(F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten())))
    return {"cross_entropy": statistics.mean(losses), "per_document_cross_entropy": losses,
            "perplexity": math.exp(min(700, statistics.mean(losses)))}


def run(args):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
    from . import splitbench as sb
    from .attacks import seam_inversion as attack
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must be unique")
    if not args.smoke and (not args.model or not args.corpus or len(args.seeds) < 3):
        raise ValueError("research runs require --model, --corpus, and >=3 distinct --seeds")
    if args.smoke and (args.model or args.corpus):
        raise ValueError("--smoke uses a random tiny model and synthetic documents only")
    if args.seq_len < 2 or min(args.train_steps, args.decode_steps, args.attack_steps, args.calibration_seqs) < 1:
        raise ValueError("sequence length >=2 and positive step/calibration budgets required")
    if args.sigma < 0 or not math.isfinite(args.sigma):
        raise ValueError("sigma must be finite and nonnegative")
    root = Path(args.output).resolve()
    if root.exists():
        raise ValueError("output directory already exists; use a fresh directory")
    root.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if args.smoke:
        docs = [{"id": f"synthetic-{i}", "text": str(i)} for i in range(18)]
        tokenizer = None
    else:
        docs = load_documents(args.corpus)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    partitions = partition(docs, args.data_seed)
    encoded = {}
    for name, part in partitions.items():
        encoded[name] = []
        for doc in part:
            if args.smoke:
                rng = random.Random(int(doc["id"].split("-")[-1])+args.data_seed)
                ids = [rng.randrange(64) for _ in range(args.seq_len)]
            else:
                ids = tokenizer.encode(doc["text"], add_special_tokens=False)
                if len(ids) < args.seq_len:
                    raise ValueError(f"document {doc['id']} shorter than seq_len; curate corpus before running")
                ids = ids[:args.seq_len]
            encoded[name].append(torch.tensor([ids], dtype=torch.long))
    split_manifest = {k: [{"id": d["id"], "text_sha256": hashlib.sha256(d["text"].encode()).hexdigest(),
                           "token_sha256": hashlib.sha256(json.dumps(t.tolist()).encode()).hexdigest()}
                          for d, t in zip(partitions[k], encoded[k])] for k in partitions}
    (root/"partitions.json").write_text(json.dumps(split_manifest, indent=2))
    cal = torch.cat(encoded["calibration"][:args.calibration_seqs])
    torch.save(cal, root/"calibration.pt")
    report = {"schema": SCHEMA, "config": vars(args), "results": [], "summary": [],
              "evidence_tier": "engineering_smoke" if args.smoke else "local_model_experiment",
              "status": "RUNNING", "partitions": split_manifest,
              "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                          "device": args.device, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                          "cpu_threads": torch.get_num_threads()},
              "threat_model": "seam observer with frozen pre-training public head and embeddings; public calibration documents; truth is scoring-only",
              "limits": ["single-process capture rig, not distributed deployment", "fixed document sample; intervals conditional on corpus",
                         "local attack implementations, not exact replications of published algorithms", "optimizer budgets differ from beam-search budgets; no equal-cost ranking"],
              "corpus_sha256": file_sha256(args.corpus) if args.corpus else None}
    started = time.perf_counter()
    artifact = root/"experiment.json"
    ev = EvidenceLogger(artifact, extra={"protocol": SCHEMA, "evidence_tier": report["evidence_tier"]})
    try:
        for seed in args.seeds:
            torch.manual_seed(seed)
            if args.smoke:
                base = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=max(64, args.seq_len+args.decode_steps),
                        n_embd=32, n_layer=2, n_head=4, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.))
            else:
                base = AutoModelForCausalLM.from_pretrained(args.model)
                if base.config.model_type != "gpt2":
                    raise ValueError("this protocol currently supports GPT-2-family public-head exports; splitbench separately supports Mistral/Qwen")
            public = root/f"public-head-{seed}.pt"
            attack.save_head_pt(str(public), base, int(args.split))
            model_hash = hashlib.sha256()
            for name, tensor in sorted(base.state_dict().items()):
                model_hash.update(name.encode()); model_hash.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            for defense in args.defenses:
                torch.manual_seed(seed)
                model = copy.deepcopy(base).to(args.device)
                split = sb.build_split(model, sb.SplitPlan.parse(args.split, n_layers=model.config.n_layer))
                folder = root/f"seed-{seed}"/defense
                folder.mkdir(parents=True)
                cfg = sb.BenchConfig(split=args.split, steps=args.train_steps, seq_len=args.seq_len,
                      batch_size=1, lr=args.lr, seed=seed, defense=defense, sigma=args.sigma if defense=="gaussian" else 0.,
                      save_canonical=True, device=args.device)
                before = utility(split, cfg, encoded["utility_eval"])
                order = list(range(len(encoded["victim_train"])))
                random.Random(seed).shuffle(order)
                batches = [encoded["victim_train"][i] for i in order]
                cap_train = folder/"training"
                initial_client = next(iter(split.blocks[0].parameters())).detach().clone()
                began = time.perf_counter()
                with sb.CaptureWriter(str(cap_train), "training", f"seed-{seed}-{defense}") as writer:
                    train_stats = sb.finetune(split, cfg, batches, writer)
                train_stats["wall_seconds"] = time.perf_counter()-began
                train_stats["tokens_processed"] = args.train_steps * args.seq_len
                train_stats["tokens_per_second_including_capture_io"] = train_stats["tokens_processed"] / train_stats["wall_seconds"]
                train_stats["client_parameter_change_l2"] = float((next(iter(split.blocks[0].parameters())).detach()-initial_client).norm())
                after = utility(split, cfg, encoded["utility_eval"])
                cap_infer = folder/"inference"
                began = time.perf_counter()
                with sb.CaptureWriter(str(cap_infer), "inference", f"seed-{seed}-{defense}") as writer:
                    infer_stats = sb.infer(split, dataclasses.replace(cfg, mode="inference", steps=args.decode_steps), encoded["utility_eval"], writer)
                infer_stats["wall_seconds"] = time.perf_counter()-began
                infer_stats["generated_tokens"] = args.decode_steps * len(encoded["utility_eval"])
                infer_stats["generated_tokens_per_second_including_prefill_and_io"] = infer_stats["generated_tokens"] / infer_stats["wall_seconds"]
                for mode, cap, arms in (("training", cap_train, args.arms), ("inference", cap_infer, [a for a in args.arms if a != "joint"])):
                    for arm in arms:
                        output = folder/f"{mode}-{arm}.json"
                        argv = ["--arm", arm, "--capture-dir", str(cap), "--canonical-pt", str(cap/"canonical.pt"),
                                "--head-pt", str(public), "--calibration-pt", str(root/"calibration.pt"),
                                "--linearization", "--steps", str(args.attack_steps), "--seed", str(seed),
                                "--samples", str(args.train_steps if mode=="training" else len(encoded["utility_eval"])),
                                "--calibration-seqs", str(len(cal)), "--device", args.device, "--output", str(output),
                                "--joint-beam", str(args.joint_beam), "--joint-candidates", str(args.joint_candidates)]
                        a = attack.build_parser().parse_args(argv); a.mode = mode
                        a.seeds = list(args.seeds)
                        if args.adaptive and defense == "gaussian":
                            a.adaptive = "noise-whitening"; a.defense_sigma = args.sigma
                        torch.manual_seed(seed)
                        began = time.perf_counter()
                        rc = attack.run(a)
                        if args.device == "cuda": torch.cuda.synchronize()
                        elapsed = time.perf_counter()-began
                        raw = json.loads(output.read_text())
                        if rc or not raw["summary"] or any("error" in c for c in raw["results"]):
                            raise RuntimeError(f"attack incomplete: {output}")
                        seqs = [q for c in raw["results"] for q in c.get("per_sequence", [])]
                        expected = args.train_steps if mode == "training" else len(encoded["utility_eval"])
                        if len(seqs) != expected:
                            raise RuntimeError(f"expected {expected} scored sequences, got {len(seqs)}")
                        for q in seqs:
                            doc_index = order[q["index"] % len(order)] if mode == "training" else q["index"]
                            q["document_id"] = partitions["victim_train" if mode == "training" else "utility_eval"][doc_index]["id"]
                        truths = {q["document_id"]: q["truth"] for q in seqs}
                        document_ids = sorted(truths)
                        random.Random(seed + 9103).shuffle(document_ids)
                        deranged = dict(zip(document_ids, document_ids[1:]+document_ids[:1]))
                        for q in seqs:
                            q["shuffled_pair_correct"] = (sum(a == b for a, b in zip(q["prediction"], truths[deranged[q["document_id"]]]))
                                                           if len(document_ids) > 1 else None)
                        total = sum(q["n_tokens"] for q in seqs)
                        rec = {"seed": seed, "defense": defense, "mode": mode, "arm": arm,
                               "public_model_sha256": model_hash.hexdigest(), "attack_seconds": elapsed,
                               "accuracy": sum(q["correct"] for q in seqs)/total,
                               "control_accuracy": sum(q["control_correct"] for q in seqs)/total,
                               "shuffled_pair_accuracy": sum(q["shuffled_pair_correct"] for q in seqs)/total if len(document_ids)>1 else None,
                               "shuffled_pair_scope": "document derangement diagnostic; not an independent replicate",
                               "exact_sequence_match": sum(q["correct"]==q["n_tokens"] for q in seqs)/len(seqs),
                               "n_tokens": total, "n_sequences": len(seqs), "per_sequence": seqs,
                               "attack_summary": raw["summary"], "utility_before": before, "utility_after": after,
                               "training": train_stats, "inference": infer_stats,
                               "attack_config": vars(a), "artifact": str(output.relative_to(root))}
                        report["results"].append(rec); ev.event("cell_complete", {k:v for k,v in rec.items() if k != "per_sequence"})
                        artifact.write_text(json.dumps(report, indent=2))
                        print(f"[experiment] seed={seed} defense={defense} {mode}/{arm}: accuracy={rec['accuracy']:.4f}", flush=True)
                del split, model
        report["summary"] = summarize(report["results"])
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        ev.event("experiment_error", report["error"])
        raise
    finally:
        report["wall_seconds"] = time.perf_counter()-started
        import resource, sys
        report["resources"] = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=="darwin" else 1024),
                               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device=="cuda" else None,
                               "scope": "entire experiment process; timings include capture I/O, not serving latency"}
        excluded = {artifact, Path(ev.events_path), Path(ev.manifest_path)}
        report["files"] = [{"path": str(p.relative_to(root)), "sha256": file_sha256(p), "bytes": p.stat().st_size}
                           for p in sorted(root.rglob("*")) if p.is_file() and p not in excluded]
        artifact.write_text(json.dumps(report, indent=2, allow_nan=False))
        ev.finalize()
    ok, problems = verify_evidence(ev.manifest_path)
    if not ok: raise RuntimeError(problems)
    print(f"[experiment] {report['status']}: {artifact}")
    return 0


def verify_run(root, expected_manifest_sha256=None):
    """Check the seal, completion, and every component hash after relocation."""
    root = Path(root).resolve()
    ok, problems = verify_evidence(root/"experiment.json.evidence.json", expected_manifest_sha256)
    try:
        report = json.loads((root/"experiment.json").read_text())
        if report.get("status") != "COMPLETE":
            problems.append("experiment incomplete")
        for entry in report["files"]:
            path = (root/entry["path"]).resolve()
            if not path.is_relative_to(root):
                problems.append(f"component outside bundle: {entry['path']}")
            elif not path.is_file() or file_sha256(path) != entry["sha256"]:
                problems.append(f"component hash mismatch: {entry['path']}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"experiment manifest invalid: {exc}")
    return not problems, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output")
    ap.add_argument("--verify", metavar="RUN_DIRECTORY")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--corpus", help="JSONL {id,text}; >=9 unique documents, each >=seq-len tokens")
    ap.add_argument("--seeds", nargs="+", type=int, default=[11,22,33])
    ap.add_argument("--data-seed", type=int, default=2026)
    ap.add_argument("--defenses", nargs="+", choices=["none","gaussian"], default=["none","gaussian"])
    ap.add_argument("--arms", nargs="+", choices=["vmi","wbi","joint"], default=["vmi","wbi","joint"])
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=3)
    ap.add_argument("--decode-steps", type=int, default=2)
    ap.add_argument("--attack-steps", type=int, default=10)
    ap.add_argument("--calibration-seqs", type=int, default=64)
    ap.add_argument("--joint-beam", type=int, default=2)
    ap.add_argument("--joint-candidates", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--sigma", type=float, default=.1)
    ap.add_argument("--split", default="1")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--adaptive", action="store_true")
    args = ap.parse_args(argv)
    if args.verify:
        ok, problems = verify_run(args.verify)
        print(json.dumps({"ok": ok, "problems": problems}))
        return 0 if ok else 1
    if not args.output:
        ap.error("--output is required for experiments")
    if "none" not in args.defenses or "vmi" not in args.arms:
        ap.error("include none defense and vmi arm for paired baselines")
    if len(set(args.defenses)) != len(args.defenses) or len(set(args.arms)) != len(args.arms):
        ap.error("duplicate defenses/arms would inflate replication counts")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
