"""Local centroid and position-free signature baselines for latent bundles.

These use labeled public/training reference features and report token recovery
on separate evaluation rows. They are not implementations of the published
causal VMA algorithm. Unseen evaluation labels count as misses rather than
being removed from the denominator.

Permutation-ORDER recovery (the published metric, reported alongside token
top-1): when the bundle carries ground-truth row order —

    train_order   [train_blocks, tokens] int64 — true pre-permutation index
                  of each released reference row (attacker-side reference
                  knowledge, like the reference labels)
    eval_order    [eval_blocks, tokens] int64 — held-out scoring ground
                  truth, never consulted for matching

— each arm additionally reports order_recovery_pct, the fraction of eval
positions whose true index is correctly recovered, against the uniform-
permutation chance line 1/(frame width). Each arm predicts an order from
its OWN position-free matching geometry — the position channel stays
excluded, so recovery is never tautological (the signature-arm postmortem):

  vma_centroid_match    two-stage: the arm's centroid class prediction
                        restricts the reference set; within the predicted
                        class, cosine-nearest reference rows vote positions
                        (fallback: all reference rows at a position when the
                        class has none there)
  sorted_signature_matching_position_free
                        the arm's euclidean 1-NN geometry against reference
                        rows with known positions

Both then solve a per-frame linear assignment (rows -> positions) so the
recovered order is a bijection, as the published permutation-recovery
protocol requires; the solve needs scipy (research extra), imported only
when order ground truth is present. Bundles without usable order fields
get token metrics only — the omission is journaled per arm (fail-closed:
the metric is never fabricated).
"""

EXPERIMENT_ID = "latent_matching"
MODES = ("training",)
REQUIRES_LABELS = True
DESCRIPTION = ("Centroid vocabulary + position-free signature matching on "
               "latent release bundles: token top-1 and, with order ground "
               "truth, permutation-order recovery (v11)")


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--evidence", action="store_true",
                        help="seal the run with an attacker.evidence chain "
                             "(<output>.evidence.json[l]); requires --output")
    return parser


def _make_evidence(args):
    """--evidence wiring: one EvidenceLogger threaded through
    make_artifact/write_artifact (contract: default None = no change)."""
    if not getattr(args, "evidence", False):
        return None
    if not args.output:
        raise SystemExit(f"[{EXPERIMENT_ID}] --evidence requires --output")
    from ..evidence import EvidenceLogger
    return EvidenceLogger(args.output, node_role="attacker/latent-matching")


def _order_ground_truth(bundle, train_y, eval_y):
    """Validate the optional order fields. Returns ((train_order,
    eval_order), None) when usable, else (None, reason) — an unusable
    order field downgrades the metric to a journaled note, never a
    fabricated number."""
    if "train_order" not in bundle and "eval_order" not in bundle:
        return None, "bundle carries no order ground truth"
    if "train_order" not in bundle or "eval_order" not in bundle:
        return None, "order recovery needs BOTH train_order and eval_order"
    train_order = bundle["train_order"].long()
    eval_order = bundle["eval_order"].long()
    if train_order.shape != train_y.shape or eval_order.shape != eval_y.shape:
        return None, "order tensors must match the label tensor shapes"
    if train_y.shape[1] != eval_y.shape[1]:
        return None, ("train/eval frame widths differ; positions are not "
                      "comparable")
    width = eval_y.shape[1]
    for name, order in (("train_order", train_order),
                        ("eval_order", eval_order)):
        if order.numel() and (int(order.min()) < 0
                              or int(order.max()) >= width):
            return None, f"{name} indices out of range [0,{width})"
    return (train_order, eval_order), None


def _assign_frame(cost):
    """Min-cost bijection rows -> positions for one frame (the published
    permutation-recovery solve). scipy is a declared research-extra
    dependency, imported lazily so token-only runs never need it."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        raise SystemExit(
            f"[{EXPERIMENT_ID}] order recovery requires scipy (research "
            "extra); install it or run without order ground truth.")
    _, cols = linear_sum_assignment(cost)
    return cols


def _order_recovery(eval_sig, ref_sig, ref_pos, true_order, distance,
                    ref_class=None, pred_class=None):
    """Fraction of eval positions whose true index is correctly recovered.

    eval_sig [B,T,F] and ref_sig [R,F] are position-FREE signatures (the
    caller strips the position channel). ref_pos [R] holds the reference
    rows' true positions; true_order [B,T] is the held-out ground truth.
    distance is the arm's native geometry ("euclidean" for the signature
    arm, "cosine" for the centroid arm). ref_class/pred_class implement the
    centroid arm's two-stage resolve: each row's candidates are restricted
    to its predicted class, falling back to all reference rows at a
    position when the class has none there.
    """
    import torch
    frames, width = true_order.shape
    correct = 0
    for b in range(frames):
        if distance == "cosine":
            dist = 1.0 - (torch.nn.functional.normalize(eval_sig[b], dim=-1)
                          @ torch.nn.functional.normalize(ref_sig, dim=-1).T)
        else:
            dist = torch.cdist(eval_sig[b], ref_sig)
        cost = torch.full((width, width), float("inf"))
        for p in range(width):
            at_p = ref_pos == p
            if not bool(at_p.any()):
                continue
            if ref_class is None:
                cand = at_p.expand(width, -1)
            else:
                cand = at_p.unsqueeze(0) & (
                    ref_class.unsqueeze(0) == pred_class[b].unsqueeze(1))
                cand = torch.where(cand.any(-1, keepdim=True), cand,
                                   at_p.expand(width, -1))
            cost[:, p] = dist.masked_fill(~cand, float("inf")).min(-1).values
        finite = torch.isfinite(cost)
        if not bool(finite.any()):
            return None
        ceiling = float(cost[finite].max()) * 2.0 + 1.0
        cost = torch.where(finite, cost, torch.full_like(cost, ceiling))
        pred = _assign_frame(cost.double().numpy())
        correct += int((torch.from_numpy(pred) == true_order[b]).sum())
    return correct / max(1, frames * width)


def _order_fields(recovery, width, note=None):
    """The order block every arm reports. recovery=None means the metric
    was not identifiable on this bundle; the reason rides in order_note."""
    chance = 100.0 / max(1, width)
    if recovery is None:
        return {"order_recovery_pct": None, "order_chance_pct": chance,
                "order_excess_pp": None,
                "order_note": note or "order ground truth unavailable"}
    return {"order_recovery_pct": 100.0 * recovery,
            "order_chance_pct": chance,
            "order_excess_pp": 100.0 * recovery - chance,
            "order_note": note}


def run(args):
    from .common import require_torch
    require_torch(EXPERIMENT_ID)
    import torch
    from ..latent_features import latent_invariants
    from .. import artifacts

    evidence = _make_evidence(args)
    bundle = torch.load(args.bundle, map_location="cpu")
    train_x = bundle["train_wire"].float()
    train_y = bundle["train_tokens"].long()
    eval_x = bundle["eval_wire"].float()
    eval_y = bundle["eval_tokens"].long()

    if not train_y.numel() or not eval_y.numel() or train_y.min() < 0 or eval_y.min() < 0:
        raise ValueError("bundle must contain nonempty nonnegative token labels")
    classes = torch.unique(train_y)
    lookup = torch.full((int(max(classes.max(), eval_y.max())) + 1,), -1, dtype=torch.long)
    lookup[classes] = torch.arange(classes.numel())
    train_c = lookup[train_y]
    eval_c = lookup[eval_y]

    order, order_note = _order_ground_truth(bundle, train_y, eval_y)
    width = eval_y.shape[1]

    results = []

    def record(rec):
        results.append(rec)
        artifacts.append_jsonl(args.output, rec)
        if evidence is not None:
            evidence.event("cell_result", rec)

    def vma_arm():
        train_inv = latent_invariants(train_x)
        eval_inv = latent_invariants(eval_x)
        centroids = torch.zeros(classes.numel(), train_inv.shape[-1])
        for index in range(classes.numel()):
            mask = train_c.reshape(-1) == index
            if mask.any():
                centroids[index] = train_inv.reshape(-1, train_inv.shape[-1]
                                                     )[mask].mean(0)
        pred = (torch.nn.functional.normalize(eval_inv, dim=-1)
                @ torch.nn.functional.normalize(centroids, dim=-1).T
                ).argmax(-1)
        valid = eval_c.reshape(-1) >= 0
        top1 = float((pred.reshape(-1) == eval_c.reshape(-1)).float().mean())
        majority = _majority(train_c, eval_c)
        rec = {
            "arm": "vma_centroid_match", "top1_pct": 100.0 * top1,
            "majority_pct": 100.0 * majority,
            "excess_pp": 100.0 * (top1 - majority),
            "eval_rows": eval_c.numel(), "known_eval_fraction": float(valid.float().mean()),
        }
        if order is not None:
            train_order, eval_order = order
            recovery = _order_recovery(
                eval_inv[..., :-1],
                train_inv[..., :-1].reshape(-1, train_inv.shape[-1] - 1),
                train_order.reshape(-1), eval_order, "cosine",
                ref_class=train_c.reshape(-1), pred_class=pred)
            rec.update(_order_fields(recovery, width))
        else:
            rec.update(_order_fields(None, width, order_note))
        record(rec)

    def order_arm():
        train_inv = latent_invariants(train_x)[..., :-1]
        eval_inv = latent_invariants(eval_x)[..., :-1]
        half = max(1, train_x.shape[0] // 2)
        ref = train_inv[:half]
        ref_labels = train_c[:half]
        ref_flat = ref.reshape(-1, ref.shape[-1])
        ref_lab_flat = ref_labels.reshape(-1)
        correct = 0
        total = 0
        chunk = 64
        for start in range(0, eval_x.shape[0], chunk):
            sig = eval_inv[start:start + chunk]
            dist = torch.cdist(sig.reshape(-1, sig.shape[-1]), ref_flat)
            matched = ref_lab_flat[dist.argmin(-1)]
            true = eval_c[start:start + chunk].reshape(-1)
            valid = true >= 0
            correct += int((matched[valid] == true[valid]).sum())
            total += int(true.numel())
        top1 = correct / max(1, total)
        majority = _majority(train_c, eval_c)
        rec = {
            "arm": "sorted_signature_matching_position_free",
            "top1_pct": 100.0 * top1,
            "majority_pct": 100.0 * majority,
            "excess_pp": 100.0 * (top1 - majority),
            "eval_rows": total,
        }
        if order is not None:
            train_order, eval_order = order
            recovery = _order_recovery(
                eval_inv, ref_flat, train_order[:half].reshape(-1),
                eval_order, "euclidean")
            rec.update(_order_fields(recovery, width))
        else:
            rec.update(_order_fields(None, width, order_note))
        record(rec)

    vma_arm()
    order_arm()

    summary = {
        "arms": [r["arm"] for r in results],
        "worst_excess_pp": max(r["excess_pp"] for r in results),
        "order_recovery_evaluated": order is not None,
        "worst_order_excess_pp": max(
            (r["order_excess_pp"] for r in results
             if r["order_excess_pp"] is not None), default=None),
        "note": "Local centroid and signature baselines; not a replication of published VMA. Unseen labels count as misses. Permutation-order recovery is reported alongside token top-1 when the bundle carries train_order/eval_order ground truth; the position channel is excluded from matching, so recovery is never tautological.",
    }
    artifact = artifacts.make_artifact(
        "dtraining.attacker.latent_matching.v1",
        {"attack": EXPERIMENT_ID, "mode": "training",
         "restarts": args.restarts, "epochs": args.epochs},
        "Local centroid and signature token matching against "
        "gauged+chaffed latent bundles; descriptive recovery, not a privacy certificate.",
        evidence=evidence)
    artifact["results"].extend(results)
    artifact["summary"].append(summary)
    artifacts.write_artifact(args.output, artifact, evidence=evidence)
    import json
    print(json.dumps(summary, indent=1))
    return 0


def _majority(train_c, eval_c):
    import torch
    flat = train_c.reshape(-1)
    mode = int(torch.bincount(flat[flat >= 0]).argmax())
    valid = eval_c.reshape(-1) >= 0
    return float((eval_c.reshape(-1) == mode).float().mean())
