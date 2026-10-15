"""Prospective single-seam BiSR building blocks; not a fidelity-pass certificate.

Torch is imported lazily. Prediction accepts observations only; scoring is a
separate operation. Original dual-cut reproduction is deliberately unsupported.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

GRU_FORMAT = 'llm_attacker.bisr_gru.v1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def gru_class():
    import torch.nn as nn

    class SipGRU(nn.Module):
        """Causal single-layer GRU, explicit output dropout (not ignored GRU dropout)."""
        def __init__(self, vocab, seam_dim, seq_len, hidden_size=256, dropout=0.1):
            super().__init__()
            self.gru = nn.GRU(seam_dim, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(hidden_size, vocab)
            self.seq_len = seq_len

        def forward(self, rows):
            if rows.ndim != 3 or rows.shape[1] != self.seq_len:
                raise ValueError('GRU requires the declared full sequence length')
            hidden, _ = self.gru(rows)
            return self.output(self.dropout(hidden))
    return SipGRU


def train_gru(inv, fetch, count, *, epochs=20, batch_size=32, seed=0, lr=0.001):
    """Visit each declared auxiliary row once per seeded epoch, including tail."""
    import torch
    import torch.nn.functional as F
    if count < 1 or epochs < 1 or batch_size < 1:
        raise ValueError('Positive row, epoch and batch counts required')
    opt = torch.optim.AdamW(inv.parameters(), lr=lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    steps = 0
    inv.train()
    for epoch in range(epochs):
        order = torch.randperm(count, generator=generator)
        for start in range(0, count, batch_size):
            ids, rows = fetch(order[start:start + batch_size].tolist())
            loss = F.cross_entropy(inv(rows).flatten(0, 1), ids.flatten())
            if not torch.isfinite(loss):
                raise ValueError('Non-finite SIP training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
    return {'epochs': epochs, 'windows_per_epoch': count, 'steps': steps,
            'final_loss': float(loss.detach()), 'lr': lr, 'weight_decay': 0,
            'eps': 1e-8, 'betas': [0.9, 0.999], 'batch_size': batch_size}


class GrantedSplit:
    """Separate exact/public bottom from an always-public downstream network."""
    def __init__(self, head, public, grant):
        self.head = head
        self.public = public
        self.grant = grant

    def __getattr__(self, name):
        return getattr(self.head, name)

    def rest_forward(self, hidden, attention_mask=None):
        return self.public.rest_forward(hidden, attention_mask)


def granted_split(public_model, boundary, *, knowledge='public', head_path=None,
                  head_sha256=None):
    """Load a complete, hash-bound GPT-2 head only; reject missing/extra weights.

    Loading into a separate model avoids changing the tied PUBLIC lm_head when
    exact input embeddings differ. The caller supplies the pinned public model.
    """
    import torch
    from .splitbench import SplitPlan, build_split
    if public_model.config.model_type != 'gpt2':
        raise ValueError('Fidelity grants currently support GPT-2 only')
    if public_model.config._attn_implementation != 'eager':
        raise ValueError('BRE second derivatives require eager attention')
    n = public_model.config.n_layer
    if not isinstance(boundary, int) or not 0 < boundary < n:
        raise ValueError('An internal GPT-2 boundary is required')
    if knowledge not in ('public', 'exact'):
        raise ValueError('Unknown knowledge grant')
    if knowledge == 'public' and (head_path is not None or head_sha256 is not None):
        raise ValueError('Public grant forbids an exact-head checkpoint')
    if knowledge == 'exact' and (not head_path or not head_sha256):
        raise ValueError('Exact grant requires head path and expected SHA-256')
    public = deepcopy(public_model).eval().requires_grad_(False)
    head = deepcopy(public_model).eval().requires_grad_(False)
    if knowledge == 'exact':
        if sha256(head_path) != head_sha256:
            raise ValueError('Exact-head SHA-256 mismatch')
        blob = torch.load(head_path, map_location='cpu', weights_only=True)
        if (blob.get('format') != 'llm_attacker.seam_head.v1'
                or blob.get('arch') != 'gpt2' or blob.get('boundary') != boundary):
            raise ValueError('Wrong head format, architecture or boundary')
        for field in ('n_layer', 'n_embd', 'n_head', 'n_positions', 'vocab_size'):
            if blob.get('config', {}).get(field) != getattr(head.config, field):
                raise ValueError('Head architecture mismatch: ' + field)
        prefixes = ('transformer.wte.', 'transformer.wpe.') + tuple(
            f'transformer.h.{i}.' for i in range(boundary))
        expected = {k for k in head.state_dict() if k.startswith(prefixes)}
        state = blob.get('state_dict', {})
        if set(state) != expected:
            raise ValueError('Exact head must contain all and only granted parameters')
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError('Non-finite checkpoint')
        head.load_state_dict(state, strict=False)
    plan = SplitPlan.parse(str(boundary), n_layers=n)
    grant = {'knowledge': knowledge, 'boundary': boundary,
             'head_sha256': head_sha256, 'downstream': 'public',
             'scope': 'single-seam adaptation; not dual-cut reproduction'}
    return GrantedSplit(build_split(head, plan), build_split(public, plan), grant)


def validate_observations(bundle):
    """No labels, canonical paths, or unrestricted metadata enter prediction."""
    import torch
    expected = {'schema', 'sample_ids', 'h', 'g', 'mask', 'gradient_batch_scale'}
    if set(bundle) != expected or bundle['schema'] != 'bisr.observations.v1':
        raise ValueError('Observation schema forbids truth and unknown fields')
    h, g, mask = bundle['h'], bundle['g'], bundle['mask']
    if h.ndim != 3 or h.shape[1] < 2 or h.shape[0] < 1 or g.shape != h.shape:
        raise ValueError('Paired nonempty forward/backward observations required')
    if h.dtype != torch.float32 or g.dtype != torch.float32:
        raise ValueError('Conformance requires FP32 observations')
    if mask.shape != h.shape[:2] or not torch.all(mask == 1):
        raise ValueError('Only full unpadded windows are implemented; no silent mask loss')
    ids = bundle['sample_ids']
    if (len(ids) != h.shape[0] or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Unique nonempty canonical sample identifiers required')
    scale = bundle['gradient_batch_scale']
    if (scale.shape != (h.shape[0],) or not torch.isfinite(scale).all()
            or not torch.all(scale >= 1) or not torch.all(scale == scale.round())):
        raise ValueError('Recorded capture batch sizes must be positive integers')
    if not torch.isfinite(h).all() or not torch.isfinite(g).all():
        raise ValueError('Non-finite wire')


def predict_arms(split, inverter, bundle, args):
    """Predict SIP/b/f/bf without accepting or reading ground-truth tokens."""
    import torch
    from .attacks.bisr_baseline import _bre_gradient_match, _bre_smashed_match
    validate_observations(bundle)
    if args.opt_batch != 1:
        raise ValueError('Conformance uses opt_batch=1; batch norms change GM objective')
    budgets = (args.gm_lr, args.gm_beta, args.gm_tau, args.sm_lr, args.sm_wd)
    if (any(not math.isfinite(x) for x in budgets)
            or any(type(x) is not int or x < 1 for x in (args.gm_iters, args.sm_iters))
            or args.gm_tau <= 0 or not 0 <= args.gm_beta <= 1
            or args.gm_lr <= 0 or args.sm_lr <= 0 or args.sm_wd < 0):
        raise ValueError('Invalid optimisation budgets')
    device = split.device
    h = bundle['h'].to(device)
    g = bundle['g'].to(device) * bundle['gradient_batch_scale'].to(device)[:, None, None]
    inverter.eval().requires_grad_(False)
    with torch.no_grad():
        scores = inverter(h)
        if not torch.isfinite(scores).all():
            raise ValueError('Non-finite SIP logits')
        sip = scores.argmax(-1)
    E = split.wte.weight.detach()
    b, b_info = _bre_gradient_match(split, h, g, scores, iters=args.gm_iters,
        lr=args.gm_lr, beta=args.gm_beta, tau=args.gm_tau, opt_batch=1, upstream=True)
    f, f_info = _bre_smashed_match(split, E, h, sip, iters=args.sm_iters,
        lr=args.sm_lr, wd=args.sm_wd, opt_batch=1, upstream=True)
    bf, bf_info = _bre_smashed_match(split, E, h, b, iters=args.sm_iters,
        lr=args.sm_lr, wd=args.sm_wd, opt_batch=1, upstream=True)
    work = {'gm': b_info, 'f': f_info, 'bf_forward': bf_info,
            'bf_reuses_b_gradient_stage': True,
            'scope': 'GM downstream work separate from SM client-head work; not FLOPs'}
    if any(not math.isfinite(v) for rec in (b_info, f_info, bf_info)
           for v in rec.values() if isinstance(v, (int, float))):
        raise ValueError('Non-finite optimisation report')
    return {'schema': 'bisr.predictions.v1', 'sample_ids': bundle['sample_ids'],
            'algorithm_profile': 'splitllm-c376590',
            'grant': split.grant, 'arms': {k: v.cpu().tolist() for k, v in
            {'sip': sip, 'b': b, 'f': f, 'bf': bf}.items()}, 'work': work}


def write_predictions(path, record):
    """Exclusive-create export; never overwrite a historical prediction record."""
    data = (json.dumps(record, sort_keys=True, allow_nan=False) + '\n').encode()
    with open(path, 'xb') as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def replay_wire(split, token_ids, saved_h, saved_g):
    """Scorer-side mechanical check only; token IDs never enter predict_arms."""
    import torch
    import torch.nn.functional as F
    h = split.head_forward(token_ids).detach().requires_grad_(True)
    logits = split.rest_forward(h)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           token_ids[:, 1:].reshape(-1))
    g, = torch.autograd.grad(loss, h)
    if not torch.equal(h.detach(), saved_h) or not torch.equal(g, saved_g):
        raise ValueError('Stored-wire replay mismatch')
    return True


def numerical_check(candidate, reference):
    import torch
    if (candidate.shape != reference.shape or not torch.isfinite(candidate).all()
            or not torch.isfinite(reference).all()):
        raise ValueError('Shape or finite-value mismatch')
    if not torch.all((candidate - reference).abs() <= 1e-6 + 1e-5 * reference.abs()):
        raise ValueError('Numerical conformance tolerance exceeded')
    return True


def equivalence(seed_differences):
    """Per-endpoint fixed-document interval; callers must require all eight."""
    if len(seed_differences) != 3 or not all(math.isfinite(v) for v in seed_differences):
        raise ValueError('Exactly three finite paired seed differences required')
    mean = statistics.mean(seed_differences)
    half = 2.919985580355516 * statistics.stdev(seed_differences) / math.sqrt(3)
    lo, hi = mean - half, mean + half
    status = ('EQUIVALENT' if -0.01 < lo and hi < 0.01 else
              'NON_EQUIVALENT' if hi < -0.01 or lo > 0.01 else 'INCONCLUSIVE')
    return {'mean': mean, 'interval90': [lo, hi], 'status': status}


def score_predictions(record, truth, tokenizer):
    """Scorer-only metrics, preserving upstream per-token decoding semantics.

    Rouge package is pinned by the execution environment; empty decoded truth
    and metric failures block scoring rather than silently dropping examples.
    METEOR requires preinstalled WordNet; this function never downloads assets.
    """
    from rouge import Rouge
    from nltk.translate.meteor_score import meteor_score
    if record['schema'] != 'bisr.predictions.v1' or truth['schema'] != 'bisr.truth.v1':
        raise ValueError('Unknown prediction/truth schema')
    ids = record['sample_ids']
    if ids != truth['sample_ids'] or len(set(ids)) != len(ids) or not ids:
        raise ValueError('Sample identities/order mismatch')
    if set(record['arms']) != {'sip', 'b', 'f', 'bf'}:
        raise ValueError('All four arms are required')
    targets = truth['token_ids']
    if len(targets) != len(ids):
        raise ValueError('Missing truth rows')
    rouge = Rouge()
    result = {}
    for arm, predictions in record['arms'].items():
        if len(predictions) != len(targets):
            raise ValueError('Missing prediction rows')
        rows = []
        for sample_id, pred, target in zip(ids, predictions, targets):
            if not target or len(pred) != len(target):
                raise ValueError('Token alignment mismatch')
            hyp = ' '.join(tokenizer.decode(t, skip_special_tokens=True) for t in pred)
            ref = ' '.join(tokenizer.decode(t, skip_special_tokens=True) for t in target)
            if not ref.strip():
                raise ValueError('Empty decoded reference cannot be scored')
            hyp = hyp if hyp.strip() else '<EMP>'
            a, b = set(tokenizer.tokenize(hyp)), set(tokenizer.tokenize(ref))
            token_f1 = 2 * len(a & b) / (len(a) + len(b)) if a or b else 0.0
            row = {'sample_id': sample_id,
                   'positional_accuracy': sum(p == t for p, t in zip(pred, target)) / len(target),
                   'rouge_l_f1': rouge.get_scores(hyp, ref, avg=True)['rouge-l']['f'],
                   'meteor': meteor_score([ref.split()], hyp.split()),
                   'token_set_f1': token_f1}
            if not all(math.isfinite(v) and 0 <= v <= 1 for k, v in row.items() if k != 'sample_id'):
                raise ValueError('Invalid metric')
            rows.append(row)
        result[arm] = rows
    return result


def compare_scores(candidate, reference, seeds=(31415, 27182, 16180), documents=256):
    """Aggregate all eight endpoints; no seed, cell or sample deletion allowed.

    Inputs map seed strings to scorer outputs. This statistical result alone
    cannot admit a campaign: mechanical and provenance checks remain separate.
    """
    keys = {str(s) for s in seeds}
    if len(keys) != 3 or set(candidate) != keys or set(reference) != keys:
        raise ValueError('Exactly the specified three paired seeds required')
    endpoints = {}
    common_ids = None
    for arm in ('sip', 'b', 'f', 'bf'):
        for metric in ('positional_accuracy', 'rouge_l_f1'):
            differences = []
            for seed in seeds:
                c, r = candidate[str(seed)], reference[str(seed)]
                if set(c) != {'sip', 'b', 'f', 'bf'} or set(r) != set(c):
                    raise ValueError('Incomplete arm set')
                cr, rr = c[arm], r[arm]
                ids = [v['sample_id'] for v in cr]
                if (len(cr) != documents or len(set(ids)) != documents
                        or ids != [v['sample_id'] for v in rr]):
                    raise ValueError('Incomplete or unpaired documents')
                if common_ids is None:
                    common_ids = ids
                if ids != common_ids:
                    raise ValueError('Document set/order differs across seeds/arms')
                for v in cr + rr:
                    score = v[metric]
                    if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError('Non-finite or out-of-range score')
                differences.append(statistics.mean(a[metric] - b[metric] for a, b in zip(cr, rr)))
            endpoints[f'{arm}/{metric}'] = equivalence(differences)
    statuses = [v['status'] for v in endpoints.values()]
    return {'statistical_status': 'EQUIVALENT' if all(s == 'EQUIVALENT' for s in statuses)
            else 'NON_EQUIVALENT' if 'NON_EQUIVALENT' in statuses else 'INCONCLUSIVE',
            'launch_admitted': False, 'requires': 'separate mechanical and provenance gates',
            'endpoints': endpoints}


def compare_frozen_scores(candidate, reference, input_manifest):
    """Use the frozen article count and identities, never the observed row count."""
    if input_manifest.get('schema') not in ('bisr.frozen_inputs.v2', 'bisr.frozen_inputs.v3'):
        raise ValueError('A frozen v2/v3 input manifest is required')
    ids = input_manifest['validation_ids']
    count = input_manifest['validation_documents']
    if input_manifest['schema'] == 'bisr.frozen_inputs.v3' and count != 256:
        raise ValueError('The fresh v3 design requires all 256 validation articles')
    if count < 1 or len(ids) != count or len(set(ids)) != count:
        raise ValueError('Invalid frozen document set')
    for scores in (candidate, reference):
        for arms in scores.values():
            for rows in arms.values():
                if [row['sample_id'] for row in rows] != ids:
                    raise ValueError('Scores do not match frozen validation identities')
    return compare_scores(candidate, reference, documents=count)
