"""Observation-separated reconstruction kernels and measured compute.

SIPIT is independently implemented from Nikolaou et al., ICLR 2026,
https://arxiv.org/abs/2510.15511 (Algorithm 1 and Appendix E).
The bounded, best-residual fallback is an explicit adaptation for stale/noisy
observations, where the original exact-match stopping rule may never fire.
No attack function receives evaluation tokens.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import time

import torch
import torch.nn.functional as F

from .attacks.seam_inversion import _head_embed_forward


@dataclass
class Work:
    head_forward_calls: int = 0
    head_forward_sequences: int = 0
    head_forward_token_positions: int = 0
    head_backward_calls: int = 0
    head_backward_token_positions: int = 0
    vocabulary_dot_products: int = 0
    wall_seconds: float = 0.0
    cuda_seconds: float | None = None
    peak_allocated_bytes: int | None = None
    incremental_peak_allocated_bytes: int | None = None


class MeteredHead:
    def __init__(self, split):
        self.split = split
        self.work = Work()

    def __call__(self, *, ids=None, embeddings=None):
        x = ids if ids is not None else embeddings
        self.work.head_forward_calls += 1
        self.work.head_forward_sequences += x.shape[0]
        self.work.head_forward_token_positions += x.shape[0] * x.shape[1]
        return (self.split.head_forward(ids) if ids is not None
                else _head_embed_forward(self.split, embeddings))

    def backward(self, batch, length):
        self.work.head_backward_calls += 1
        self.work.head_backward_token_positions += batch * length

    @contextmanager
    def measure(self):
        self.work = Work()
        cuda = self.split.device.type == 'cuda'
        if cuda:
            torch.cuda.synchronize(self.split.device)
            before = torch.cuda.memory_allocated(self.split.device)
            torch.cuda.reset_peak_memory_stats(self.split.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        try:
            yield self.work
        finally:
            if cuda:
                end_event.record()
                torch.cuda.synchronize(self.split.device)
                self.work.cuda_seconds = start_event.elapsed_time(end_event) / 1000
                self.work.peak_allocated_bytes = torch.cuda.max_memory_allocated(self.split.device)
                self.work.incremental_peak_allocated_bytes = self.work.peak_allocated_bytes - before
            self.work.wall_seconds = time.perf_counter() - start


def forward_scores(h, embedding, decoder, head):
    head.work.vocabulary_dot_products += h.shape[0] * h.shape[1] * len(embedding)
    return F.normalize(h @ decoder, dim=-1) @ F.normalize(embedding, dim=-1).T


def backward_scores(g, embedding, head, first_token):
    """Shifted negative-gradient direction heuristic; NOT full published TAG.

    No label gradient identifies input position zero. Use only the public
    calibration majority there, never an activation-derived first token.
    """
    head.work.vocabulary_dot_products += g.shape[0] * (g.shape[1] - 1) * len(embedding)
    scores = torch.zeros((*g.shape[:2], len(embedding)), device=g.device)
    scores[:, 0, first_token] = 1
    scores[:, 1:] = F.normalize(-g[:, :-1], dim=-1) @ F.normalize(embedding, dim=-1).T
    return scores


def candidate_ids(fwd, bwd, count, variant):
    """Exactly K distinct proposals per row; joint preserves forward top K/2.

    Fill overlap with additional forward candidates, so matched forward/joint
    beam searches execute the same number of prefix evaluations.
    """
    if variant not in ('forward', 'joint', 'legacy_joint'):
        raise ValueError('unknown candidate variant')
    if count < 2 or count % 2 or count > fwd.shape[-1]:
        raise ValueError('candidate count must be even, >=2, and <= vocabulary')
    if variant == 'forward':
        return fwd.topk(count, dim=-1).indices
    if bwd is None or bwd.shape != fwd.shape:
        raise ValueError('joint proposals require aligned backward scores')
    if variant == 'legacy_joint':
        out = bwd.topk(count, dim=-1).indices
        out[:, 0] = fwd[:, 0].topk(count, dim=-1).indices
        return out
    front = fwd.topk(count, dim=-1).indices
    back = bwd.topk(count // 2, dim=-1).indices
    combined = torch.cat((front[..., :count // 2], back, front), dim=-1)
    chosen = []
    for row in combined.reshape(-1, combined.shape[-1]).tolist():
        chosen.append(list(dict.fromkeys(row))[:count])
    out = torch.tensor(chosen, device=fwd.device).reshape(*fwd.shape[:2], count)
    out[:, 0] = front[:, 0]
    return out


@torch.no_grad()
def beam_reconstruct(head, h, candidates, beam_width=2):
    """Causal prefix search ranked by activation squared residual.

    Finite beam pruning is approximate. Causality makes prefix residuals
    final but does not make a finite-width beam globally optimal.
    """
    n, length, count = candidates.shape
    if h.shape[:2] != (n, length) or beam_width < 1:
        raise ValueError('invalid beam shape/budget')
    prefixes = torch.empty(n, 1, 0, dtype=torch.long, device=h.device)
    scores = torch.zeros(n, 1, device=h.device)
    for pos in range(length):
        width = prefixes.shape[1]
        expanded = prefixes[:, :, None, :].expand(n, width, count, pos)
        tokens = candidates[:, None, pos, :, None].expand(n, width, count, 1)
        extensions = torch.cat((expanded, tokens), dim=-1).reshape(n, width * count, pos + 1)
        hidden = head(ids=extensions.flatten(0, 1))[:, -1].reshape(n, width * count, -1)
        residual = (hidden - h[:, None, pos]).square().sum(-1)
        total = residual + scores.repeat_interleave(count, dim=1)
        scores, chosen = total.topk(min(beam_width, width * count), largest=False, dim=1)
        prefixes = extensions.gather(1, chosen[..., None].expand(-1, -1, pos + 1))
    return prefixes[:, 0], {'forward_residual_sum': scores[:, 0].tolist()}


def sipit_reconstruct(head, h, embedding, *, steps, seed, step_size=1.0):
    """Batched, budgeted SIPIT; random init, SGD, exclusion and Euclidean NN.

    Paper operations match the author implementation pinned in the protocol:
    discrete verification each iteration, gradient clip1, linear lr decay,
    snaps every50 iterations (150 at GPT-2 position0). All stopping and
    fallback decisions use observed-hidden residuals, never evaluation truth.
    Return the best *verified* discrete token when a budget ends under mismatch.
    """
    if steps < 1 or steps > len(embedding) or step_size <= 0:
        raise ValueError('invalid SIPIT iteration/learning-rate budget')
    n, length, dim = h.shape
    vocab = len(embedding)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    prefixes = torch.empty(n, 0, dtype=torch.long, device=h.device)
    norm_e = embedding.square().sum(-1)
    iterations, matched_positions, residuals = [], [], []
    for pos in range(length):
        token = torch.randint(vocab, (n,), generator=generator).to(h.device)
        continuous = embedding[token].detach().clone()
        visited = torch.zeros(n, vocab, dtype=torch.bool, device=h.device)
        best_loss = torch.full((n,), float('inf'), device=h.device)
        best_token = token.clone()
        done = torch.zeros(n, dtype=torch.bool, device=h.device)
        counts = torch.zeros(n, dtype=torch.long, device=h.device)
        reset = 50 * (1 + (vocab // 25000) * int(pos == 0))
        fixed = embedding[prefixes].detach()
        for iteration in range(steps):
            variable = continuous.detach().requires_grad_(True)
            pred_h = head(embeddings=torch.cat((fixed, variable[:, None]), dim=1))[:, -1]
            with torch.no_grad():
                discrete = torch.cat((prefixes, token[:, None]), dim=1)
                verify_h = head(ids=discrete)[:, -1]
                loss = (verify_h - h[:, pos]).square().mean(-1)
                improved = (~done) & (loss < best_loss)
                best_loss = torch.where(improved, loss, best_loss)
                best_token = torch.where(improved, token, best_token)
                counts += (~done).long()
                exact = torch.isclose(verify_h, h[:, pos], rtol=1e-5, atol=1e-5).all(-1)
                done |= exact
            if bool(done.all()):
                break
            objective = (pred_h - h[:, pos]).square().mean(-1)
            gradient, = torch.autograd.grad(objective.sum(), variable)
            head.backward(n, pos + 1)
            if not torch.isfinite(gradient).all() or not torch.isfinite(loss).all():
                raise RuntimeError('non-finite SIPIT optimization')
            with torch.no_grad():
                gradient /= gradient.norm(dim=-1, keepdim=True).clamp_min(1)
                lr = step_size * (1 - .99 * min(iteration / reset, 1))
                continuous = variable - lr * gradient
                visited.scatter_(1, token[:, None], True)
                distance = norm_e[None] + continuous.square().sum(-1, keepdim=True) - 2 * continuous @ embedding.T
                distance.masked_fill_(visited, float('inf'))
                head.work.vocabulary_dot_products += n * vocab
                token = distance.argmin(-1)
                if (iteration + 1) % reset == 0:
                    continuous = embedding[token].clone()
        prefixes = torch.cat((prefixes, best_token[:, None]), dim=1)
        iterations.append(counts.tolist())
        matched_positions.append(done.tolist())
        residuals.append(best_loss.tolist())
    return prefixes, {'iterations_by_position': iterations, 'exact_match_by_position': matched_positions,
                      'best_discrete_mse_by_position': residuals,
                      'budget_per_token': steps, 'fallback': 'best verified discrete residual'}


def fit_pca_surrogate(hidden_rows, k):
    """Surrogate encoder/decoder for the defense-aware arm.

    hidden_rows  [N, H] public-calibration head-forward activations
    k            the defense's released coordinate count

    Returns {'mean': [H], 'basis': [H, k],
             'explained_variance_at_k': float}. encode(x) = (x - mean) @
    basis; decode(wire) = wire[..., :k] @ basis.T + mean."""
    if hidden_rows.shape[0] < 2:
        raise ValueError('need at least 2 calibration rows')
    if not 1 <= k <= hidden_rows.shape[-1]:
        raise ValueError('surrogate k must be in [1, seam width]')
    h = hidden_rows.detach().to(dtype=torch.float64)
    mean = h.mean(0)
    _, s, vh = torch.linalg.svd(h - mean, full_matrices=True)
    basis = vh[:k].T.contiguous()
    total = s.square().sum().clamp_min(1e-12)
    ev = float((s[:k].square() / total).sum())
    return {'mean': mean.to(torch.float32), 'basis': basis.to(torch.float32),
            'explained_variance_at_k': ev}


def surrogate_encode(surrogate, hidden):
    return (hidden - surrogate['mean'].to(hidden)) @ surrogate['basis'].to(hidden)


def surrogate_decode(surrogate, wire):
    basis = surrogate['basis'].to(wire)
    return wire[..., :basis.shape[1]] @ basis.T + surrogate['mean'].to(wire)


def surrogate_self_consistency(surrogate, hidden_rows):
    """Reconstruction error of decode(encode(x)) on rows from the
    surrogate's own fitting distribution. Pairing this (small) value with
    wire-recovery accuracy isolates cross-basis/cross-distribution mismatch
    from within-distribution fit: explained_variance_at_k and this residual
    both describe the surrogate on calibration data only, never the defense's
    frame."""
    hidden = hidden_rows.detach()
    reconstructed = surrogate_decode(surrogate, surrogate_encode(surrogate, hidden))
    residual = (reconstructed - hidden).square().sum()
    total = (hidden - hidden.mean(0)).square().sum().clamp_min(1e-12)
    return {'relative_reconstruction_mse': float(residual / total)}
