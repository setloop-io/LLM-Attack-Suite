"""Truth-free controls for gradient information and access sensitivity."""
import math
import torch


def transform_gradient(g, condition, seed):
    if g.ndim != 3 or min(g.shape) < 1 or not g.is_floating_point() or not torch.isfinite(g).all():
        raise ValueError('finite [documents, positions, hidden] gradients required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    if condition == 'true':
        return g.clone(), {}
    if condition == 'shuffled':
        if len(g) < 2:
            raise ValueError('shuffling requires at least two distinct documents')
        order = torch.randperm(len(g), generator=generator)
        donor = torch.empty_like(order)
        donor[order] = order.roll(1)
        return g[donor.to(g.device)].clone(), {'donor_indices': donor.tolist()}
    if condition.startswith('noise_'):
        scale = float(condition[6:])
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('negative noise')
        rms = g.square().mean(dim=(-2, -1), keepdim=True).sqrt()
        noise = torch.randn(g.shape, generator=generator).to(g.device, g.dtype)
        return g + noise * rms * scale, {'relative_rms_sigma': scale}
    if condition.startswith('bits_'):
        bits = int(condition[5:])
        if not 2 <= bits <= 16:
            raise ValueError('bits must be 2..16')
        limit = 2 ** (bits - 1) - 1
        step = g.abs().amax(dim=(-2, -1), keepdim=True) / limit
        safe = step.clamp_min(torch.finfo(g.dtype).tiny)
        return (g / safe).round().clamp(-limit, limit) * step, {'signed_bits': bits, 'scale': 'per document absmax'}
    raise ValueError('unknown gradient condition')


def random_candidates(fwd, count, seed):
    """Forward half plus uniformly sampled distinct non-forward tokens.

    Position zero uses forward K, matching the joint proposal policy.
    """
    from .reconstruction_baselines import candidate_ids
    if fwd.ndim != 3 or min(fwd.shape) < 1 or not torch.isfinite(fwd).all():
        raise ValueError('finite nonempty rank-three scores required')
    front = candidate_ids(fwd, None, count, 'forward')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    rows = []
    for index, row in enumerate(front.reshape(-1, count).tolist()):
        if index % fwd.shape[1] == 0:
            rows.append(row)
            continue
        keep = row[:count // 2]
        excluded = set(keep)
        extra = []
        while len(extra) < count // 2:
            value = int(torch.randint(fwd.shape[-1], (1,), generator=generator))
            if value not in excluded:
                excluded.add(value)
                extra.append(value)
        rows.append(keep + extra)
    return torch.tensor(rows, device=fwd.device).reshape_as(front)
