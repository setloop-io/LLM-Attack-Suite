"""Local rotation-invariant features for latent bundle probes.

Feature contract: norm, off-diagonal cosine mean/std/max/min, eight sorted
cosines, eight sorted absolute cosines, and normalized row position.
Position is an observed row coordinate, not recovered original ordering.
"""


def latent_invariants(latent):
    import torch
    from torch.nn.functional import pad
    if latent.ndim != 3 or latent.shape[1] < 1:
        raise ValueError("latent must have shape [blocks, tokens, features]")
    x = latent.float()
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cosine = (x / norm) @ (x / norm).transpose(-1, -2)
    length = x.shape[1]
    diagonal = torch.eye(length, dtype=torch.bool, device=x.device)[None]
    off = cosine.masked_fill(diagonal, 0)
    mean = off.sum(-1, keepdim=True) / max(1, length - 1)
    std = ((off - mean).square().masked_fill(diagonal, 0).sum(-1, keepdim=True)
           / max(1, length - 1)).sqrt()
    hi = off.masked_fill(diagonal, -1).amax(-1, keepdim=True)
    lo = off.masked_fill(diagonal, 1).amin(-1, keepdim=True)
    count = min(8, length)
    ordered = pad(off.sort(descending=True).values[..., :count], (0, 8-count))
    absolute = pad(off.abs().sort(descending=True).values[..., :count], (0, 8-count))
    position = torch.linspace(-1, 1, length, device=x.device).reshape(1, length, 1)
    return torch.cat((norm, mean, std, hi, lo, ordered, absolute,
                      position.expand(x.shape[0], -1, -1)), -1)
