"""Fast tensor-only checks for the defense-aware surrogate diagnostics."""
import pytest

torch = pytest.importorskip('torch')
from attacker.reconstruction_baselines import (fit_pca_surrogate,
                                               surrogate_decode,
                                               surrogate_encode,
                                               surrogate_self_consistency)


def _low_rank_rows(n=64, h=32, rank=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    basis, _ = torch.linalg.qr(torch.randn(h, rank, generator=g))
    return torch.randn(n, rank, generator=g) @ basis.T


def test_self_consistency_matches_explained_variance_residual():
    rows = _low_rank_rows()
    surrogate = fit_pca_surrogate(rows, k=4)
    result = surrogate_self_consistency(surrogate, rows)
    assert set(result) == {'relative_reconstruction_mse'}
    assert result['relative_reconstruction_mse'] == pytest.approx(
        1 - surrogate['explained_variance_at_k'], abs=1e-3)
    assert result['relative_reconstruction_mse'] < 1e-6


def test_cross_basis_decode_mismatch_is_large_despite_high_explained_variance():
    rows = _low_rank_rows()
    surrogate = fit_pca_surrogate(rows, k=4)
    assert surrogate['explained_variance_at_k'] > 0.999
    defense_basis, _ = torch.linalg.qr(
        torch.randn(rows.shape[-1], 4, generator=torch.Generator().manual_seed(1)))
    wire = torch.zeros_like(rows)
    wire[:, :4] = rows @ defense_basis
    decoded = surrogate_decode(surrogate, wire)
    relative_error = ((decoded - rows).square().sum()
                      / (rows - rows.mean(0)).square().sum())
    assert relative_error > 0.5
    own_wire = torch.zeros_like(rows)
    own_wire[:, :4] = surrogate_encode(surrogate, rows)
    assert (surrogate_decode(surrogate, own_wire) - rows).abs().max() < 1e-4
