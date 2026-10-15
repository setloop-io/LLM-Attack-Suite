import pytest

def test_head_preserves_embedding_gradient_and_exact_layer_prefix():
    torch = pytest.importorskip('torch')
    from transformers import GPT2Config, GPT2LMHeadModel
    from attacker import splitbench as sb
    from attacker.attacks.seam_inversion import _head_embed_forward
    for depth in (1, 2, 3):
        torch.manual_seed(35)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=31,n_positions=16,n_embd=12,n_layer=4,n_head=3)).eval()
        model.requires_grad_(False)
        split = sb.build_split(model, sb.SplitPlan.parse(str(depth), n_layers=4))
        ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            expected = split.head_forward(ids)
        seen = []
        handles = [block.register_forward_hook(lambda m,a,o,i=i: seen.append(i)) for i,block in enumerate(split.blocks)]
        variable = model.transformer.wte(ids).detach().clone().requires_grad_(True)
        actual = _head_embed_forward(split,variable)
        torch.testing.assert_close(actual,expected)
        actual.square().mean().backward()
        assert seen == list(range(depth))
        assert variable.grad is not None and torch.isfinite(variable.grad).all()
        assert variable.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.parameters())
        for h in handles:h.remove()
