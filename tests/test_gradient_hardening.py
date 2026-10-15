import inspect
import pytest

torch=pytest.importorskip('torch')
from attacker.attacks.gradient_hardening import harden_gradient,decode,synthetic

@pytest.mark.parametrize('granularity,dims',[('token',(-1,)),('sequence',(-2,-1)),('batch',(0,1,2))])
def test_clip_is_bounded_and_preserves_direction(granularity,dims):
    g,e,truth,_=synthetic()
    x,info=harden_gradient(g,clip=.1,granularity=granularity)
    assert info['clipped_units']>0 and (torch.linalg.vector_norm(x,dim=dims)<=.100001).all()
    assert decode(g,e,0)==decode(x,e,0)
    assert decode(x,e,0)['backward']==truth.tolist()

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_aborts_entire_release(value):
    g,*_=synthetic();g[0,0,0]=value
    x,info=harden_gradient(g,clip=1.)
    assert x is None and info['status']=='ABORT'


def test_zero_gradient_and_large_finite_norm_are_safe():
    x,info=harden_gradient(torch.zeros(2,3,4),clip=1.)
    assert torch.equal(x,torch.zeros_like(x)) and info['clipped_units']==0
    x,info=harden_gradient(torch.full((2,3,4),1e38),clip=1.)
    assert torch.isfinite(x).all() and (x.norm(dim=-1)<=1.00001).all()


def test_seeded_noise_quantization_and_access_boundary():
    g,*_=synthetic()
    a,_=harden_gradient(g,clip=1.,noise_sigma=.1,bits=4,seed=7)
    b,_=harden_gradient(g,clip=1.,noise_sigma=.1,bits=4,seed=7)
    assert torch.equal(a,b)
    assert 'truth' not in inspect.signature(decode).parameters
    with pytest.raises(ValueError):harden_gradient(g,clip=0)
    with pytest.raises(ValueError):harden_gradient(g,noise_sigma=float('nan'))
    with pytest.raises(ValueError):harden_gradient(g,bits=True)


def test_capture_replay_opens_scoring_tokens_only_after_prediction_write(tmp_path,monkeypatch):
    import json
    from attacker.attacks.gradient_hardening import build_parser,run
    from attacker.evidence import verify_evidence
    g,e,truth,_=synthetic()
    observations=tmp_path/'observations.pt'; embeddings=tmp_path/'embeddings.pt'; canonical=tmp_path/'truth.pt'
    torch.save({'g':g},observations);torch.save(e,embeddings);torch.save(truth,canonical)
    original_load=torch.load
    def guarded(path,*args,**kwargs):
        if path==canonical: assert list(tmp_path.glob('out-*.predictions.json'))
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(torch,'load',guarded)
    args=build_parser().parse_args(['--observations-pt',str(observations),'--embedding-pt',str(embeddings),
       '--canonical-pt',str(canonical),'--seeds','17','--output',str(tmp_path/'out.json')])
    args.mode='training';assert run(args)==0
    result=json.loads((tmp_path/'out.json').read_text())
    assert result['status']=='COMPLETE' and result['evidence_tier']=='capture_replay'
    raw=next(r for r in result['results'] if r['condition']=='raw')
    clip=next(r for r in result['results'] if r['condition']=='clip-0.1')
    assert raw['metrics']==clip['metrics']
    assert raw['compute']['vocabulary_dot_products']>0
    assert raw['compute']['wall_seconds']>=0
    assert raw['compute']['head_forward_calls']==0
    assert verify_evidence(str(tmp_path/'out.json.evidence.json'))[0]
    with pytest.raises(ValueError,match='fresh output'):run(args)
