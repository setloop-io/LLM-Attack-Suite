"""Post-capture clipping/abort audit; separate from the frozen validation study.

Clipping rescales gradient vectors and need not erase their direction. The
backward decoder is the existing shifted direction heuristic, not full TAG.
Optional joint beam uses activations and a supplied public head. Captures must
be raw gradients; replaying already clipped/noisy gradients compounds defenses.
"""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from ..evidence import EvidenceLogger, file_sha256

EXPERIMENT_ID='gradient_hardening'
MODES=('training',)
REQUIRES_LABELS='public embeddings; optional public calibration; evaluation tokens are scoring-only'
DESCRIPTION='Clipping/non-finite-abort, noise and quantization replay against backward direction and optional joint reconstruction'


def harden_gradient(g, *, clip=None, granularity='token', noise_sigma=0., bits=None, seed=0):
    import torch
    if g.ndim!=3 or min(g.shape)<1 or not g.is_floating_point(): raise ValueError('floating [documents,tokens,hidden] required')
    if granularity not in ('token','sequence','batch'): raise ValueError('unknown clipping granularity')
    if clip is not None and (not math.isfinite(clip) or clip<=0): raise ValueError('clip must be finite positive')
    if not math.isfinite(noise_sigma) or noise_sigma<0: raise ValueError('invalid absolute noise sigma')
    if bits is not None and (type(bits) is not int or not 2<=bits<=16): raise ValueError('bits must be2..16')
    if not torch.isfinite(g).all(): return None,{'status':'ABORT','reason':'nonfinite_gradient','abort_scope':'entire supplied release'}
    x=g.detach().clone()
    dimensions={'token':(-1,),'sequence':(-2,-1),'batch':(0,1,2)}[granularity]
    norm=torch.linalg.vector_norm(x.double(),dim=dimensions,keepdim=True)
    clipped=0
    if clip is not None:
        scale=(clip/norm.clamp_min(torch.finfo(torch.float64).tiny)).clamp(max=1.)
        clipped=int((norm>clip).sum());x=(x.double()*scale).to(g.dtype)
    if noise_sigma:
        generator=torch.Generator(device='cpu').manual_seed(seed)
        x=x+torch.randn(x.shape,generator=generator).to(x.device,x.dtype)*noise_sigma
    if bits is not None:
        from ..validation_controls import transform_gradient
        x,_=transform_gradient(x,f'bits_{bits}',seed)
    if not torch.isfinite(x).all(): return None,{'status':'ABORT','reason':'nonfinite_after_transform','abort_scope':'entire supplied release'}
    return x,{'status':'ACCEPT','clip':clip,'granularity':granularity,'clipped_units':clipped,
              'units':norm.numel(),'absolute_noise_sigma':noise_sigma,'signed_bits':bits,
              'differential_privacy_claim':False}


def decode(g,embedding,first_token,*,h=None,head=None,decoder=None,candidates=8,return_work=False):
    """Attacker API: no truth, annotations, private labels or victim tail."""
    import torch
    from ..reconstruction_baselines import Work,backward_scores,forward_scores,candidate_ids,beam_reconstruct
    class Counter:
        def __init__(self): self.work=Work()
    meter=head if head is not None else Counter()
    before=asdict(meter.work)
    if g.is_cuda: torch.cuda.synchronize(g.device)
    started=time.perf_counter()
    with torch.no_grad():
        bwd=backward_scores(g,embedding,meter,first_token)
        predictions={'backward':bwd.argmax(-1).cpu().tolist()}
        if head is not None:
            fwd=forward_scores(h,embedding,decoder,head)
            pool=candidate_ids(fwd,bwd,candidates,'joint')
            pred,_=beam_reconstruct(head,h,pool,2)
            predictions['joint']=pred.cpu().tolist()
    if g.is_cuda: torch.cuda.synchronize(g.device)
    work={key:value-before[key] for key,value in asdict(meter.work).items()
          if isinstance(value,(int,float)) and isinstance(before[key],(int,float))}
    work['wall_seconds']=time.perf_counter()-started
    work['device']=str(g.device)
    work['scope']='decoding only; excludes calibration, sanitization and scoring'
    return (predictions,work) if return_work else predictions


def synthetic(seed=17):
    import torch
    rng=torch.Generator().manual_seed(seed)
    embedding=torch.eye(32)
    truth=torch.randint(1,32,(8,8),generator=rng);truth[:,0]=0
    g=torch.randn(8,8,32,generator=rng)*.01
    g[:,:-1]=-embedding[truth[:,1:]]*5
    null=torch.randn(g.shape,generator=rng)
    return g,embedding,truth,null


def build_parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--toy',action='store_true');ap.add_argument('--quick',action='store_true')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--evidence',action='store_true',help='Evidence is always recorded')
    ap.add_argument('--seed',type=int,default=17);ap.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    ap.add_argument('--observations-pt',type=Path,help='weights-only dict containing raw g and optional h tensors')
    ap.add_argument('--embedding-pt',type=Path,help='public [vocabulary,hidden] tensor, or use head export')
    ap.add_argument('--canonical-pt',type=Path,help='aligned evaluation tokens, opened after predictions')
    ap.add_argument('--head-pt',type=Path);ap.add_argument('--calibration-pt',type=Path)
    ap.add_argument('--first-token',type=int,default=0,help='publicly fixed first-position guess, never fitted to target truth')
    ap.add_argument('--granularity',choices=['token','sequence','batch'],default='token')
    ap.add_argument('--clip',type=float,nargs='+',default=[1.,.1])
    ap.add_argument('--noise-sigma',type=float,default=.01,help='absolute sigma after clip1; no DP accounting implied')
    ap.add_argument('--candidates',type=int,default=8);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    return ap


def run(args):
    import torch
    torch.set_num_threads(1)
    from ..reconstruction_metrics import reconstruction_metrics
    from ..attacks import seam_inversion as si
    from ..reconstruction_baselines import MeteredHead
    if getattr(args,'mode','training')!='training': raise ValueError('training gradients required')
    if any(Path(str(args.output)+suffix).exists() for suffix in ('','.evidence.json','.evidence.jsonl')): raise ValueError('fresh output required')
    if len(args.clip)!=len(set(args.clip)): raise ValueError('distinct clipping thresholds required')
    if len(args.seeds)!=len(set(args.seeds)) or not args.seeds: raise ValueError('distinct seeds required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    artifact={'schema':'llm_attacker.gradient_hardening.v1','status':'RUNNING','evidence_tier':'synthetic' if args.toy else 'capture_replay',
              'results':[],'source_hashes':{},'limits':['Replay transforms raw observations, not a deployed combined defense.',
              'Clipping is not DP; positive per-vector scaling preserves direction.',
              'Backward direction is heuristic; no full published-gradient-inversion reproduction.',
              'Evaluation is conditional on supplied captures; attack repetitions are not independent victim runs.']}
    head=decoder=h=None
    with EvidenceLogger(args.output) as ev:
        try:
            if not args.toy:
                if not args.observations_pt or (not args.embedding_pt and not args.head_pt): raise ValueError('observations and public embeddings/head required')
                obs=torch.load(args.observations_pt,map_location=args.device,weights_only=True)
                if not isinstance(obs,dict) or 'g' not in obs or set(obs)-{'g','h'}: raise ValueError('observation dict must contain only g and optional h')
                raw=obs['g'].float();h=obs.get('h')
                if args.head_pt:
                    if h is None or not args.calibration_pt: raise ValueError('joint head path requires h and public calibration')
                    split,embedding=si._load_head_pt(str(args.head_pt),args.device);split.model.requires_grad_(False);embedding=embedding.detach()
                    head=MeteredHead(split);cal=torch.load(args.calibration_pt,map_location=args.device,weights_only=True)
                    with torch.no_grad(): decoder,_=si._ridge_fit(head(ids=cal).reshape(-1,embedding.shape[-1]),embedding[cal.flatten()],0)
                    first=si.majority_control_token(cal.tolist())
                else:
                    embedding=torch.load(args.embedding_pt,map_location=args.device,weights_only=True).float();first=args.first_token
                if embedding.ndim!=2 or not torch.isfinite(embedding).all() or embedding.shape[1]!=raw.shape[-1] or not 0<=first<len(embedding): raise ValueError('invalid public embedding/first token')
                if h is not None and (h.shape!=raw.shape or not torch.isfinite(h).all()): raise ValueError('invalid forward observation')
                for path in (args.observations_pt,args.embedding_pt,args.head_pt,args.calibration_pt,args.canonical_pt):
                    if path: artifact['source_hashes'][str(path)]=file_sha256(path)
            for seed in args.seeds:
                if args.toy:
                    raw,embedding,truth,null=synthetic(seed);first=0
                    raw,embedding,null=(x.to(args.device) for x in (raw,embedding,null))
                conditions=[('raw',dict()),*[(f'clip-{c}',{'clip':c}) for c in args.clip],
                            ('clip1-noise',{'clip':1.,'noise_sigma':args.noise_sigma}),
                            ('clip1-bits8',{'clip':1.,'bits':8}),('clip1-bits4',{'clip':1.,'bits':4})]
                if args.toy: conditions += [('null',{}),('nonfinite',{})]
                for condition,settings in conditions:
                    source=null if condition=='null' else raw
                    if condition=='nonfinite':source=raw.clone();source[0,0,0]=float('nan')
                    transformed,sanitizer=harden_gradient(source,granularity=args.granularity,seed=seed,**settings)
                    cell={'seed':seed,'condition':condition,'sanitizer':sanitizer}
                    if transformed is not None:
                        pred,work=decode(transformed,embedding,first,h=h,head=head,decoder=decoder,candidates=args.candidates,return_work=True)
                        cell['compute']=work
                        path=args.output.with_name(f'{args.output.stem}-{seed}-{condition}.predictions.json')
                        if path.exists(): raise ValueError('prediction path already exists')
                        path.write_text(json.dumps(pred,allow_nan=False))
                        cell.update(prediction_file=path.name,prediction_sha256=file_sha256(path))
                        if not args.toy and args.canonical_pt: truth=torch.load(args.canonical_pt,map_location='cpu',weights_only=True)
                        if args.toy or args.canonical_pt:
                            if truth.shape!=raw.shape[:2]: raise ValueError('unaligned canonical tokens')
                            cell['metrics']={arm:[reconstruction_metrics(t,p) for t,p in zip(truth.tolist(),values)] for arm,values in pred.items()}
                    artifact['results'].append(cell);ev.event('cell',cell)
            artifact['status']='COMPLETE'
        except BaseException as error:
            artifact.update(status='FAILED',error=f'{type(error).__name__}: {error}');raise
        finally:args.output.write_text(json.dumps(artifact,indent=2,allow_nan=False))
    return 0


def self_test():
    try: import torch
    except ImportError:
        print("[SKIP] gradient-hardening requires torch")
        return 0
    g,_,_,_=synthetic()
    x,meta=harden_gradient(g,clip=1.)
    assert meta['status']=='ACCEPT' and float(x.norm(dim=-1).max())<=1.000001
    assert torch.allclose(torch.nn.functional.normalize(g,dim=-1),torch.nn.functional.normalize(x,dim=-1),atol=1e-6)
    return 0
