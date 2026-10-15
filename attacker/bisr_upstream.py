"""Pinned upstream attack adapter for the explicitly single-seam C-A route.

Source methods execute unchanged except annotations and progress-only scoring.
The caller supplies frozen granted segments; simulator restoration is therefore
a no-op. No canonical tokens or scorer batch are accepted by this adapter.
"""
import ast
from contextlib import nullcontext
from types import SimpleNamespace
from experiments.bisr_upstream_preflight import extract


def adapter(split, sources):
    import torch
    from .attacks.bisr_baseline import _embed_forward
    env = {'torch':torch, 'CrossEntropyLoss':torch.nn.CrossEntropyLoss,
           'ParamRestored':lambda **kw:nullcontext()}
    extract(sources['model'], 'calc_shifted_loss_logits', env)
    extract(sources['dlg'], '_extract_args_from_inters', env)
    loss = extract(sources['dlg'], '_gma_loss', env)
    cls = next(x for x in ast.parse(sources['dlg']).body
               if isinstance(x,ast.ClassDef) and x.name=='TAGAttacker')
    gm = extract(ast.unparse(cls), 'attack', env)
    class Top(torch.nn.Module):
        def forward(self, x, **kw): return split.rest_forward(x)
    target=SimpleNamespace(top_mocker=Top(),model_type='decoder',pk=None,
                           target_model_config=split.public.model.config)
    target._gma_loss=lambda *a,**kw:loss(target,*a,**kw)
    class Holder:
        def __init__(self,*a,**kw): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def change_config(self): pass
    class Progress:
        def __init__(self,*a,**kw): pass
        def update(self,*a): pass
        def set_description(self,*a): pass
    class Bottom:
        def __init__(self):
            self.config=split.head.model.config
            self.device=split.device
            self.fl_config=SimpleNamespace()
        def __call__(self,inputs_embeds): return _embed_forward(split,inputs_embeds)
    env.update({'get_embedding_layer':lambda llm:split.wte,
        'ChatGLMForConditionalGenerationSplit':type('UnusedGLM',(),{}),
        'FLConfigHolder':Holder,'tqdm':Progress,
        'evaluate_attacker_rouge':lambda *a:({'rouge-l':{'f':0}},None,None)})
    rec=extract(sources['sma'],'_rec_text',env)
    sm=extract(sources['sma'],'attack',env)
    sm_target=SimpleNamespace(pk=None)
    sm_target._rec_text=lambda *a:rec(sm_target,*a)
    simulator=SimpleNamespace(parameter_keeper=None)
    def run_gm(h,g,scores,iters=32):
        return gm(target,SimpleNamespace(model_name='gpt2'),
            SimpleNamespace(softmax=True,init_temp=1.2,epochs=iters,lr=.09,beta=.85),
            SimpleNamespace(device=split.device),None,simulator,None,None,
            SimpleNamespace(fx=h,grad=g),{},init=scores)
    def run_sm(h,initial,iters=800):
        return sm(sm_target,None,
            SimpleNamespace(at='b2tr',init=True,epochs=iters,lr=.01,wd=.01,cosine_loss=True),
            Bottom(),None,simulator,None,SimpleNamespace(fx=h),None,{},init=initial)
    return run_gm, run_sm


def reference_gru(sources, arch, state, device):
    """Execute upstream GRU class with its identity GPT-2 SIP base forward."""
    import torch
    from torch import nn
    class SIPBase(nn.Module):
        def __init__(self,config,*a,**kw): super().__init__();self.config=config
        def forward(self,x): return x.float() if x.dtype==torch.float16 else x
    env={'torch':torch,'nn':nn,'SIPInverter':SIPBase,'LSTMDRAttackerConfig':object}
    cls=next(n for n in ast.parse(sources['sip']).body
             if isinstance(n,ast.ClassDef) and n.name=='GRUDRInverter')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),
                 '<pinned-GRUDRInverter>','exec'),env)
    config=SimpleNamespace(n_embed=arch['seam_dim'],hidden_size=arch['hidden_size'],
        vocab_size=arch['vocab'],bidirectional=False,dropout=arch['dropout'])
    inv=env['GRUDRInverter'](config).to(device)
    inv.load_state_dict({k.replace('output.','mlp.'):v for k,v in state.items()},strict=True)
    return inv.eval().requires_grad_(False)
