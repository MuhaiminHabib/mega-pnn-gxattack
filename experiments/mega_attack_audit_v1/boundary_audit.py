"""Numerical diagnosis, not an attack-setting sweep. One previous pilot node."""
import json,msvcrt
from pathlib import Path
from audit import core,torch,F,ROOT

lock=open(ROOT.parent/'mega_attack_eval/worker.lock','a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
s=core.RepairStudy();core.load_pg(s);c,support,base,_=s.context(588)
rows=[]
for precision in ['float32','float64']:
    if precision=='float64':
        s.model.double();s.pg.double()
        support={k:(v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v) for k,v in support.items()};base=base.double();s.bound=None
    with torch.no_grad():_,v=s.pg_scores(support,base);initial=v*base
    def objective(p):
        w=base+(1-2*base)*p;l,v=s.pg_scores(support,w);q=.99*p+.005
        return F.log_softmax(l[support['target']],0)[1]+.1*(1-F.cosine_similarity(initial[None],(v*w)[None])[0])+.0005*(q*q.log()+(1-q)*(1-q).log()).mean()
    for at in [0.,.0001,.001]:
        p=torch.full_like(base,at,requires_grad=True);value=objective(p);g=torch.autograd.grad(value,p)[0]
        index=344
        finite=[]
        for eps in [.001,.0001,.00001]:
            plus=p.detach().clone();minus=p.detach().clone();plus[index]+=eps
            if at>0 and eps<=at:minus[index]-=eps
            divisor=eps*(2 if at>0 and eps<=at else 1)
            with torch.no_grad():fd=float((objective(plus)-objective(minus))/divisor)
            finite.append(dict(eps=eps,value=fd))
        rows.append(dict(precision=precision,at=at,index=index,automatic=float(g[index]),finite_differences=finite))
        print(rows[-1],flush=True)
s.model.float();s.pg.float();assert core.state_sha(s.model)==core.MODEL_HASH
core.dump(ROOT/'boundary_diagnostics.json',dict(node=588,checks=rows,detector_weights_unchanged=True))
