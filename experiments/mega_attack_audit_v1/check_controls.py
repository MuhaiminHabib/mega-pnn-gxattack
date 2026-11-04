"""Cheap invariant check of the actual search/control orchestration."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mega_priority_repair'))
import repair_core
sys.path.insert(0,str(ROOT))
import audited_attacks as a
import torch

class Toy:
    def predict(self,c,w):return torch.stack([w.sum()*0,w.sum()*.01+1])[None]
    def pg_scores(self,c,w):return self.predict(c,w),(torch.arange(len(w))*.1+w).sigmoid()

saved=a.CONFIG.copy();rank=a.deduction_ranking;ll=a.likelihood
try:
    a.CONFIG.update(budget=3,gx_epochs=3,gx_restarts=3,gx_rounding_draws=25,gx_min_edits=1)
    base=torch.cat([torch.ones(12),torch.zeros(6)])
    support=dict(global_node=0,target=0,tx_rows=torch.arange(18))
    progress=lambda **kw:None
    def explanation(flips):
        ids=a.weights(base,flips).nonzero().flatten().tolist()
        return dict(top5=ids[:5],top25=ids,top5_pairs=[[i,i+1] for i in ids[:5]])
    a.deduction_ranking=lambda s,c,support,base,protected,kind,*rest:([i for i in (range(12) if kind=='D' else range(12,18)) if i not in protected],[])
    a.likelihood=lambda *args:0.
    gx=a.gx(Toy(),support,base,progress)
    deduction=a.deduction(Toy(),support,support,base,explanation([]),'pgexplainer',lambda method,flips,seed:explanation(flips),progress)
    reports=[]
    for name,result in [('gx',gx),('deduction',deduction)]:
        x=result['attack_proposals'];y=result['random_proposals']
        assert len(x)>0 and len(x)==len(y)
        for proposals in [x,y]:assert len({tuple(sorted(v['flips'])) for v in proposals})==len(proposals)
        for left,right in zip(x,y):
            assert len(left['flips'])==len(right['flips'])
            assert sum(i<12 for i in left['flips'])==sum(i<12 for i in right['flips'])
        assert result['ordinary'] is not result['random']
        reports.append(dict(method=name,attack=len(x),random_search=len(y),ordinary=1))
    print(json.dumps(dict(passed=True,counts=reports)))
finally:
    a.CONFIG=saved;a.deduction_ranking=rank;a.likelihood=ll
