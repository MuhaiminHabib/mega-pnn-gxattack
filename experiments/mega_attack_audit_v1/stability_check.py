"""One prespecified consensus check; no fitting, selection or new test targets."""
import json,hashlib,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'mega_selected_attack_pilot_v1/results'
OUT=ROOT/'stability'
OUT.mkdir(exist_ok=True)
NODES=[588,7944,5024,9844]
def rank_score(raw):
    # Tied raw values have tied average percentile ranks, including -inf hard zeros.
    _,inverse,counts=np.unique(raw,return_inverse=True,return_counts=True)
    scores=(np.cumsum(counts)-.5*counts)/len(raw)
    result=scores[inverse];result[~np.isfinite(raw)]=0
    return result
def top(ids,score,k):return ids[np.lexsort((ids,-score))[:k]].tolist()
def displacement(a,b):return 1-len(set(a)&set(b))/len(a)
rows=[]
for node,method in itertools.product(NODES,['gnnexplainer','pgexplainer']):
    paths=[];raws=[];ids=None
    suffix=hashlib.sha256(json.dumps([]).encode()).hexdigest()[:20]
    for r in range(3):
        path=OLD/'explanations'/f'{node}_{method}_{r}_{suffix}.npz'
        with np.load(path) as data:
            if ids is None:ids=data['ids'].copy()
            assert np.array_equal(ids,data['ids']);raws.append(data['raw'].copy())
        paths.append(str(path))
    ranks=np.stack([rank_score(x) for x in raws])
    score=np.median(ranks,axis=0)
    row=dict(node=node,method=method,sources=paths,consensus_top5=top(ids,score,5),consensus_top25=top(ids,score,25))
    for k in [5,25]:
        singles=[top(ids,x,k) for x in ranks]
        # The median of two values is their mean. Each pair shares one fit with
        # another pair: this is sensitivity analysis, not independent replication.
        subsets=[top(ids,np.median(ranks[list(pair)],axis=0),k) for pair in itertools.combinations(range(3),2)]
        row[f'top{k}']=dict(single_pair_replacement=[displacement(a,b) for a,b in itertools.combinations(singles,2)],
             leave_one_out_pair_replacement=[displacement(a,b) for a,b in itertools.combinations(subsets,2)],
             full_to_leave_one_out=[displacement(top(ids,score,k),b) for b in subsets],leave_one_out_lists=subsets)
    rows.append(row)
summary={}
for method in ['gnnexplainer','pgexplainer']:
    selected=[r for r in rows if r['method']==method]
    summary[method]={f'top{k}':{key:float(np.mean([r[f'top{k}'][key] for r in selected])) for key in ['single_pair_replacement','leave_one_out_pair_replacement','full_to_leave_one_out']} for k in [5,25]}
result=dict(method='Median within-graph percentile rank of three frozen fits/seeds; canonical ID breaks exact aggregate ties.',nodes=NODES,rows=rows,summary=summary,
    limitation='Leave-one-out groups overlap and are not independent fit groups. A deterministic consensus is not proof of cross-fit stability. Uses previously examined validation accounts; no retraining or attack changes.')
(OUT/'saved_score_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(summary,indent=2))
