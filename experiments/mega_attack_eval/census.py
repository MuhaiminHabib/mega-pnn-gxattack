"""Read-only input census for the authorized MEGA-PNA attack evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REPAIR=HERE.parent/'mega_explainer_audit/repair_v2'
sys.path.insert(0,str(REPAIR))
from transaction_adapter_v2 import ROOT,prepare,FORWARD,sha,torch

def dump(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)

def main():
    prior=json.loads((HERE.parent/'aml_attack_pilot/full_dataset_attacks_v1/status.json').read_text())
    assert prior['state']=='complete' and prior['completed']==1740,'Wait for the earlier experiment'
    torch.set_num_threads(2)
    graph,masks,ids,txids,meta=prepare()
    p=np.load(ROOT/'results/amlsim_edge_audit_v1/predictions.npz')['full_graph']
    y=graph['node'].y.numpy();ei=graph[FORWARD].edge_index.numpy();n=len(y)
    incident=[[] for _ in range(n)]
    for u,v in np.unique(ei.T,axis=0):
        incident[u].append(int(v));incident[v].append(int(u))
    rows=[]
    for target in range(n):
        split=next(s for s in ['train','validation','test'] if masks[s][target])
        r=dict(node=target,account_id=int(ids[target]),split=split,label=int(y[target]),
               probability=float(p[target]),correctly_detected_illicit=bool(y[target]==1 and p[target]>.5))
        if r['correctly_detected_illicit']:
            first=np.zeros(n,dtype=bool);first[incident[target]]=True;first[target]=True
            chosen=first[ei[0]]|first[ei[1]]
            r.update(context_transactions=int(chosen.sum()),context_nodes=int(len(np.unique(ei[:,chosen]))),one_hop_nodes=int(first.sum()))
        rows.append(r)
    counts={s:dict(accounts=sum(s=='all' or r['split']==s for r in rows),
                   illicit=sum(r['label']==1 and (s=='all' or r['split']==s) for r in rows),
                   correct_illicit=sum(r['correctly_detected_illicit'] and (s=='all' or r['split']==s) for r in rows))
            for s in ['all','train','validation','test']}
    sizes=np.array([r['context_transactions'] for r in rows if r['correctly_detected_illicit']])
    out=dict(dataset=meta,counts=counts,rows=rows,context_transactions_quantiles=dict(zip(['min','median','p90','p95','max'],np.quantile(sizes,[0,.5,.9,.95,1]).tolist())),
             above_context_sizes={str(k):int((sizes>k).sum()) for k in [60000,100000,200000,500000]},
             source_sha256=sha(__file__),prediction_sha256=sha(ROOT/'results/amlsim_edge_audit_v1/predictions.npz'))
    dump(HERE/'census.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['rows','dataset']},indent=2),flush=True)

if __name__=='__main__':main()
