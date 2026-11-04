"""Check prediction relevance of the fixed consensus; no model fitting."""
import json,msvcrt,itertools
from audit import core,torch,ROOT
read=lambda p:json.loads(p.read_text())
lock=open(ROOT.parent/'mega_attack_eval/worker.lock','a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
input=read(ROOT/'stability/saved_score_check.json');s=core.RepairStudy();out=[]
for row in input['rows']:
    node=row['node'];c,support,base,_=s.context(node);mapping={int(v):i for i,v in enumerate(c['tx_rows'].cpu().tolist())}
    results=dict(node=node,method=row['method'])
    for k in [5,25]:
        ix=[mapping[t] for t in row[f'consensus_top{k}']]
        w=torch.ones(len(c['tx_rows']),device=core.DEVICE);w[ix]=0
        with torch.no_grad():local=float(s.predict(c,w)[c['target']].softmax(0)[1])
        full=s.full_probability(c,torch.ones_like(w),ix) if k==5 or node==588 else None
        if full is not None:assert abs(full-local)<5e-5
        results[f'top{k}']=dict(probability=local,drop_pp=100*(float(s.probs[node])-local),full_graph_probability=full)
    old=read(ROOT.parent/'mega_selected_attack_pilot_v1/results/quality'/f'{node}.json')
    results['old_single_top5_drop_pp']=old['selected_removal'][row['method']]['drop_pp']
    results['random_top5_drop_pp']=100*(float(s.probs[node])-sum(old['random_removal']['probabilities'])/50)
    out.append(results);core.dump(ROOT/'stability/removal_checks.json',dict(completed=len(out),expected=8,rows=out));print(results,flush=True)
assert core.state_sha(s.model)==core.MODEL_HASH
core.dump(ROOT/'stability/verification.json',dict(passed=True,detector_unchanged=True,full_graph_replays=10,rows=8))
