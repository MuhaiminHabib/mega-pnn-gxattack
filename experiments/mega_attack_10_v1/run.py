"""One validation account per process, identical audited science, bounded cohort."""
import argparse
import gc
import json
import msvcrt
import os
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
PARENT = ROOT.parent/'mega_attack_audit_v1'
sys.path.insert(0, str(ROOT.parent/'mega_priority_repair'))
import repair_core as core
sys.path.insert(0, str(PARENT))
import audited_attacks
sys.path.insert(0, str(ROOT))
import cases
from analyse import analyse
import torch
read = lambda p: json.loads(Path(p).read_text())


class SelectedStudy(core.RepairStudy):
    def describe_raw(self,c,raw,scores,seed,trace=None):
        row=super().describe_raw(c,raw,scores,seed,trace)
        self.last_raw=raw.detach().cpu().numpy().copy()
        return row


def check_pg_repeat(first, second):
    """Exact top-five IDs; bounded float32 roundoff in their raw scores."""
    assert first['top5']==second['top5'], 'Same-fit PG top-five ranking changed'
    a,b=first['top5_scores'],second['top5_scores']
    assert len(a)==len(b)==5
    errors=[abs(x-y) for x,y in zip(a,b)]
    assert all(error<=1e-5+1e-6*abs(y) for error,y in zip(errors,b)), 'Same-fit PG scores differ beyond float32 tolerance'
    return max(errors)


def preflight(s, protocol, node):
    path=OUT/'preflight'/f'{node}.json'; ph=core.sha(OUT/'protocol.json')
    if path.exists():
        old=read(path); assert old['passed'] and old['protocol_sha256']==ph
        return
    c,support,base,_=s.context(node)
    with torch.no_grad():
        clean=float(s.predict(c)[c['target']].softmax(0)[1])
        padded=float(s.predict(support,base)[support['target']].softmax(0)[1])
    assert abs(clean-float(s.probs[node]))<5e-5 and abs(clean-padded)<5e-5
    assert bool(s.masks['validation'][node]) and not bool(s.masks['test'][node])
    assert int(s.graph['node'].y[node])==1 and clean>.5
    _,exp=s.explain(c,'pgexplainer',0)
    _,again=s.explain(c,'pgexplainer',0)
    score_error=check_pg_repeat(exp,again)
    physical=[]
    if node==protocol['cohorts']['validation_pilot'][0]:
        for flips in [[0],[int(base.sum())]]:
            with torch.no_grad():
                v=float(s.predict(support,audited_attacks.weights(base,flips))[support['target']].softmax(0)[1])
            f=s.full_probability(support,base,flips); assert abs(f-v)<5e-5
            physical.append(dict(flips=flips,error=abs(f-v)))
    assert all(not p.requires_grad for p in s.model.parameters())
    assert all(not p.requires_grad for fit in s.pg_fits for p in fit.parameters())
    core.dump(path,dict(passed=True,protocol_sha256=ph,node=node,p_clean=clean,
        padded_error=abs(clean-padded),pg_same_fit_reproduces=True,pg_max_absolute_score_error=score_error,
        pg_score_tolerance=dict(atol=1e-5,rtol=1e-6),pg_top5_exact=True,physical_checks=physical))
    del c,support,base; s.bound=None; gc.collect(); torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--node',type=int,required=True)
    node=parser.parse_args().node
    for name in ['','records','explanations','audits','domains','quality','preflight']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    lock=open(ROOT.parent/'mega_attack_eval/worker.lock','a+b')
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    p=read(OUT/'protocol.json'); assert node in p['cohorts']['validation_pilot']
    assert all(core.sha(path)==h for path,h in p['source_hashes'].items())
    assert p['attack_config']==audited_attacks.CONFIG
    assert read(ROOT.parent/'mega_attack_eval/results_v1/status.json')['state']=='paused'
    assert read(PARENT/'results/independent_verification.json')['passed']
    assert not (OUT/'runner_verification.json').exists(), 'Already complete'
    expected={f'{family}_{method}_{node}.json' for family in ['deduction','gx'] for method in ['pgexplainer','gnnexplainer']}
    assert not all((OUT/'records'/n).exists() for n in expected), 'Node already complete'
    cases.progress(stage='load',node=node)
    s=SelectedStudy();core.load_pg(s);preflight(s,p,node)
    cases.run_cases(s,p,only_node=node)
    assert core.state_sha(s.model)==core.MODEL_HASH
    assert all(core.sha(path)==h for path,h in p['source_hashes'].items())
    assert read(ROOT.parent/'mega_attack_eval/results_v1/status.json')['state']=='paused'
    summary=analyse()
    if summary['completed']==p['expected']:
        assert summary['accounts_evaluated']==10
        core.dump(OUT/'runner_verification.json',dict(passed=True,
            records={f.name:core.sha(f) for f in (OUT/'records').glob('*.json')},
            protocol_sha256=core.sha(OUT/'protocol.json'),sources_and_checkpoints_unchanged=True,full_run_paused=True))
    core.dump(OUT/'status.json',dict(state='complete' if summary['completed']==40 else 'account_complete',
        pid=os.getpid(),updated=time.time(),node=node,completed=summary['completed'],expected=40))
    print('ACCOUNT COMPLETE',node,summary['completed'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        core.dump(OUT/'failure.json',dict(pid=os.getpid(),updated=time.time(),traceback=traceback.format_exc()))
        raise
