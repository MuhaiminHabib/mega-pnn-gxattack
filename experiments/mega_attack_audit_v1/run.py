"""One frozen, resumable confirmation after the implementation audit."""
import gc,json,os,sys,time,traceback,msvcrt
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mega_priority_repair'))
import repair_core as core
sys.path.insert(0,str(ROOT))
import audited_attacks,cases
from analyse import analyse
import torch
OUT=ROOT/'results'
read=lambda p:json.loads(Path(p).read_text())

class SelectedStudy(core.RepairStudy):
    def describe_raw(self,c,raw,scores,seed,trace=None):
        row=super().describe_raw(c,raw,scores,seed,trace)
        self.last_raw=raw.detach().cpu().numpy().copy();return row

def freeze(s):
    previous=read(ROOT.parent/'mega_selected_attack_pilot_v1/results/protocol.json')
    hashes=previous['source_hashes'].copy()
    assert all(core.sha(p)==h for p,h in hashes.items())
    files=list(ROOT.glob('*.py'))+list((ROOT/'reference').glob('*.py'))+[ROOT/n for n in ['README.txt','DECISIONS.txt','resume.ps1','reference_audit.json','mega_diagnostics.json','boundary_diagnostics.json']]
    hashes.update({str(p):core.sha(p) for p in files})
    nodes=[1584,7441,8050,9990]
    assert not set(nodes)&set(previous['cohorts']['validation_pilot'])
    for node in nodes:
        assert bool(s.masks['validation'][node]) and not bool(s.masks['test'][node])
        assert int(s.graph['node'].y[node])==1 and float(s.probs[node])>.5
    protocol=dict(cohorts=dict(validation_pilot=nodes),expected=16,source_hashes=hashes,attack_config=audited_attacks.CONFIG,
      detector_state_sha256=core.MODEL_HASH,selected_explainers=previous['selection_path'],modes=previous['modes'],
      changes=['Exact capped-simplex projection before clipping.','PG deduction addition inputs use addition-graph embeddings.',
      'GX evaluates unchanged objective gradient at p clamped to[1e-4,1-1e-4]; actual projection, rounding and discrete objective unchanged.',
      'Equal distinct evaluated random-search candidates with per-candidate composition matching. Ordinary random is one separate prespecified draw; invalid candidates retained.'],
      labels=['GXAttack interior-gradient transaction adaptation; unchanged cosine/prediction/entropy objective.',
      'Attack-XGNN deduction complementary-input PG-TX adaptation; signed loss with illicit target.',
      'Three frozen60-epoch PG fits and1500-step GNN seeds unchanged; consensus not adopted as a replacement.'],
      stopping='One confirmation. No tuning, detector/PG training, test queries, defenses or full-run resumption.',
      caveats=['Four exploratory accounts previously used for explainer quality assessment.',
      'Author functions checked on Syn1 GCN with a fixed scoring probe, not end-to-end paper success reproduction.',
      'Equal discrete candidates are not equal compute: attacks also use gradients.',
      'No-valid arms contribute zero prediction-preserving displacement; validity denominators also reported.',
      'Bootstrap unit is account. Explanation displacement is not F1 loss.'])
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==protocol,'Frozen confirmation changed'
    else:core.dump(path,protocol)
    return protocol

def preflight(s,p):
    path=OUT/'preflight.json';ph=core.sha(OUT/'protocol.json')
    if path.exists():assert read(path)['passed'] and read(path)['protocol_sha256']==ph;return
    checks=[]
    for node in p['cohorts']['validation_pilot']:
        c,support,base,_=s.context(node)
        with torch.no_grad():
            clean=float(s.predict(c)[c['target']].softmax(0)[1]);padded=float(s.predict(support,base)[support['target']].softmax(0)[1])
        assert abs(clean-float(s.probs[node]))<5e-5 and abs(clean-padded)<5e-5
        _,exp=s.explain(c,'pgexplainer',0)
        old=read(ROOT.parent/'mega_pg_refinement_v1/results/quality'/f'confirmation_{node}.json')['variants']['baseline']['explanations'][0]
        assert old['top5']==exp['top5']
        checks.append(dict(node=node,p_clean=clean,padded_error=abs(clean-padded),pg_selected_reproduces=True))
    c,support,base,_=s.context(p['cohorts']['validation_pilot'][0]);physical=[]
    for flips in [[0],[int(base.sum())]]:
        with torch.no_grad():v=float(s.predict(support,audited_attacks.weights(base,flips))[support['target']].softmax(0)[1])
        f=s.full_probability(support,base,flips);assert abs(f-v)<5e-5
        physical.append(dict(flips=flips,error=abs(f-v)))
    assert all(not p.requires_grad for p in s.model.parameters())
    assert all(not p.requires_grad for fit in s.pg_fits for p in fit.parameters())
    core.dump(path,dict(passed=True,protocol_sha256=ph,baseline=checks,physical_checks=physical))
    print('AUDITED CONFIRMATION PREFLIGHT PASSED',flush=True)

def main():
    for name in ['','records','explanations','audits','domains','quality']:(OUT/name).mkdir(parents=True,exist_ok=True)
    lock=open(ROOT.parent/'mega_attack_eval/worker.lock','a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert read(ROOT.parent/'mega_attack_eval/results_v1/status.json')['state']=='paused'
    assert read(ROOT/'reference_audit.json')['passed']
    if (OUT/'runner_verification.json').exists():raise RuntimeError('Already complete; verify only')
    cases.progress(stage='load');s=SelectedStudy();core.load_pg(s);p=freeze(s);preflight(s,p);cases.run_cases(s,p)
    assert core.state_sha(s.model)==core.MODEL_HASH
    assert all(core.sha(path)==h for path,h in p['source_hashes'].items())
    assert read(ROOT.parent/'mega_attack_eval/results_v1/status.json')['state']=='paused'
    summary=analyse();assert summary['completed']==16 and summary['accounts_evaluated']==4
    core.dump(OUT/'runner_verification.json',dict(passed=True,records={f.name:core.sha(f) for f in (OUT/'records').glob('*.json')},protocol_sha256=core.sha(OUT/'protocol.json'),sources_and_checkpoints_unchanged=True,full_run_paused=True))
    core.dump(OUT/'status.json',dict(state='complete',pid=os.getpid(),updated=time.time(),completed=16,expected=16,independent_verification_pending=True))
    print('AUDITED CONFIRMATION COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True,parents=True);core.dump(OUT/'failure.json',dict(pid=os.getpid(),updated=time.time(),traceback=traceback.format_exc()));raise
