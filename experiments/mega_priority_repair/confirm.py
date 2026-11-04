"""Quality confirmation only for an explainer passing development gates."""
import ast,runpy,msvcrt
from repair_core import *
OUT=TASK/'confirmation'

def main():
    OUT.mkdir(exist_ok=True)
    lock=open(OLD/'worker.lock','a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert json.loads((OLD/'results_v1/status.json').read_text())['state']=='paused'
    plan=json.loads((TASK/'plan.json').read_text());methods=[]
    for method,folder in [('gnnexplainer','gnn_convergence'),('pgexplainer','pg_normalized')]:
        result=json.loads((TASK/folder/'summary.json').read_text())
        if result['masked_preserved']/result['n']>=.95 and result['replacement']<=.2 and result['mean_mask']<=.5:methods.append(method)
    if not methods:
        dump(OUT/'summary.json',dict(state='not_supported',methods=[],reason='Neither explainer passed the predeclared development gate. No confirmation or attack runs authorized by this protocol.'));print('NO QUALIFYING EXPLAINER',flush=True);return
    s=RepairStudy();load_pg(s)
    protocol=dict(cohorts=dict(validation_pilot=plan['confirmation']),methods=methods,hashes={str(p):sha(p) for p in [Path(__file__),TASK/'repair_core.py',TASK/'plan.json',CHECKPOINT,*list((TASK/'pg_normalized').glob('pg_*.pt'))]},mode='Independent validation quality only; no attacks in this stage')
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:dump(OUT/'protocol.json',protocol)
    for folder in ['records','quality','explanations','domains','audits']:(OUT/folder).mkdir(exist_ok=True)
    original=TASK.parent/'mega_explainer_pilot_v3/run_pilot.py';namespace=runpy.run_path(str(original),run_name='quality_import')
    tree=ast.parse(original.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_cases')
    # Remove the attack loop entirely; retained code makes clean explanations,
    # repeat/permutation checks and physical deletion tests only.
    outer=fn.body[1]
    assert isinstance(outer,ast.For)
    outer.body=[n for n in outer.body if not (isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts]==['family','method'])]
    code=ast.unparse(fn).replace("['pgexplainer', 'gnnexplainer']",repr(methods))
    def progress(**kw):
        # Status is diagnostic only; a brief Windows reader lock must not abort science.
        try:dump(OUT/'status.json',dict(state='running',**kw))
        except PermissionError:pass
    namespace.update(OUT=OUT,progress=progress)
    exec(compile(code,str(OUT/'quality_extracted.py'),'exec'),namespace);(OUT/'quality_extracted.py').write_text(code)
    namespace['run_cases'](s,protocol)
    quality=[json.loads(p.read_text()) for p in (OUT/'quality').glob('*.json')];assert len(quality)==8
    result=dict(state='complete',methods={},accounts=8)
    for method in methods:
        masked=sum(e['soft_label_preserved'] for q in quality for e in q['clean'][method]);replacement=float(np.mean([v['top5_changed_fraction'] for q in quality for v in q['repeat_changes'][method]]));mean_mask=float(np.mean([e['mean_mask'] for q in quality for e in q['clean'][method]]))
        delta=np.array([q['selected_removal'][method]['drop_pp']-100*(q['p_clean']-np.mean(q['random_removal']['probabilities'])) for q in quality]);rng=np.random.default_rng(20261019);ci=np.quantile(delta[rng.integers(0,8,(10000,8))].mean(1),[.025,.975]).tolist()
        checks=[q['same_seed_checks'][method] for q in quality if method in q['same_seed_checks']];repro=all(c['repeat_change']['top5_changed_fraction']==0 and c['permutation_change']['top5_changed_fraction']==0 for c in checks)
        result['methods'][method]=dict(masked_preserved=masked,n=24,replacement=replacement,mean_mask=mean_mask,selected_drop_pp=float(np.mean([q['selected_removal'][method]['drop_pp'] for q in quality])),random_drop_pp=float(np.mean([100*(q['p_clean']-np.mean(q['random_removal']['probabilities'])) for q in quality])),removal_advantage_ci95=ci,reproducible=repro,quality_pass=masked/24>=.95 and replacement<=.2 and mean_mask<=.5 and repro and ci[0]>0)
    assert state_sha(s.model)==MODEL_HASH and all(sha(p)==h for p,h in protocol['hashes'].items())
    dump(OUT/'summary.json',result);dump(OUT/'status.json',dict(state='complete'));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
