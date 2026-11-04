"""Conditional small attack check; never resumes the old full dataset run."""
import ast,runpy,msvcrt
from repair_core import *
OUT=TASK/'confirmation'

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    methods=[k for k,v in summary.get('methods',{}).items() if v['quality_pass']] if isinstance(summary.get('methods'),dict) else []
    if not methods:dump(TASK/'attack_summary.json',dict(state='skipped',reason='No explainer passed independent quality confirmation.'));print('ATTACKS SKIPPED',flush=True);return
    lock=open(OLD/'worker.lock','a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert json.loads((OLD/'results_v1/status.json').read_text())['state']=='paused'
    plan=json.loads((TASK/'plan.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    s=RepairStudy();load_pg(s);config=configure_attacks()
    audit_protocol=dict(methods=methods,nodes=plan['attack_pilot'],attack_config=config,hashes={str(p):sha(p) for p in [Path(__file__),TASK/'repair_core.py',TASK/'plan.json',OUT/'protocol.json',CHECKPOINT]},scope='Four preselected validation accounts; existing adapted attacks, same matched random searches; deduction direct, GX directPG and transferGNN. Fixed checkpoint/fits, no attack sweep.')
    path=TASK/'attack_protocol.json'
    if path.exists():assert json.loads(path.read_text())==audit_protocol
    else:dump(path,audit_protocol)
    original=TASK.parent/'mega_explainer_pilot_v3/run_pilot.py';namespace=runpy.run_path(str(original),run_name='attack_import')
    source=original.read_text();tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_cases')
    code=ast.get_source_segment(source,fn).replace("['pgexplainer','gnnexplainer']",repr(methods))
    combos=[(f,m) for f in ['deduction','gx'] for m in methods]
    code=code.replace("[('deduction','pgexplainer'),('deduction','gnnexplainer'),('gx','pgexplainer'),('gx','gnnexplainer')]",repr(combos))
    code=code.replace('            from analyse import analyse\n            analyse()','            pass')
    def progress(**kw):
        try:dump(TASK/'attack_status.json',dict(state='running',**kw))
        except PermissionError:pass
    namespace.update(OUT=OUT,progress=progress);exec(compile(code,str(TASK/'attack_extracted.py'),'exec'),namespace);(TASK/'attack_extracted.py').write_text(code)
    dispatch=dict(protocol);dispatch['cohorts']=dict(validation_pilot=plan['attack_pilot'])
    namespace['run_cases'](s,dispatch)
    records=[json.loads(p.read_text()) for p in (OUT/'records').glob('*.json')];assert len(records)==len(combos)*4
    result=dict(state='complete',accounts=4,comparisons=len(records),groups={})
    for family,method in combos:
        rows=[r for r in records if r['family']==family and r['explainer']==method];g={}
        for arm in ['attack','random']:
            g[arm]=dict(valid=sum(r[arm]['valid'] for r in rows),illicit_preserved=sum(r[arm].get('p_after',0)>.5 for r in rows),replacement=float(np.mean([r[arm]['change']['top5_changed_fraction'] for r in rows])),repeat_replacement=float(np.mean([x['top5_changed_fraction'] for r in rows for x in r[arm].get('repeat_changes',[{},dict(top5_changed_fraction=0),dict(top5_changed_fraction=0)])[1:]])),absolute_confidence_drift_pp=float(np.mean([abs(r[arm].get('confidence_change_pp',0)) for r in rows])),added_top5_accounts=sum(bool(r[arm].get('added_top5')) for r in rows))
        delta=np.array([r['attack']['change']['top5_changed_fraction']-r['random']['change']['top5_changed_fraction'] for r in rows]);rng=np.random.default_rng(20261019);ci=np.quantile(delta[rng.integers(0,4,(10000,4))].mean(1),[.025,.975]).tolist();g['advantage_ci95']=ci;g['supported']=ci[0]>0 and g['attack']['repeat_replacement']>g['random']['repeat_replacement'];result['groups'][family+'_'+method]=g
    assert state_sha(s.model)==MODEL_HASH and all(sha(p)==h for p,h in audit_protocol['hashes'].items())
    dump(TASK/'attack_summary.json',result);dump(TASK/'attack_status.json',dict(state='complete'));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
