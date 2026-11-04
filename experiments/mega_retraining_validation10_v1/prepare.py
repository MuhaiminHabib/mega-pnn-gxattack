"""Create a frozen ten-account evaluation, without training any model."""
import ast,copy,hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;OLD=WORK/'mega_retraining_pilot_v1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')
assert not (ROOT/'protocol.json').exists()
parent=read(OLD/'protocol.json');verification=read(OLD/'independent_verification.json')
assert verification['passed'] and read(OLD/'controller_status.json')['state']=='complete'
for path,h in parent['source_hashes'].items():assert sha(path)==h,path
for path,h in verification['result_hashes'].items():assert sha(path)==h,path
seen={}
def visit(x,path):
    if isinstance(x,dict):
        for k,v in x.items():
            if k in ['node','global_node','account'] and type(v)==int:seen.setdefault(v,str(path))
            if k in ['validation_nodes','validation_accounts'] and isinstance(v,list):
                for n in v:
                    if type(n)==int:seen.setdefault(n,str(path))
            visit(v,path)
    elif isinstance(x,list):
        for v in x:visit(v,path)
for path in sorted(WORK.glob('mega*/**/*.json')):
    if ROOT in path.parents or 'census' in path.name or 'before' in path.parts:continue
    try:visit(read(path),path)
    except (ValueError,UnicodeError):pass
rows=read(WORK/'mega_attack_eval/census.json')['rows']
pool=sorted(r['node'] for r in rows if r['split']=='validation' and r['correctly_detected_illicit'] and r['node'] not in seen)
nodes=random.Random(2026101801).sample(pool,10)
assert nodes==[1655,9872,9952,2516,9670,5911,7547,422,5979,3463]
cohort=copy.deepcopy(parent['cohort']);cohort.update(validation_nodes=nodes,validation_previously_exposed=False,
    selection_seed=2026101801,eligible_pool=pool,validation_rows=[rows[n] for n in nodes],
    new_targeted_explanations_only=True,validation_detection_previously_evaluated=True,no_context_size_filter=True)
write(ROOT/'cohort.json',cohort)
write(ROOT/'exclusion_audit.json',dict(excluded_node_evidence=seen,excluded_evidence_hashes={p:sha(p) for p in sorted(set(seen.values()))},
    rule='Exclude historical scalar node/global_node/account and validation_nodes/validation_accounts; census describes eligibility only',selection_seed=2026101801))
for d in ['bank','checkpoints','explanations','completed','resource_runs']:(ROOT/d).mkdir(exist_ok=True)
common=(OLD/'common.py').read_text().replace('WORK=ROOT.parent','WORK=ROOT.parent\nORIGIN=WORK/\'mega_retraining_pilot_v1\'')
common=common.replace("ROOT/f'models/","ORIGIN/f'models/")
common=common.replace("read(ROOT/'protocol.json').get('preserved_training_protocol_sha256')","read(ORIGIN/'protocol.json').get('preserved_training_protocol_sha256')")
(ROOT/'common.py').write_text(common,encoding='utf8')
for name in ['traced.py','selfcheck.py']:(ROOT/name).write_text((OLD/name).read_text(),encoding='utf8')
evaluate=(OLD/'evaluate.py').read_text()
start=evaluate.index("            if arm=='baseline' and variant=='clean':")
end=evaluate.index("            flips=graph_rows[variant]['flips']",start)
evaluate=evaluate[:start]+evaluate[end:]
evaluate=evaluate.replace("assert all(not par.requires_grad", "assert all(not par.requires_grad")
(ROOT/'evaluate.py').write_text(evaluate,encoding='utf8')
bank=(OLD/'bank.py').read_text();start=bank.index('    torch=pc.torch;is_train=');end=bank.index("    assert len(audit['attack_proposals'])",start)
bank=bank[:start]+'''    torch=pc.torch;is_train=False
    assert node in p['cohort']['validation_nodes']
    assert bool(s.masks['validation'][node]) and not bool(s.masks['test'][node])
    c,support,base,donors=s.context(node)
    def progress(**kw):status(stage='validation_bank',node=node,**{k:v for k,v in kw.items() if k!='stage'})
    audit=attacks.gx(s,support,base,progress)
'''+bank[end:]
bank=bank.replace("source='generated TRAIN only' if is_train else 'read-only historical validation edits'","source='New validation GX and matched controls generated once against ORIGINAL checkpoint, shared unchanged by all four models'")
(ROOT/'bank.py').write_text(bank,encoding='utf8')
supervise=(OLD/'supervise.py').read_text();start=supervise.index("    for node in p['cohort']['attack_training_nodes']");end=supervise.index("    subprocess.run",start)
supervise=supervise[:start]+'''    for node in p['cohort']['validation_nodes']:
        if not (ROOT/f'bank/{node}.json').exists():worker('bank.py',[node])
        for arm in p['arms']:
            if not (ROOT/f'completed/eval_{arm}_{node}.json').exists():worker('evaluate.py',[arm,node])
'''+supervise[end:]
(ROOT/'supervise.py').write_text(supervise,encoding='utf8')
(ROOT/'resume.ps1').write_text((OLD/'resume.ps1').read_text().replace('mega_retraining_pilot_v1','mega_retraining_validation10_v1'),encoding='utf8')
report=(OLD/'report.py').read_text().replace("ROOT/'recovery_baseline_replay/amendment.json'","ORIGIN/'recovery_baseline_replay/amendment.json'").replace('sha(ROOT/path)==value','sha(ORIGIN/path)==value')
report=report.replace("ROOT/f'models/","ORIGIN/f'models/").replace("ROOT/f'detection/","ORIGIN/f'detection/")
report=report.replace('assert checked==40','assert checked==200').replace("a['clean_illicit']==a['attack_illicit']==2","a['clean_illicit']==a['attack_illicit']==10")
report=report.replace("['bank','models','explanations','detection']","['bank','explanations']")
report=report.replace("SMALL OFFLINE MEGA-PNA RETRAINING PILOT","TEN NEW VALIDATION ACCOUNTS: FOUR FROZEN MEGA-PNA CHECKPOINTS")
report=report.replace('Two previously exposed validation targets for explanations.','Ten new explanation targets; original validation detection metrics reused unchanged.')
report=report.replace("}/2,{m[\"attack_illicit\"]}/2", "}/10,{m[\"attack_illicit\"]}/10")
report=report.replace("summary=dict(models=models,gates=gates", "from comparisons import paired_comparisons\n    paired=paired_comparisons(models)\n    summary=dict(paired_comparisons=paired,models=models,gates=gates")
report=report.replace("'Explanation replacement is not detector F1 loss. Seed pairs are dependent; no population confidence intervals.'", "'Explanation replacement is not F1 loss. Paired account-bootstrap intervals in summary.json; seed pairs are not independent samples.'")
report=report.replace("    assert checked==200", "    from comparisons import verify_banks\n    verify_banks(pc,p)\n    assert checked==200")
report=report.replace("model_copies=3,updates_per_copy=16", "model_copies=3,new_training_updates=0,original_updates_per_copy=16")
(ROOT/'report.py').write_text(report,encoding='utf8')
p=copy.deepcopy(parent);p.update(cohort=cohort,expected_explanation_records=200,new_explanation_fits=200,reused_clean_explanations=0,
    evaluation_only=True,new_training_updates=0,parent_protocol_sha256=sha(OLD/'protocol.json'),
    detection_metrics='Read-only reuse of unchanged four checkpoints and existing 1500-validation metrics',
    attack_bank='Generate once per new validation target against original PG-TX; identical edits transferred to four frozen models; NOT adaptive attacks on each model',
    candidate_screen='F1 within1pp original; all10 clean/GX illicit; mean top5 agreement no worse than original; lower primary GX displacement than BOTH controls. Paired account CIs reported separately.',
    known_limitations=['Ten validation accounts; full validation F1 already exposed','One original fine-tuning seed and eight original training labels','Static original-model PG-targeted attacks, not adaptive attacks against each checkpoint','Accepted mask-gradient and seed-instability limitations','Primary-only edited explanation fits','No additional retraining; no pooling with two-node pilot'],
    bootstrap_seed=2026101802,bootstrap_replicates=10000)
hashes=dict(parent['source_hashes'])
for path in [OLD/'protocol.json',OLD/'summary.json',OLD/'independent_verification.json',OLD/'recovery_baseline_replay/amendment.json',WORK/'mega_attack_eval/census.json']:
    hashes[str(path)]=sha(path)
for folder in ['models','detection']:
    for path in (OLD/folder).iterdir():hashes[str(path)]=sha(path)
for path in ROOT.iterdir():
    if path.suffix in ['.py','.txt','.ps1','.json']:hashes[str(path)]=sha(path)
for path in ROOT.glob('*.py'):ast.parse(path.read_text())
p['source_hashes']=hashes;write(ROOT/'protocol.json',p)
print('Frozen new cohort',nodes,'eligible pool',len(pool),'protocol',sha(ROOT/'protocol.json'),flush=True)
