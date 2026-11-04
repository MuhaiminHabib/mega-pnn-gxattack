"""Independent saved-result audit and concise report. No optimization."""
import json,hashlib,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(ROOT/'plan.json');assert all(sha(p)==h for p,h in plan['source_manifest'].items())
assert read(ROOT.parent/'mega_attack_eval/results_v1/status.json')['state']=='paused'
gnn=read(ROOT/'gnn_convergence/summary.json');pg=read(ROOT/'pg_normalized/summary.json');confirmation=read(ROOT/'confirmation/summary.json');attacks=read(ROOT/'attack_summary.json')
grows=[read(ROOT/'gnn_convergence'/f'{n}_{r}.json') for n in plan['development'] for r in range(3)]
assert len(grows)==24 and sum(r['soft']>.5 for r in grows)==gnn['masked_preserved']
replacement=np.mean([1-len(set(a['top5'])&set(b['top5']))/5 for n in plan['development'] for a,b in itertools.combinations([r for r in grows if r['node']==n],2)])
assert np.isclose(replacement,gnn['replacement'])
for r in grows:
    p=np.load(ROOT/'gnn_convergence'/f"{r['node']}_{r['repeat']}.npz");raw=p['raw'].copy();raw[~p['hard']]=-np.inf;order=np.lexsort((p['ids'],-raw));assert p['ids'][order[:5]].tolist()==r['top5']
prows=[read(ROOT/'pg_normalized'/f'development_{n}.json') for n in plan['development']]
assert sum(e['soft_label_preserved'] for r in prows for e in r['explanations'])==pg['masked_preserved']
assert np.isclose(np.mean([r['replacement'] for r in prows]),pg['replacement'])
pp=read(ROOT/'pg_normalized/protocol.json');assert all(sha(p)==h for p,h in pp['hashes'].items())
if confirmation['state']=='complete':
    cp=read(ROOT/'confirmation/protocol.json');assert all(sha(p)==h for p,h in cp['hashes'].items())
    quality=[read(p) for p in (ROOT/'confirmation/quality').glob('*.json')];assert {q['node'] for q in quality}==set(plan['confirmation'])
    for q in quality:
        assert q['protocol_sha256']==sha(ROOT/'confirmation/protocol.json')
        for r in q['selected_removal'].values():assert r['full_replay_error']<5e-5
if attacks['state']=='complete':
    rows=[read(p) for p in (ROOT/'confirmation/records').glob('*.json')];assert len(rows)==attacks['comparisons']
    for r in rows:
        for arm in ['attack','random']:
            if r[arm]['valid']:assert r[arm]['p_after']>.5 and r[arm]['full_graph_replay_error']<5e-5
lines=['MEGA-PNA PRIORITIZED EXPLAINER REPAIR','Detector checkpoint frozen. No detector training, test queries, defense or old full-run restart.','Stage1: adapter matched physical deletion in40 binary checks. Raw-logit ranking removes sigmoid saturation ties. Frozen PG cross-fit replacement fell60.83% to55.0%; masked label preservation remained20/24.','Development: eight previously exposed validation accounts. Three independent fits/seeds each.']
for label,result in [('GNNExplainer1500steps',gnn),('Normalized PGExplainer-TX60epochs',pg)]:lines.append(f"{label}: masked labels {result['masked_preserved']}/{result['n']}; cross-seed top5 replacement {100*result['replacement']:.2f}%; mean mask {result['mean_mask']:.5f}.")
lines+=['Confirmation:\n'+json.dumps(confirmation,indent=2),'Attack stage:\n'+json.dumps(attacks,indent=2),'These small validation pilots do not establish population reliability. Top5 displacement is not F1 loss or proof of misleading explanations. Failure to pass a gate is reported without automatically expanding or retuning. PG-TX and the fractional PNA mask are adaptations.']
(ROOT/'REPORT.txt').write_text('\n\n'.join(lines),encoding='utf8')
verification=dict(passed=True,source_hashes=True,detector_checkpoint_sha256=next(h for p,h in pp['hashes'].items() if p.endswith('best.pt')),gnn_records=24,pg_development_records=24,full_run_paused=True,independent_metric_checks=True,report_sha256=sha(ROOT/'REPORT.txt'),summary_hashes={str(p):sha(p) for p in [ROOT/'gnn_convergence/summary.json',ROOT/'pg_normalized/summary.json',ROOT/'confirmation/summary.json',ROOT/'attack_summary.json']})
(ROOT/'independent_verification.json').write_text(json.dumps(verification,indent=2));print((ROOT/'REPORT.txt').read_text())
