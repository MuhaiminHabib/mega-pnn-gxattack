import json,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
old=root.parent/'mega_explainer_pilot_v3/results/protocol.json'
p=json.loads(old.read_text());census=json.loads((root.parent/'mega_attack_eval/census.json').read_text())
excluded=set(p['cohorts']['excluded_previous_validation_targets'])|set(p['cohorts']['validation_pilot'])
excluded|={json.loads(x.read_text())['node'] for x in (root.parent/'mega_attack_eval/results_v1/records').glob('*.json')}
eligible=sorted([r for r in census['rows'] if r['split']=='validation' and r['correctly_detected_illicit'] and r['node'] not in excluded],key=lambda r:r['context_transactions'])
rng=np.random.default_rng(2026101909);nodes=[]
for block in np.array_split(np.arange(len(eligible)),4):nodes.extend(int(eligible[i]['node']) for i in rng.choice(block,2,replace=False))
plan=dict(authorization='User on2026-10-16: take your time and try fixing issues in importance order.',detector='Frozen MEGA-PNA AMLSim; no detector training, defense work, test queries or full-run resumption.',development=p['cohorts']['validation_pilot'],confirmation=nodes,attack_pilot=nodes[::2],excluded_validation=sorted(excluded),stages=['Audit adapter binary equivalence and raw-score ranking on frozen fits.','Native GNNExplainer1500-step convergence diagnostic with3seeds on8 previously exposed validation accounts.','One normalized PG-TX configuration: training-only input mean/std,60epochs,80original training accounts,3independent fits; unchanged native loss/lr.','Only explainers with >=95% masked-label preservation, <=20% cross-seed replacement and mean_mask<=0.5 advance to8 other validation accounts.','Fresh confirmation also requires exact same-seed/top5 permutation reproducibility and paired top5-deletion advantage95%lowerCI>0 vs50random sets/account.','Only a qualifying explainer advances to a4-account attack pilot with matched random searches and repeat fits/seeds. GX is directPG and transferGNN; deduction directboth.','Report failures without further sweeps; no full-dataset restart.'],gates=dict(masked_label_min=.95,cross_seed_max=.2,mean_mask_max=.5,same_seed_top5_max=0,removal_ci_lower_min=0,attack_ci_lower_min=0,fresh_attack_advantage_min=0),limitations='All are exploratory validation pilots. Earlier project work exposed test results. No universal stability/faithfulness guarantee.',source_manifest={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [root/'diagnose.py',root/'gnn_convergence.py',root/'repair_core.py',root/'fit_pg.py']})
path=root/'plan.json'
if path.exists():assert json.loads(path.read_text())==plan
else:path.write_text(json.dumps(plan,indent=2))
print(json.dumps(dict(confirmation=nodes,attack_pilot=nodes[::2],eligible_remaining=len(eligible))))
