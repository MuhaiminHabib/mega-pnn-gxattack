"""Freeze cohorts/settings before new attack or retraining outcomes."""
import random,ast
from common import ROOT,WORK,read,write,sha,rg
assert not (ROOT/'protocol.json').exists()
for folder in ['bank','models','checkpoints','explanations','detection','resource_runs','completed']:(ROOT/folder).mkdir(exist_ok=True)
rows=read(WORK/'mega_attack_eval/census.json')['rows'];rng=random.Random(20261017771)
pool=[r for r in rows if r['split']=='train' and r['correctly_detected_illicit'] and r['context_transactions']<=10000]
targets=rng.sample(sorted(r['node'] for r in pool),2)
positive=rng.sample(sorted(r['node'] for r in rows if r['split']=='train' and r['label']==1 and r['node'] not in targets),2)
negative=rng.sample(sorted(r['node'] for r in rows if r['split']=='train' and r['label']==0),4)
training=targets+positive+negative;rng.shuffle(training)
cohort=dict(train_nodes=training,attack_training_nodes=targets,validation_nodes=[6649,1149],
    selection_seed=20261017771,attack_target_context_cap=10000,attack_eligible_pool=len(pool),
    training_rows=[rows[n] for n in training],validation_previously_exposed=True)
write(ROOT/'cohort.json',cohort)
parent=read(WORK/'mega_gnn_five_seeds_v1/protocol.json');hashes=dict(parent['source_hashes'])
for path,value in hashes.items():assert sha(path)==value,path
for folder in ['mega_gnn_convergence_check_v1','mega_gnn_five_seeds_v1']:
    for p in (WORK/folder/'results').glob('*'):
        if p.suffix in ['.json','.npz']:hashes[str(p)]=sha(p)
    for name in ['protocol.json','summary.json','independent_verification.json']:
        p=WORK/folder/name;hashes[str(p)]=sha(p)
for node in cohort['validation_nodes']:
    for rel in [f'records/gx_pgexplainer_{node}.json',f'audits/gx_shared_{node}.json',f'domains/{node}.json']:
        p=WORK/'mega_attack_10_v1/results'/rel;hashes[str(p)]=sha(p)
for p in [WORK/'mega_attack_audit_v1/audited_attacks.py',WORK/'mega_attack_audit_v1/audit.py',WORK/'mega_gnn_gradient_init_v1/summary.json']:
    hashes[str(p)]=sha(p)
for p in ROOT.iterdir():
    if p.suffix in ['.py','.txt','.ps1'] or p.name=='cohort.json':hashes[str(p)]=sha(p)
for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
write(ROOT/'protocol.json',dict(cohort=cohort,arms=['baseline','clean','random','adversarial'],
    optimizer='Adam',lr=1e-5,weight_decay=1e-5,gradient_clip=5.,epochs=2,updates_per_arm=16,training_seed=771,
    loss='0.5CE(clean)+0.5CE(augmentation)+JS(predictions); eval-mode fine-tuning, BN buffers fixed',
    training_graphs='Official physical graphs; target TRAIN labels only; no fractional masking in optimizer updates',
    attack_family='Audited GX interior-gradient PG-TX-targeted transfer; unchanged attack settings',
    random_training_control='One ordinary draw matched to attack winner edit composition, validity not resampled',
    gnn_steps=4500,clean_seed_offsets=[0,100000,200000],edited_seed_offsets=[0],
    explanation_lr=.01,explanation_edge_size=.005,explanation_entropy=1.,
    evaluation_edits=['attack','random'],probability_only_edits=['ordinary'],
    expected_explanation_records=40,new_explanation_fits=34,reused_clean_explanations=6,
    detection_split='validation',decision_threshold=.5,test_metrics=False,
    candidate_screen='F1 drop<=1pp vsbaseline; clean/attack illicit2/2; mean clean top5 overlap>=baseline; attack displacement lower than clean AND random fine-tuning',
    no_tuning=True,no_pg_training=True,no_further_experiment=True,
    known_limitations=['Two exposed validation accounts','One training seed','Static original-model PG-targeted attacks, not adaptive post-training attacks','Accepted GNN seed instability and unresolved fractional-mask gradient diagnostic','Primary-only edited explanation fits','Prediction consistency is not direct explanation regularization'],
    resource_policy=rg.POLICY,source_hashes=hashes))
print(cohort,flush=True);print('Frozen',len(hashes),'inputs',sha(ROOT/'protocol.json'),flush=True)
