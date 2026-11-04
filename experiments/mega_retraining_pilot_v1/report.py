"""Verify saved masks, split use, models, controls and detection; report every arm."""
import itertools,hashlib
import numpy as np
from common import *
def average(values):return float(np.mean(values)) if values else None
def overlap(a,b,k):return len(set(a[f'top{k}'])&set(b[f'top{k}']))/k
def state_hash(pc,state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(pc.tensor_sha(v).encode())
    return h.hexdigest()
def main():
    p=validate();pc,_=science();torch=pc.torch;torch.set_num_threads(2)
    amendment=read(ROOT/'recovery_baseline_replay/amendment.json')
    assert amendment['scientific_settings_unchanged']
    for path,value in amendment['preserved_result_hashes'].items():assert sha(ROOT/path)==value,path
    baseline=state_for(pc,'baseline');assert state_hash(pc,baseline)==pc.MODEL_HASH
    source_rows=read(WORK/'mega_attack_eval/census.json')['rows']
    assert all(source_rows[n]['split']=='train' for n in p['cohort']['train_nodes'])
    assert all(source_rows[n]['split']=='validation' for n in p['cohort']['validation_nodes'])
    assert not set(p['cohort']['train_nodes'])&set(p['cohort']['validation_nodes'])
    models={};checked=0;eligible_count=0;full_checks=0
    for arm in p['arms']:
        state=state_for(pc,arm);mh=state_hash(pc,state)
        if arm!='baseline':
            training=read(ROOT/f'models/{arm}.json')
            assert training['state_sha256']==mh and training['baseline_unchanged'] and training['bn_buffers_unchanged']
            assert len(training['logs'])==training['updates']==16
            assert [r['node'] for r in training['logs']]==p['cohort']['train_nodes']*2
            assert all(r['label']==source_rows[r['node']]['label'] for r in training['logs'])
        detection=read(ROOT/f'detection/{arm}.json');assert detection['state_sha256']==mh
        with np.load(ROOT/f'detection/{arm}.npz') as z:
            y=z['labels'];v=z['probabilities'];pred=v>.5
            assert len(y)==1500 and np.isfinite(v).all()
            assert all(source_rows[int(n)]['split']=='validation' for n in z['nodes'])
            assert np.array_equal(y,[source_rows[int(n)]['label'] for n in z['nodes']])
            tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum())
            assert abs(detection['metrics']['illicit_f1']-2*tp/max(2*tp+fp+fn,1))<1e-12
        accounts=[]
        for node in p['cohort']['validation_nodes']:
            graphs=read(ROOT/f'explanations/{arm}_{node}_graphs.json')
            assert graphs['state_sha256']==mh
            rows={}
            for variant in ['clean','attack','random']:
                rows[variant]=[]
                for repeat in (range(3) if variant=='clean' else range(1)):
                    row=read(ROOT/f'explanations/{arm}_{node}_{variant}_{repeat}.json');checked+=1
                    assert row['protocol_sha256']==sha(ROOT/'protocol.json') and row['state_sha256']==mh
                    if row['eligible']:
                        eligible_count+=1
                        assert sha(row['raw_file'])==row['raw_sha256'] and row['seed']==20261017+node+repeat*100000
                        with np.load(row['raw_file']) as z:
                            assert np.isfinite(z['raw']).all() and np.isfinite(z['scores']).all()
                            raw=z['raw'].copy();raw[~z['hard']]=-np.inf;order=np.lexsort((z['ids'],-raw))
                            for k in [5,25]:assert z['ids'][order[:k]].tolist()==row[f'top{k}']
                    rows[variant].append(row)
                g=graphs['graphs'][variant]
                if g['valid_bank']:
                    assert abs(g['full_graph_probability']-g['p'])<5e-5 and g['binary_replay_error']<5e-5;full_checks+=1
            clean=[r for r in rows['clean'] if r['eligible']]
            result=dict(node=node,graphs=graphs['graphs'],isolated_probability=graphs['isolated_probability'],
                clean_eligible=len(clean),soft_illicit=sum(r['p_soft']>.5 for r in clean),
                top5_only_illicit=sum(r['p_keep_only_top5']>.5 for r in clean),
                deletion_drop_pp=average([r['drop_pp'] for r in clean]))
            for k in [5,25]:result[f'top{k}_overlap']=average([overlap(a,b,k) for a,b in itertools.combinations(clean,2)])
            result['changes']={}
            for variant in ['attack','random']:
                before,after=rows['clean'][0],rows[variant][0]
                result['changes'][variant]=None if not (before['eligible'] and after['eligible']) else {
                    f'top{k}_replacement':1-overlap(before,after,k) for k in [5,25]}
                if after['eligible']:
                    bank=read(ROOT/f'bank/{node}.json');m=len(torch.load(ROOT/f'bank/{node}_contexts.pt',map_location='cpu',weights_only=False)['clean']['tx_rows'])
                    context=torch.load(ROOT/f'bank/{node}_contexts.pt',map_location='cpu',weights_only=False)
                    added=set(context['support']['tx_rows'][i].item() for i in bank[variant]['flips'] if i>=m)
                    result['changes'][variant]['added_top5']=len(added&set(after['top5']))
            accounts.append(result)
        model=dict(detection=detection['metrics'],accounts=accounts,
            mean_top5_overlap=average([r['top5_overlap'] for r in accounts if r['top5_overlap'] is not None]),
            mean_top25_overlap=average([r['top25_overlap'] for r in accounts if r['top25_overlap'] is not None]),
            clean_illicit=sum(r['graphs']['clean'].get('illicit',False) for r in accounts),
            attack_illicit=sum(r['graphs']['attack'].get('illicit',False) for r in accounts),
            mean_attack_replacement=average([r['changes']['attack']['top5_replacement'] for r in accounts if r['changes']['attack'] is not None]),
            mean_search_replacement=average([r['changes']['random']['top5_replacement'] for r in accounts if r['changes']['random'] is not None]))
        models[arm]=model
    assert checked==40
    a=models['adversarial'];b=models['baseline']
    comparable=all(models[x]['mean_attack_replacement'] is not None for x in ['clean','random','adversarial'])
    gates=dict(validation_f1_within1pp=a['detection']['illicit_f1']>=b['detection']['illicit_f1']-.01,
        clean_and_attack_illicit_retained=a['clean_illicit']==a['attack_illicit']==2,
        clean_top5_agreement_not_worse=a['mean_top5_overlap'] is not None and b['mean_top5_overlap'] is not None and a['mean_top5_overlap']>=b['mean_top5_overlap']-1e-12,
        less_attack_displacement_than_both_controls=bool(comparable and a['mean_attack_replacement']<models['clean']['mean_attack_replacement'] and a['mean_attack_replacement']<models['random']['mean_attack_replacement']))
    summary=dict(models=models,gates=gates,exploratory_candidate=all(gates.values()),
        interpretation='Static-bank diagnostic only; no adaptive robustness or generalization claim',
        no_test_metrics=True,no_automatic_extension=True)
    write(ROOT/'summary.json',summary)
    validate()
    evidence={str(path):sha(path) for folder in ['bank','models','explanations','detection'] for path in (ROOT/folder).iterdir() if path.suffix in ['.json','.npz','.pt']}
    write(ROOT/'independent_verification.json',dict(passed=True,records_verified=checked,eligible_raw_masks_verified=eligible_count,
        model_copies=3,updates_per_copy=16,training_validation_disjoint=True,frozen_inputs_verified=len(p['source_hashes']),
        original_detector_unchanged=True,physical_full_graph_checks=full_checks,no_test_metrics=True,
        old_ten_node_state='stopped',old_full_run_state='paused',baseline_replay_amendment_verified=True,
        protocol_sha256=sha(ROOT/'protocol.json'),result_hashes=evidence))
    def pct(x):return 'N/A' if x is None else f'{100*x:.2f}%'
    lines=['SMALL OFFLINE MEGA-PNA RETRAINING PILOT','AMLSim;8 training labels,2 training attack targets,16 updates,one training seed.',
        'All1500 validation accounts for detection. Two previously exposed validation targets for explanations.',
        'Three clean GNN seeds per model/account,4500steps; primary seed only for edited explanations.',
        'GX bank targets original frozen PG-TX and transfers to GNN. No fresh adaptive attacks on changed detectors.',
        'Prediction-consistency loss, not direct explanation-consistency training. Existing gradient limitations remain.',
        '', 'Model | validation illicit F1 | clean top5/top25 agreement | GX/search top5 replacement | clean/GX illicit targets']
    for arm,m in models.items():lines.append(f'{arm} | {pct(m["detection"]["illicit_f1"])} | {pct(m["mean_top5_overlap"])}/{pct(m["mean_top25_overlap"])} | {pct(m["mean_attack_replacement"])}/{pct(m["mean_search_replacement"])} | {m["clean_illicit"]}/2,{m["attack_illicit"]}/2')
    lines+=['',f'Exploratory candidate screen: {all(gates.values())}',str(gates),
        'Per-account retention, confidence drift, added-top5 counts, physical fidelity and edge isolation are in summary.json.',
        'Explanation replacement is not detector F1 loss. Seed pairs are dependent; no population confidence intervals.',
        'Original checkpoint unchanged. Every arm reported; no parameter/checkpoint selection from validation outcomes.',
        'No automatic next experiment or historical run resumption.']
    (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
