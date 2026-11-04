"""Independently audit all saved records of the preceding GraphSAGE run."""
import json,hashlib,math
from pathlib import Path
from statistics import mean
BASE=Path(__file__).resolve().parent.parent/'aml_attack_pilot/full_dataset_attacks_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    load=lambda n:json.loads((BASE/n).read_text())
    status=load('status.json');v=load('verification.json');summary=load('summary.json');p=load('protocol.json');c=load('census.json')
    assert status['state']=='complete' and status['completed']==1740
    assert v['completed']==v['expected']==1740 and v['classifier_unchanged'] and v['source_hashes_unchanged']
    for f,h in p['sources'].items():assert sha(f)==h,f
    manifest=json.loads((BASE.parent/'portfolio_v1/input_manifest.json').read_text())
    for x in manifest['inputs']:assert sha(x['path'])==x['sha256'],x['path']
    eligible={r['target']:r for r in c['rows'] if r['eligible']}
    assert len(c['rows'])==10000 and len(eligible)==870
    rows=[];hashes={};seen=set()
    for f in (BASE/'cases').glob('*.json'):
        r=json.loads(f.read_text());key=(r['family'],r['target']);assert key not in seen;seen.add(key)
        meta=eligible[r['target']]
        assert r['split']==meta['split'] and r['seed']==101 and r['config']==p['configurations'][r['family']]
        assert f.name==f'{r["family"]}_{r["target"]}_s101.json'
        for arm in ['attack','random']:
            a=r[arm]
            if a['valid']:
                assert a['prediction_preserved'] and a['p_before']>.5 and a['p_after']>.5
                assert 0<a['actual_edits']<=10 and a['actual_edits']==len(a['edits'])
                assert abs(100*(a['p_after']-a['p_before'])-a['probability_change_pp'])<1e-5
                assert {z['fit'] for z in a['top_rankings']}==set(p['fits'])
        hashes[f.name]=sha(f);rows.append(r)
    assert seen=={(family,target) for family in ['deduction','gx'] for target in eligible}
    extra={}
    for split in ['all','train','validation','test']:
        for family in ['deduction','gx']:
            rr=[r for r in rows if r['family']==family and (split=='all' or r['split']==split)]
            s=summary['groups'][f'{split}_{family}'];assert len(rr)==s['evaluated']==s['expected']
            extra[f'{split}_{family}']={}
            for arm in ['attack','random']:
                vals=[r[arm] for r in rr];valid=[r for r in vals if r['valid']]
                def change(a,fit):return next(z['changed_fraction'] for z in a['comparisons'] if z['k']==25 and z['fit']==fit) if a['valid'] else 0.
                metrics=dict(valid=len(valid),label_preserved=sum(a.get('prediction_preserved',False) for a in vals),
                    target_change_all_attempts=mean(change(a,2026) for a in vals),
                    fresh_change_all_attempts=mean(mean(change(a,f) for f in p['fits'][1:]) for a in vals),
                    target_added_top5_nodes=sum(bool(a['top_rankings'][0]['added_top5']) for a in valid),
                    fresh_majority_added_top5_nodes=sum(sum(bool(z['added_top5']) for z in a['top_rankings'][1:])>=3 for a in valid),
                    confidence_drift_over5pp=sum(abs(a['probability_change_pp'])>5 for a in valid),
                    empty_explanation_nodes=sum(not a['top_rankings'][0]['top5'] for a in valid))
                for k,x in metrics.items():assert math.isclose(x,s[arm][k],rel_tol=1e-10,abs_tol=1e-10),(split,family,arm,k)
                extra[f'{split}_{family}'][arm]=dict(mean_signed_confidence_drift_pp=mean(a['probability_change_pp'] for a in valid),
                    mean_absolute_confidence_drift_pp=mean(abs(a['probability_change_pp']) for a in valid),max_absolute_confidence_drift_pp=max(abs(a['probability_change_pp']) for a in valid))
    result=dict(passed=True,case_count=len(rows),case_sha256=hashes,source_and_input_hashes_verified=True,
                all_saved_summary_metrics_recomputed=True,per_split_coverage={s:sum(r['split']==s for r in eligible.values()) for s in ['train','validation','test']},
                confidence_statistics=extra,summary_sha256=sha(BASE/'summary.json'))
    (BASE/'independent_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=True,coverage=result['per_split_coverage'],confidence=extra),indent=2))
if __name__=='__main__':main()
