"""Saved-record summaries, with account-paired uncertainty and explicit modes."""
import itertools
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent/'results'
read=lambda p:json.loads(Path(p).read_text())


def interval(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261021)
    samples=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),accounts=len(values))


def mean(values):
    return float(np.mean(list(values)))


def primary(arm):
    return arm['change']['top5_changed_fraction'] if arm['valid'] else 0.


def repeat(arm):
    return mean(x['top5_changed_fraction'] for x in arm['repeat_changes'][1:]) if arm['valid'] else 0.


def summarize():
    rows=[read(p) for p in sorted((OUT/'records').glob('*.json'))]
    quality=[read(p) for p in sorted((OUT/'quality').glob('*.json'))]
    result=dict(completed=len(rows),expected=40,accounts_evaluated=len(quality),expected_accounts=10,quality={},attacks={})
    for method in ['pgexplainer','gnnexplainer']:
        if not quality:
            continue
        descriptions=[e for q in quality for e in q['clean'][method]]
        displacement=[mean(x['top5_changed_fraction'] for x in q['repeat_changes'][method]) for q in quality]
        noedit=[q['same_seed_checks'][method] for q in quality if method in q['same_seed_checks']]
        selected=[q['selected_removal'][method]['drop_pp'] for q in quality]
        random=[100*(q['p_clean']-mean(q['random_removal']['probabilities'])) for q in quality]
        result['quality'][method]=dict(accounts=len(quality),explanations=len(descriptions),
             masked_preserved=sum(e['soft_label_preserved'] for e in descriptions),mean_mask=mean(e['mean_mask'] for e in descriptions),
             cross_fit_or_seed_replacement=interval(displacement),same_seed_checks=noedit,
             same_seed_top5_exact=bool(noedit) and all(q['repeat_change']['top5_changed_fraction']==0 for q in noedit),
             permutation_top5_exact=bool(noedit) and all(q['permutation_change']['top5_changed_fraction']==0 for q in noedit),
             selected_drop_pp=mean(selected),random_drop_pp=mean(random),
             removal_advantage_pp=interval(np.array(selected)-np.array(random)),
             selected_flips=sum(q['selected_removal'][method]['p_after']<=.5 for q in quality),
             random_flips=sum(p<=.5 for q in quality for p in q['random_removal']['probabilities']),
             random_trials=sum(len(q['random_removal']['probabilities']) for q in quality))
    for family,method in itertools.product(['deduction','gx'],['pgexplainer','gnnexplainer']):
        group=[r for r in rows if r['family']==family and r['explainer']==method]
        if not group:
            continue
        metrics=dict(accounts=len(group),mode=group[0]['mode'])
        for arm in ['attack','ordinary','random']:
            values=[r[arm] for r in group]
            valid=[v for v in values if v['valid']]
            metrics[arm]=dict(valid=len(valid),no_valid_proposal=len(values)-len(valid),
                illicit_preserved=sum(v['p_after']>.5 for v in valid),
                top5_replacement=mean(primary(v) for v in values),repeat_top5_replacement=mean(repeat(v) for v in values),
                top25_replacement=mean(v['change']['top25_changed_fraction'] if v['valid'] else 0 for v in values),
                repeat_top25_replacement=mean(mean(x['top25_changed_fraction'] for x in v['repeat_changes'][1:]) if v['valid'] else 0 for v in values),
                pair_jaccard_loss=mean(1-v['change']['pair_jaccard'] if v['valid'] else 0 for v in values),
                confidence_drift_pp=mean(v['confidence_change_pp'] for v in valid) if valid else None,
                absolute_confidence_drift_pp=mean(abs(v['confidence_change_pp']) for v in valid) if valid else None,
                added_top5_accounts=sum(bool(v.get('added_top5')) for v in values),
                changed_accounts=sum(primary(v)>0 for v in values))
        metrics['advantage_pp']=interval([100*(primary(r['attack'])-primary(r['random'])) for r in group])
        metrics['repeat_advantage_pp']=interval([100*(repeat(r['attack'])-repeat(r['random'])) for r in group])
        metrics['advantage_vs_ordinary_pp']=interval([100*(primary(r['attack'])-primary(r['ordinary'])) for r in group])
        metrics['repeat_advantage_vs_ordinary_pp']=interval([100*(repeat(r['attack'])-repeat(r['ordinary'])) for r in group])
        metrics['top25_advantage_pp']=interval([100*((r['attack']['change'].get('top25_changed_fraction',0) if r['attack']['valid'] else 0)-(r['random']['change'].get('top25_changed_fraction',0) if r['random']['valid'] else 0)) for r in group])
        metrics['exploratory_positive_signal']=(len(group)==10 and metrics['advantage_pp']['ci95'][0]>0 and metrics['repeat_advantage_pp']['mean']>0)
        audits=[]
        for row in group:
            suffix=method if family=='deduction' else 'shared'
            audit=read(OUT/'audits'/f"{family}_{suffix}_{row['node']}.json")
            audits.append(dict(node=row['node'],attack_evaluated=len(audit['attack_proposals']),random_evaluated=len(audit['random_proposals']),
                               attack_valid=sum(x['valid'] for x in audit['attack_proposals']),random_valid=sum(x['valid'] for x in audit['random_proposals'])))
        metrics['proposal_counts']=audits
        result['attacks'][family+'_'+method]=metrics
    return result


def analyse():
    result=summarize()
    path=OUT/'summary.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
    lines=['TEN-NODE AUDITED MEGA-PNA ATTACK EVALUATION',f"Completed {result['completed']}/40 combinations on {result['accounts_evaluated']}/10 validation accounts.",
           'Frozen MEGA-PNA. Selected normalized60-epoch PG-TX, native1500-step GNNExplainer. No PG/detector training or test-target queries.',
           'Attack-XGNN deduction directly targets each explainer. GXAttack directly targets PG-TX; identical edits transfer to GNNExplainer.',
           'One attack seed/account; up to10 transaction edits. Ordinary random is one prespecified draw; random SEARCH matches distinct evaluated candidates and their compositions. Validity is not resampled. Attacks also use gradients, so total compute is not matched.',
           'GX interior-gradient transaction adaptation: gradient queried at [1e-4,1-1e-4] to correct clamp endpoint behavior; original cosine objective retained. PG deduction addition embeddings corrected; exact projection corrected. See ../mega_attack_audit_v1/DECISIONS.txt.',
           'Selected winners are evaluated using two other frozen PG fits/GNN seeds. No-valid outcomes contribute zero prediction-preserving displacement; also inspect valid denominators.',
           'Source/checkpoint hashes frozen. Full-run remains paused. Earlier explainer stability targets were not met; user explicitly authorized this ten-node evaluation.',
           'Paired bootstrap intervals are across ten accounts, not repeated fits. Interpret as exploratory; no proof of generality, F1 loss or misleading explanations.']
    for method,q in result['quality'].items():
        lines.append(f"QUALITY {method}: masked labels {q['masked_preserved']}/{q['explanations']}; cross-fit/seed top5 replacement {100*q['cross_fit_or_seed_replacement']['mean']:.2f}%; same-seed/permutation top5 exact {q['same_seed_top5_exact']}/{q['permutation_top5_exact']}. Selected/random deletion drop {q['selected_drop_pp']:.3f}/{q['random_drop_pp']:.3f}pp.")
    for name,g in result['attacks'].items():
        a=g['attack'];r=g['random'];ordinary=g['ordinary']
        lines.append(f"{name}: ordinary random top5 primary/repeat {100*ordinary['top5_replacement']:.2f}/{100*ordinary['repeat_top5_replacement']:.2f}%, valid {ordinary['valid']}/{g['accounts']}; attack/ordinary/search top25 {100*a['top25_replacement']:.2f}/{100*ordinary['top25_replacement']:.2f}/{100*r['top25_replacement']:.2f}%. Ordinary-control paired intervals: {g['advantage_vs_ordinary_pp']}; repeats: {g['repeat_advantage_vs_ordinary_pp']}.")
        lines.append(f"{name} ({g['mode']}), n={g['accounts']}: primary attack/random {100*a['top5_replacement']:.2f}/{100*r['top5_replacement']:.2f}%; repeat {100*a['repeat_top5_replacement']:.2f}/{100*r['repeat_top5_replacement']:.2f}%; advantage {g['advantage_pp']}; repeat advantage {g['repeat_advantage_pp']}; illicit retained {a['illicit_preserved']}/{a['valid']} valid attacks ({a['no_valid_proposal']} no-valid) and {r['illicit_preserved']}/{r['valid']} controls; absolute confidence drift {a['absolute_confidence_drift_pp']}/{r['absolute_confidence_drift_pp']}pp; added-top5 accounts {a['added_top5_accounts']}/{r['added_top5_accounts']}; exploratory positive signal {g['exploratory_positive_signal']}.")
    lines.append('Actual search candidate counts, pair overlap and no-edit details are saved in summary.json. No full-dataset resumption or new tuning is authorized by completion of this pilot.')
    (OUT/'REPORT.txt').write_text('\n\n'.join(lines),encoding='utf8')
    return result


if __name__=='__main__':
    print(json.dumps(analyse(),indent=2))
