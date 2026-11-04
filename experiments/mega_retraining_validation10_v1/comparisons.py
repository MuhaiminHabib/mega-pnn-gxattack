import numpy as np
from common import ROOT,read
def paired_comparisons(models):
    result={}
    for control in ['baseline','clean','random']:
        a={r['node']:r for r in models['adversarial']['accounts']};b={r['node']:r for r in models[control]['accounts']}
        metrics={}
        for metric in ['top5_overlap','top25_overlap','attack_top5_replacement','attack_top25_replacement','random_top5_replacement']:
            def value(r):
                if metric in ['top5_overlap','top25_overlap']:return r[metric]
                variant,k=metric.split('_',1);entry=r['changes'][variant];return None if entry is None else entry[k]
            ids=[n for n in sorted(a) if value(a[n]) is not None and value(b[n]) is not None]
            delta=np.array([100*(value(a[n])-value(b[n])) for n in ids])
            if not len(delta):metrics[metric]=dict(n=0,mean_difference_pp=None,ci95_pp=None);continue
            rng=np.random.default_rng(2026101802);boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
            metrics[metric]=dict(n=len(ids),nodes=ids,mean_difference_pp=float(delta.mean()),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),direction='adversarial minus control; overlap higher better, replacement lower better')
        result[control]=metrics
    return result
def verify_banks(pc,p):
    for node in p['cohort']['validation_nodes']:
        b=read(ROOT/f'bank/{node}.json');a=b['audit'];c=pc.torch.load(ROOT/f'bank/{node}_contexts.pt',map_location='cpu',weights_only=False);m=len(c['clean']['tx_rows'])
        assert b['source_model_state']==pc.MODEL_HASH and not b['training']
        assert len(a['attack_proposals'])==len(a['random_proposals'])
        for key in ['attack_proposals','random_proposals']:
            assert len({tuple(x['flips']) for x in a[key]})==len(a[key])
        for x,y in zip(a['attack_proposals'],a['random_proposals']):
            assert len(x['flips'])==len(y['flips'])<=10
            assert sum(i<m for i in x['flips'])==sum(i<m for i in y['flips'])
        for arm in p['arms']:
            g=read(ROOT/f'explanations/{arm}_{node}_graphs.json')['graphs']
            for variant in ['attack','random','ordinary']:
                assert g[variant]['valid_bank']==b[variant]['valid']
                if b[variant]['valid']:assert g[variant]['flips']==b[variant]['flips']
