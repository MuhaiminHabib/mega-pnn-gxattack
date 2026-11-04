"""Independent saved-result audit plus predetermined physical replays."""
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
sys.path.insert(0,str(ROOT.parent/'mega_priority_repair'))
import repair_core as core
sys.path.insert(0,str(ROOT))
from analyse import summarize

read=lambda p:json.loads(Path(p).read_text())


def main():
    import msvcrt
    lock=open(ROOT.parent/'mega_attack_eval/worker.lock','a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    protocol=read(OUT/'protocol.json')
    verification=read(OUT/'runner_verification.json')
    assert verification['passed']
    ph=core.sha(OUT/'protocol.json')
    assert ph==verification['protocol_sha256']
    assert all(core.sha(p)==h for p,h in protocol['source_hashes'].items())
    assert all(core.sha(OUT/'records'/n)==h for n,h in verification['records'].items())
    rows=[read(p) for p in (OUT/'records').glob('*.json')]
    targets=protocol['cohorts']['validation_pilot']
    expected=set(itertools.product(targets,['deduction','gx'],['pgexplainer','gnnexplainer']))
    assert len(rows)==40 and {(r['node'],r['family'],r['explainer']) for r in rows}==expected
    quality=[read(OUT/'quality'/f'{n}.json') for n in targets]
    for q in quality:
        assert q['protocol_sha256']==ph
        assert q['p_clean']>.5 and q['split']=='validation'
        assert len(q['random_removal']['indices'])==len(q['random_removal']['probabilities'])==50
        assert abs(q['random_removal']['full_replay_first']-q['random_removal']['probabilities'][0])<5e-5
        for method in ['gnnexplainer','pgexplainer']:
            assert len(q['clean'][method])==3
            assert q['selected_removal'][method]['full_replay_error']<5e-5
            pairs=[core.change(a,b) for a,b in itertools.combinations(q['clean'][method],2)]
            assert pairs==q['repeat_changes'][method]
    cache_count=0
    for path in (OUT/'explanations').glob('*.json'):
        row=read(path)
        assert row['protocol_sha256']==ph
        with np.load(path.with_suffix('.npz')) as arrays:
            ids,raw,scores=arrays['ids'],arrays['raw'],arrays['scores']
            assert len(ids)==len(raw)==len(scores) and len(set(ids.tolist()))==len(ids)
            assert not np.isnan(raw).any() and not np.isposinf(raw).any()
            assert np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1))
            order=np.lexsort((ids,-raw))
            assert row['top5']==ids[order[:5]].tolist() and row['top25']==ids[order[:25]].tolist()
            assert np.allclose(row['top5_scores'],raw[order[:5]],rtol=0,atol=0)
        if '_gnnexplainer_' in path.name:
            assert len(row['trace'])==1500
        cache_count+=1
    for r in rows:
        assert r['protocol_sha256']==ph and r['split']=='validation'
        mode=protocol['modes'][r['family']+'_'+r['explainer']]
        assert r['mode']==mode
        audit_suffix=r['explainer'] if r['family']=='deduction' else 'shared'
        audit=read(OUT/'audits'/f"{r['family']}_{audit_suffix}_{r['node']}.json")
        assert audit['protocol_sha256']==ph
        domain=read(OUT/'domains'/f"{r['node']}.json")
        assert domain['protocol_sha256']==ph
        assert len(audit['attack_proposals'])==len(audit['random_proposals'])
        def composition_indices(v):
            m=domain['original_transactions']
            return (sum(i>=m for i in v['flips']),sum(i<m for i in v['flips']))
        for a,b in zip(audit['attack_proposals'],audit['random_proposals']):
            assert composition_indices(a)==composition_indices(b)
        for proposals in [audit['attack_proposals'],audit['random_proposals']]:
            assert len({tuple(sorted(v['flips'])) for v in proposals})==len(proposals)
        for arm in ['attack','ordinary','random']:
            v=r[arm]
            if not v['valid']:
                assert audit[arm] is None or not audit[arm]['valid']
                continue
            assert v['p_before']>.5 and v['p_after']>.5 and v['full_graph_replay_error']<5e-5
            assert 0<len(v['flips'])==len(v['edits'])==v['actual_edits']<=10
            assert len(v['flips'])==len(set(v['flips']))
            assert v['flips']==audit[arm]['flips']
            assert abs(v['p_after']-audit[arm]['p'])<5e-5
            assert core.change(r['clean'][0],v['after'][0])==v['change']
            assert [core.change(a,b) for a,b in zip(r['clean'],v['after'])]==v['repeat_changes']
            assert np.isclose(v['confidence_change_pp'],100*(v['p_after']-v['p_before']))
            added={x['transaction_key'] for x in v['edits'] if x['operation']=='add'}
            assert v['added_top5']==sorted(added&set(v['after'][0]['top5']))
            if r['family']=='deduction':
                deleted={x['transaction_key'] for x in v['edits'] if x['operation']=='delete'}
                assert not deleted&set(r['clean'][0]['top5'])
                assert audit[arm]['degree_ratio']<protocol['attack_config']['degree_threshold']
            else:
                assert v['actual_edits']>=protocol['attack_config']['gx_min_edits']
        if r['attack']['valid'] and r['ordinary']['valid']:
            def composition(v):
                return (sum(e['operation']=='add' for e in v['edits']),sum(e['operation']=='delete' for e in v['edits']))
            assert composition(r['attack'])==composition(r['ordinary'])
    indexed={(r['node'],r['family'],r['explainer']):r for r in rows}
    for node in targets:
        for arm in ['attack','ordinary','random']:
            assert indexed[node,'gx','pgexplainer'][arm].get('edits')==indexed[node,'gx','gnnexplainer'][arm].get('edits')
    # Reconstruct directly from saved transaction edits, independently of the
    # runner's local flip indices. First account and three distinct attacks.
    s=core.RepairStudy()
    checks=[]
    node=targets[0]
    for family,method in [('deduction','pgexplainer'),('deduction','gnnexplainer'),('gx','pgexplainer')]:
        for arm in ['attack','ordinary','random']:
            v=indexed[node,family,method][arm]
            if not v['valid']:
                continue
            keep=torch.ones(s.graph[core.FORWARD].num_edges,dtype=torch.bool)
            keep[[e['transaction_key'] for e in v['edits'] if e['operation']=='delete']]=False
            added=[e for e in v['edits'] if e['operation']=='add']
            new=torch.tensor([[e['source'],e['destination']] for e in added],dtype=torch.long).reshape(-1,2).T
            donor=torch.tensor([e['donor_csv_row'] for e in added],dtype=torch.long)
            ei=torch.cat([s.graph[core.FORWARD].edge_index[:,keep],new],1)
            attrs=torch.cat([s.graph[core.FORWARD].edge_attr[keep],s.graph[core.FORWARD].edge_attr[donor]])
            graph=core.HeteroData()
            for key in ['x','y','num_nodes','batch_size']:
                graph['node'][key]=s.graph['node'][key]
            for rel,edges in [(core.FORWARD,ei),(core.REVERSE,ei.flip(0))]:
                graph[rel].edge_index=edges;graph[rel].edge_attr=attrs
                graph[rel].simp_edge_batch=ei[0]*10000+ei[1]
            s.model.cpu()
            with torch.no_grad():
                probability=float(core.forward(s.model,graph)[node].softmax(0)[1])
            assert probability>.5 and abs(probability-v['p_after'])<5e-5
            checks.append(dict(node=node,family=family,method=method,arm=arm,error=abs(probability-v['p_after'])))
    assert core.state_sha(s.model)==core.MODEL_HASH
    assert read(OUT/'summary.json')==summarize()
    assert read(ROOT.parent/'mega_attack_eval/results_v1/status.json')['state']=='paused'
    core.dump(OUT/'independent_verification.json',dict(passed=True,records=40,accounts=10,cached_explanations=cache_count,
              full_graph_checks=checks,frozen_hashes=True,summary_independently_recomputed=True,full_run_paused=True,
              report_sha256=core.sha(OUT/'REPORT.txt'),summary_sha256=core.sha(OUT/'summary.json')))
    print('INDEPENDENT SELECTED-PILOT VERIFICATION PASSED',flush=True)


if __name__=='__main__':
    main()
