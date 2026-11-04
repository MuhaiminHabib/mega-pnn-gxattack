"""Independent saved-record audit, with independently assembled full-graph spot replays."""
from bridge import *
from statistics import mean
import math

def main():
    out=HERE/'results_v1';census=json.loads((HERE/'census.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    verify=json.loads((out/'verification.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert verify['passed'] and verify['completed']==verify['expected']==5252
    for p,h in protocol['source_hashes'].items():assert sha(p)==h
    for name,h in verify['record_sha256'].items():assert sha(out/'records'/name)==h
    assert sha(out/'summary.json')==verify['summary_sha256']
    eligible={r['node']:r for r in census['rows'] if r['correctly_detected_illicit']}
    rows=[json.loads(p.read_text()) for p in (out/'records').glob('*.json')]
    actual={(r['family'],r['explainer'],r['node']) for r in rows}
    expected={(f,e,n) for n in eligible for f in ['gx','deduction'] for e in ['pgexplainer','gnnexplainer']}
    assert len(rows)==len(actual)==5252 and actual==expected
    protocol_hash=sha(out/'protocol.json')
    for r in rows:
        assert r['protocol_sha256']==protocol_hash and r['split']==eligible[r['node']]['split']
        for arm in ['attack','random']:
            v=r[arm]
            if not v['valid']:continue
            assert v['p_before']>.5 and v['p_after']>.5 and v['full_graph_replay_error']<5e-5
            assert 0<v['actual_edits']<=10 and v['actual_edits']==len(v['edits'])==len(set(v['flips']))
            assert abs(v['confidence_change_pp']-100*(v['p_after']-v['p_before']))<1e-8
            expected_change=change(r['clean'],v['after'])
            assert expected_change==v['change']
            if r['family']=='deduction':assert not v['clean_top5_deleted']
    for split in ['all','train','validation','test']:
        for family in ['gx','deduction']:
            for method in ['pgexplainer','gnnexplainer']:
                rr=[r for r in rows if r['family']==family and r['explainer']==method and (split=='all' or r['split']==split)]
                group=summary['groups'][f'{split}_{family}_{method}'];assert len(rr)==group['completed']==group['expected']
                for arm in ['attack','random']:
                    values=[r[arm] for r in rr]
                    displacement=mean(v['change']['top5_changed_fraction'] if v['valid'] else 0 for v in values)
                    assert math.isclose(displacement,group[arm]['mean_top5_change_all_attempts'],abs_tol=1e-12)
                    assert sum(v['valid'] for v in values)==group[arm]['valid']
    # Same GX edits must be used in the PG attack and GNN transfer result.
    indexed={(r['family'],r['explainer'],r['node']):r for r in rows}
    for n in eligible:
        for arm in ['attack','random']:
            assert indexed['gx','pgexplainer',n][arm].get('edits')==indexed['gx','gnnexplainer',n][arm].get('edits')
    s=Study();checks=[]
    # Predeclared lowest-ID valid case in each split/combination/arm:24 replays.
    for split in ['train','validation','test']:
        for f,e in [('gx','pgexplainer'),('deduction','pgexplainer'),('deduction','gnnexplainer')]:
            rr=sorted((r for r in rows if r['split']==split and r['family']==f and r['explainer']==e),key=lambda r:r['node'])
            for arm in ['attack','random']:
                r=next((r for r in rr if r[arm]['valid']),None)
                if r is None:continue
                v=r[arm];g=s.graph.clone();deleted=[x['transaction_key'] for x in v['edits'] if x['operation']=='delete']
                additions=[x for x in v['edits'] if x['operation']=='add']
                keep=torch.ones(s.graph[FORWARD].num_edges,dtype=torch.bool);keep[deleted]=False
                added=torch.tensor([[x['source'],x['destination']] for x in additions],dtype=torch.long).reshape(-1,2).T
                donor=torch.tensor([x['donor_csv_row'] for x in additions],dtype=torch.long)
                ei=torch.cat([s.graph[FORWARD].edge_index[:,keep],added],1)
                attrs=torch.cat([s.graph[FORWARD].edge_attr[keep],s.graph[FORWARD].edge_attr[donor]])
                for rel,edge in [(FORWARD,ei),(REVERSE,ei.flip(0))]:
                    g[rel].edge_index=edge;g[rel].edge_attr=attrs;g[rel].simp_edge_batch=ei[0]*10000+ei[1]
                s.model.cpu()
                with torch.no_grad():p=float(forward(s.model,g)[r['node']].softmax(0)[1])
                assert abs(p-v['p_after'])<5e-5 and p>.5
                checks.append(dict(node=r['node'],split=split,family=f,explainer=e,arm=arm,error=abs(p-v['p_after'])))
    assert state_sha(s.model)==MODEL_HASH
    dump(out/'independent_verification.json',dict(passed=True,all5252_records_audited=True,all1313_targets_covered=True,
        all10k_accounts_audited=True,summary_metrics_checked=True,source_hashes_verified=True,full_graph_spot_checks=checks,
        summary_sha256=sha(out/'summary.json')))
    print('INDEPENDENT VERIFICATION PASSED',flush=True)

if __name__=='__main__':main()
