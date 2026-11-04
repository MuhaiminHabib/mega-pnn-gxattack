"""Independent counts and saved-checkpoint checks; never trains."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from mega_amlsim_adapter import ROOT,prepare,make_model,no_transactions,sha,tensor_sha

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
out=ROOT/'results'/'amlsim_edge_audit_v1'
summary=json.loads((out/'summary.json').read_text())
protocol=json.loads((out/'protocol.json').read_text())
verification=json.loads((out/'verification.json').read_text())
assert verification['passed'] and verification['mode']=='checkpoint_replay'
assert max(verification['saved_checkpoint_replay_errors'].values())<=1e-6
assert sha(out/'summary.json')==verification['summary_sha256']
assert sha(out/'REPORT.txt')==verification['report_sha256']
assert sha(out/'best.pt')==summary['checkpoint_sha256']
assert sha(out/'predictions.npz')==summary['predictions_sha256']
index=np.load(out/'node_index.npz')
pred=np.load(out/'predictions.npz')
y=index['label']
assert len(y)==len(set(index['account_id']))==10000 and y.sum()==1685
assert np.array_equal(sum(index[k+'_mask'].astype(int) for k in ['train','validation','test']),np.ones(10000,dtype=int))
checks=[]
for condition in ['full_graph','no_transaction_edges']:
    p=pred[condition]
    assert p.shape==y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    for split in ['train','validation','test','all']:
        m=index[split+'_mask']
        yy=y[m]; pp=p[m]>.5
        tp=int(((yy==1)&pp).sum()); fp=int(((yy==0)&pp).sum())
        fn=int(((yy==1)&~pp).sum()); tn=int(((yy==0)&~pp).sum())
        saved=summary['metrics'][condition][split]
        assert [tp,fp,fn,tn]==[saved[k] for k in ['tp','fp','fn','tn']]
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        assert abs(saved['illicit_f1']-f1)<1e-12
        assert abs(saved['average_precision']-average_precision_score(yy,p[m]))<1e-12
        assert abs(saved['auroc']-roc_auc_score(yy,p[m]))<1e-12
        if condition=='no_transaction_edges':
            cohort=m & (y==1) & (pred['full_graph']>.5)
            stayed=int((p[cohort]>.5).sum())
            r=summary['retention'][split]
            assert int(cohort.sum())==r['clean_true_positives']
            assert stayed==r['remained_illicit']
        checks.append(dict(condition=condition,split=split,counts_f1_ap_auroc_passed=True))

g,masks,ids,tx_ids,metadata=prepare()
assert metadata==protocol['dataset']
raw_edges=g[('node','to','node')].edge_index
incident=torch.bincount(raw_edges.flatten(),minlength=10000).numpy()
assert np.array_equal(ids,index['account_id'])
assert np.array_equal(tx_ids,np.load(out/'transaction_ids.npy'))
assert np.array_equal(y,g['node'].y.numpy())
for name,mask in masks.items():
    assert np.array_equal(mask.numpy(),index[name+'_mask'])
def graph_hashes(graph):
    result={'x':tensor_sha(graph['node'].x),'y':tensor_sha(graph['node'].y)}
    for rel in graph.edge_types:
        for key in ['edge_index','edge_attr','simp_edge_batch']:
            result[str(rel)+key]=tensor_sha(graph[rel][key])
    return result
before=graph_hashes(g)
model=make_model(g)
model.load_state_dict(torch.load(out/'best.pt',map_location='cpu')['state_dict'])
model.eval()
state_before={k:tensor_sha(v) for k,v in model.state_dict().items()}
replay={}
for condition,graph in [('full_graph',g),('no_transaction_edges',no_transactions(g))]:
    with torch.no_grad():
        logits,labels=model(graph.clone())
    assert torch.equal(labels,g['node'].y),'Node head changed account order'
    p=logits.softmax(1)[:,1].numpy()
    error=float(np.max(np.abs(p-pred[condition])))
    assert error<=1e-6
    replay[condition]=error
assert before==graph_hashes(g)
assert state_before=={k:tensor_sha(v) for k,v in model.state_dict().items()}
empty_graph=no_transactions(g)
assert sum(int(empty_graph[r].edge_index.size(1)) for r in g.edge_types)==0
for filename,expected in protocol['source_hashes'].items():
    assert sha(ROOT/filename)==expected
result=dict(passed=True,metric_checks=checks,checkpoint_replay_max_errors=replay,
            raw_data_and_account_order_verified=True,all_graph_inputs_unchanged=True,
            all_model_parameters_and_buffers_unchanged=True,literal_zero_edges_verified=True,
            every_recorded_source_hash_verified=True,
            original_isolated_accounts=int((incident==0).sum()),
            clean_test_true_positives_already_isolated=int((index['test_mask'] & (y==1) & (pred['full_graph']>.5) & (incident==0)).sum()))
(out/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
