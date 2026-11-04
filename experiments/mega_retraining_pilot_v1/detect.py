"""Full validation detection metrics; no held-out test labels are queried."""
import sys,traceback
from common import *
def baseline_replay(np,indices,prob,old):
    reference=np.load(ROOT/'recovery_baseline_replay/same_preprocessing_reference.npz')
    assert np.array_equal(reference['nodes'],indices)
    current_error=float(np.max(np.abs(reference['probabilities']-prob)))
    historical_error=float(np.max(np.abs(old-prob)))
    assert current_error<=1e-6,'Same-runtime original-checkpoint replay changed'
    assert historical_error<=5e-5,'Historical-runtime difference exceeds diagnosed bound'
    assert np.array_equal(old>.5,prob>.5),'Historical baseline labels changed'
    return dict(same_runtime_max_error=current_error,historical_runtime_max_error=historical_error,historical_labels_identical=True)
def main(arm):
    p,lock,guard,pc,attacks,s=start(f'Validation detection {arm}')
    torch,np=pc.torch,pc.np
    s.model.load_state_dict(state_for(pc,arm));s.model.eval().cpu();before=pc.state_sha(s.model)
    selected=s.masks['validation'];indices=selected.nonzero().flatten().numpy()
    with torch.no_grad():prob=pc.forward(s.model,s.graph)[selected].softmax(1)[:,1].numpy().copy()
    y=s.graph['node'].y[selected].numpy();pred=prob>.5
    tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tn=int(((y==0)&~pred).sum())
    metrics=dict(n=len(y),illicit=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        illicit_f1=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
    assert len(y)==1500 and int(y.sum())==253 and pc.state_sha(s.model)==before
    replay=None
    if arm=='baseline':
        old=np.load(pc.ROOT/'results/amlsim_edge_audit_v1/predictions.npz')['full_graph'][indices]
        replay=baseline_replay(np,indices,prob,old)
    np.savez_compressed(ROOT/f'detection/{arm}.npz',nodes=indices,labels=y,probabilities=prob)
    write(ROOT/f'detection/{arm}.json',dict(arm=arm,split='validation',state_sha256=before,metrics=metrics,
        no_test_metrics=True,baseline_replay=replay,protocol_sha256=sha(ROOT/'protocol.json')))
    validate();guard.close();print('DETECTION',arm,metrics,flush=True)
if __name__=='__main__':
    try:main(sys.argv[1])
    except BaseException:write(ROOT/'failure.json',dict(stage='detect',traceback=traceback.format_exc()));raise
