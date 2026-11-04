"""Train the official MEGA-PNA on AMLSim accounts; audit global edge removal."""
import os
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
import argparse
import copy
import json
import random
import time
import traceback
import subprocess
import platform
from pathlib import Path
import numpy as np
import torch
import torch_geometric
import sklearn
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, confusion_matrix
from mega_amlsim_adapter import ROOT, prepare, make_model, no_transactions, forward, sha, tensor_sha, configure

OUT = ROOT/'results'/'amlsim_edge_audit_v1'
SEED = 42
MAX_EPOCHS = 200
PATIENCE = 40
EVAL_EVERY = 5


def dump(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def log(text):
    print(time.strftime('%Y-%m-%d %H:%M:%S'),text,flush=True)


def status(**values):
    dump(OUT/'status.json',dict(pid=os.getpid(),updated=time.time(),**values))


def weights_sha(model):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(tensor_sha(v).encode())
    return h.hexdigest()


def metrics(y,p):
    pred=p>.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),illicit=int(y.sum()),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),
                illicit_f1=float(f1_score(y,pred,zero_division=0)),
                macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
                precision=float(tp/(tp+fp)) if tp+fp else 0.,
                recall=float(tp/(tp+fn)) if tp+fn else 0.,
                average_precision=float(average_precision_score(y,p)),
                auroc=float(roc_auc_score(y,p)),predicted_illicit=int(pred.sum()))


def checkpoint_save(path, obj):
    temp=path.with_suffix('.tmp')
    torch.save(obj,temp)
    temp.replace(path)


def train(model,g,masks,protocol_hash):
    opt=torch.optim.Adam(model.parameters(),lr=.0008,weight_decay=1e-5)
    y=g['node'].y
    positive=int(y[masks['train']].sum())
    loss_weight=torch.tensor([1.,(int(masks['train'].sum())-positive)/positive])
    best_score=(-1.,-1.)
    best_epoch=0
    best_state=None
    start_epoch=1
    history=[]
    elapsed=0.
    latest=OUT/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu')
        assert saved['protocol_sha256']==protocol_hash
        model.load_state_dict(saved['state_dict']); opt.load_state_dict(saved['optimizer'])
        best_score=tuple(saved['best_score']); best_epoch=saved['best_epoch']
        best_state=saved['best_state']; start_epoch=saved['epoch']+1
        history=saved['history']; elapsed=saved['elapsed_seconds']
        torch.set_rng_state(saved['torch_rng'])
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        log(f'Resuming at epoch {start_epoch}; best validation epoch {best_epoch}')
    started=time.perf_counter()
    for epoch in range(start_epoch,MAX_EPOCHS+1):
        if epoch-1-best_epoch >= PATIENCE:
            break
        model.train(); opt.zero_grad(set_to_none=True)
        logits=forward(model,g)
        loss=torch.nn.functional.cross_entropy(logits[masks['train']],y[masks['train']],weight=loss_weight)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
        del logits
        row=dict(epoch=epoch,loss=float(loss.detach()))
        if epoch==1 or epoch % EVAL_EVERY==0 or epoch==MAX_EPOCHS:
            model.eval()
            with torch.no_grad():
                p=forward(model,g)[masks['validation']].softmax(1)[:,1].numpy()
            vm=metrics(y[masks['validation']].numpy(),p)
            row['validation']=vm
            score=(vm['illicit_f1'],vm['average_precision'])
            if score>best_score:
                best_score=score; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
                checkpoint_save(OUT/'best.pt',dict(state_dict=best_state,epoch=epoch,validation=vm,
                                protocol_sha256=protocol_hash,seed=SEED))
            log(f'Epoch {epoch}/{MAX_EPOCHS}: loss={row["loss"]:.4f}, val F1={vm["illicit_f1"]:.4f}, val AP={vm["average_precision"]:.4f}, best={best_score[0]:.4f}@{best_epoch}')
        history.append(row)
        seconds=elapsed+time.perf_counter()-started
        checkpoint_save(latest,dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                        best_state=best_state,best_score=best_score,best_epoch=best_epoch,history=history,
                        elapsed_seconds=seconds,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
                        python_rng=random.getstate(),protocol_sha256=protocol_hash))
        dump(OUT/'training_history.json',history)
        status(state='training',epoch=epoch,max_epochs=MAX_EPOCHS,best_epoch=best_epoch,
               best_validation_f1=best_score[0],elapsed_seconds=seconds)
    assert best_state is not None
    dump(OUT/'training_complete.json',dict(epochs_run=history[-1]['epoch'],best_epoch=best_epoch,
         elapsed_seconds=elapsed+time.perf_counter()-started,stopping='patience40_or_max200',protocol_sha256=protocol_hash))
    model.load_state_dict(best_state)


def evaluate(model,g,masks,ids,tx_ids,meta,protocol_hash,verify_only):
    saved=torch.load(OUT/'best.pt',map_location='cpu')
    assert saved['protocol_sha256']==protocol_hash
    model.load_state_dict(saved['state_dict']); model.eval()
    initial=weights_sha(model)
    xhash=tensor_sha(g['node'].x)
    old=None
    if (OUT/'predictions.npz').exists():
        with np.load(OUT/'predictions.npz') as file:
            old={k:file[k].copy() for k in ['full_graph','no_transaction_edges']}
    empty=no_transactions(g)
    assert sum(empty[rel].num_edges for rel in empty.edge_types)==0
    assert torch.equal(empty['node'].x,g['node'].x)
    with torch.no_grad():
        full=forward(model,g).softmax(1)[:,1].numpy().copy()
        removed=forward(model,empty).softmax(1)[:,1].numpy().copy()
        repeat=forward(model,g).softmax(1)[:,1].numpy().copy()
    assert np.isfinite(full).all() and np.isfinite(removed).all()
    assert np.max(np.abs(full-repeat))<=1e-6
    assert np.ptp(removed)<=1e-6,'Equal node features and zero edges must yield equal outputs'
    assert initial==weights_sha(model) and xhash==tensor_sha(g['node'].x)
    replay=None
    if old is not None:
        replay={k:float(np.max(np.abs(new-old[k]))) for k,new in [('full_graph',full),('no_transaction_edges',removed)]}
        assert max(replay.values())<=1e-6
    np.savez_compressed(OUT/'predictions.npz',full_graph=full,no_transaction_edges=removed)
    np.savez_compressed(OUT/'node_index.npz',account_id=ids,label=g['node'].y.numpy(),
                        **{k+'_mask':m.numpy() for k,m in masks.items()})
    # Row order is exactly the raw CSV, so saved IDs bind individual transfers to edges.
    np.save(OUT/'transaction_ids.npy',tx_ids)
    y=g['node'].y.numpy()
    all_metrics={c:{s:metrics(y[m.numpy()],p[m.numpy()]) for s,m in masks.items()}
                 for c,p in [('full_graph',full),('no_transaction_edges',removed)]}
    retention={}
    for name,mask in masks.items():
        cohort=mask.numpy() & (y==1) & (full>.5)
        n=int(cohort.sum()); stayed=int((removed[cohort]>.5).sum())
        delta=removed[cohort]-full[cohort]
        retention[name]=dict(clean_true_positives=n,remained_illicit=stayed,flipped_to_licit=n-stayed,
                             fraction=stayed/n if n else None,
                             mean_clean_probability=float(full[cohort].mean()) if n else None,
                             mean_removed_probability=float(removed[cohort].mean()) if n else None,
                             mean_signed_probability_change=float(delta.mean()) if n else None,
                             mean_absolute_probability_change=float(np.abs(delta).mean()) if n else None)
    result=dict(model='official MEGA-PNA (AMLSim account adapter)',seed=SEED,selected_epoch=saved['epoch'],
                protocol_sha256=protocol_hash,dataset=meta,metrics=all_metrics,retention=retention,
                empty_graph_probability_range=[float(removed.min()),float(removed.max())],
                parameter_count=sum(p.numel() for p in model.parameters()),
                checkpoint_sha256=sha(OUT/'best.pt'),predictions_sha256=sha(OUT/'predictions.npz'))
    dump(OUT/'summary.json',result)
    lines=['MEGA-PNA / AMLSim: account classification and global edge-removal audit','',
           'One fixed training seed (42); existing stratified account split seed 42.',
           '10000 accounts (1685 illicit), 1323234 individual transactions, 68947 directed account pairs.',
           'Node features are constant zero (upstream normalized ones); edges use normalized timestamp and amount.',
           'No account aggregates, account/customer IDs, behavior IDs, transaction fraud flags or alert IDs are input features.',
           'All accounts and edges are visible transductively; only training labels enter the supervised loss.',
           'AMLSim adapter uses the official MEGA model, raw encoders and node head; original model source is unchanged.',
           'Two MEGA-PNA layers, hidden width20, five towers, forward/reverse relations, edge updates, residuals, batch normalization, dropout0.1.',
           'Adam lr0.0008, weight decay0.00001; train-class inverse-frequency weights; full-batch CPU training.',
           'Maximum200 epochs, patience40, validation every5 epochs (also epoch1); checkpoint chosen by validation illicit F1, AP breaks ties.',
           'All classifications use p_illicit>0.5. No test threshold or model tuning.',
           f'Selected epoch: {saved["epoch"]}.',
           'Edge removal deletes all transfers, including 181 self-transfers and all reverse relation copies.',
           'No computational self-loops are inserted. PNA root-feature computations remain active.',
           'Trained weights, normalization buffers, feature preprocessing and node features stay fixed.',
           '', 'RESULTS (percent)', 'Split | condition | illicit F1 | precision | recall | average precision | TP | FP | FN | TN']
    for split in masks:
        for condition in all_metrics:
            m=all_metrics[condition][split]
            lines.append(' | '.join([split,condition]+[f'{100*m[k]:.2f}' for k in ['illicit_f1','precision','recall','average_precision']]+[str(m[k]) for k in ['tp','fp','fn','tn']]))
    lines+=['','CORRECTLY DETECTED ILLICIT ACCOUNTS AFTER REMOVAL', 'Split | remained illicit / originally correctly detected | mean confidence change (percentage points)']
    for split,r in retention.items():
        lines.append(f'{split} | {r["remained_illicit"]}/{r["clean_true_positives"]} | {100*r["mean_signed_probability_change"]:.2f}' if r['clean_true_positives'] else f'{split} | no clean true positives')
    lines+=['','LIMITATIONS',
            'This is a single-seed AMLSim adaptation, not a reproduction of the paper\'s IBM AML edge-classification benchmark.',
            'The dataset/task, full-batch training, class weights, stopping rule and edge feature availability differ from paper benchmarks.',
            'All original transfers are retained; no sampling, filtering of difficult accounts or collapsing parallel transfers was used.',
            'Constant node inputs mean that deleting all transactions makes all account inputs indistinguishable.',
            'The empty graph must therefore produce one common probability. It may classify all accounts as licit OR all as illicit.',
            'Zero label retention alone would not prove faithful explanations, and high retention could just mean an all-illicit output.',
            'This extreme out-of-distribution test measures transaction dependence, not explanation quality or adversarial robustness.',
            'Its feature design differs from the earlier aggregate-feature GraphSAGE; differences cannot be attributed to architecture alone.',
            'An account-random split does not establish performance on future time periods. No explainer or defense is trained here.',
            'Checkpoint replay, feature/weight hashes, every account prediction and all split denominators are saved.','']
    (OUT/'REPORT.txt').write_text('\n'.join(lines),encoding='utf-8')
    dump(OUT/'verification.json',dict(passed=True,mode='checkpoint_replay' if verify_only else 'initial_run',
         source_and_protocol_hashes_checked=True,weights_unchanged=True,node_features_unchanged=True,
         literal_remaining_edges=0,empty_output_range=float(np.ptp(removed)),clean_replay_max_error=float(np.max(np.abs(full-repeat))),
         saved_checkpoint_replay_errors=replay,checkpoint_sha256=result['checkpoint_sha256'],
         predictions_sha256=result['predictions_sha256'],summary_sha256=sha(OUT/'summary.json'),report_sha256=sha(OUT/'REPORT.txt')))
    log('Test results: '+json.dumps({c:m['test'] for c,m in all_metrics.items()}))
    log('Test retention: '+json.dumps(retention['test']))


def main():
    import hashlib
    parser=argparse.ArgumentParser()
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    OUT.mkdir(parents=True,exist_ok=True)
    status(state='preparing')
    g,masks,ids,tx_ids,meta=prepare()
    model=make_model(g)
    # Track every upstream Python source, the adapter and this runner.
    source_paths=sorted((ROOT/'MegaGNN').rglob('*.py'))+[ROOT/'mega_amlsim_adapter.py',Path(__file__)]
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    cfg,_=configure()
    protocol=dict(repo='https://github.com/hcagri/MEGA-GNN-TMLR',revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_hashes=sources,dataset=meta,seed=SEED,threads=args.threads,device='cpu',
                  optimizer='Adam',learning_rate=.0008,weight_decay=1e-5,epochs=MAX_EPOCHS,patience=PATIENCE,
                  eval_every=EVAL_EVERY,selection='validation_illicit_F1_then_AP',threshold=.5,
                  training='full_batch_all_transactions',loss='train_inverse_frequency_weighted_cross_entropy',
                  model_config=cfg.dump(),parameter_count=sum(p.numel() for p in model.parameters()),
                  environment=dict(python=platform.python_version(),torch=torch.__version__,pyg=torch_geometric.__version__,numpy=np.__version__,sklearn=sklearn.__version__))
    protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text())==protocol,'Protocol changed; refusing to mix runs'
    dump(OUT/'protocol.json',protocol)
    (OUT/'model.txt').write_text(str(model),encoding='utf-8')
    log(f'Dataset ready; {meta["transactions"]} transfers; {protocol["parameter_count"]} parameters; protocol {protocol_hash}')
    if not (OUT/'training_complete.json').exists():
        assert not args.verify_only,'Training is incomplete'
        train(model,g,masks,protocol_hash)
    assert sources=={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    status(state='evaluating')
    evaluate(model,g,masks,ids,tx_ids,meta,protocol_hash,args.verify_only)
    status(state='complete',report=str(OUT/'REPORT.txt'))
    log('COMPLETE '+str(OUT/'REPORT.txt'))


if __name__=='__main__':
    try:
        main()
    except Exception:
        status(state='failed',error=traceback.format_exc())
        raise
