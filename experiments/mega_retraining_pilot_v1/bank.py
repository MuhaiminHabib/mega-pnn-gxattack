"""New training-only attack bank; import historical validation edits without tuning."""
import sys,time,traceback
from common import *
def main(node):
    p,lock,guard,pc,attacks,s=start(f'Retraining bank account{node}')
    torch=pc.torch;is_train=node in p['cohort']['attack_training_nodes']
    assert is_train or node in p['cohort']['validation_nodes']
    assert bool(s.masks['train' if is_train else 'validation'][node]) and not bool(s.masks['test'][node])
    c,support,base,donors=s.context(node)
    if is_train:
        def progress(**kw):status(stage='training_bank',node=node,**{k:v for k,v in kw.items() if k!='stage'})
        audit=attacks.gx(s,support,base,progress)
    else:
        audit=read(WORK/f'mega_attack_10_v1/results/audits/gx_shared_{node}.json')
        domain=read(WORK/f'mega_attack_10_v1/results/domains/{node}.json');m=int(base.sum())
        assert support['tx_rows'][m:].cpu().tolist()==domain['added_ids'] and donors==domain['donor_csv_rows']
        assert support['nodes'][support['edge_index'][:,m:]].T.cpu().tolist()==domain['added_pairs']
    assert len(audit['attack_proposals'])==len(audit['random_proposals'])
    m=int(base.sum())
    for a,b in zip(audit['attack_proposals'],audit['random_proposals']):
        assert len(a['flips'])==len(b['flips']) and sum(i<m for i in a['flips'])==sum(i<m for i in b['flips'])
    row=dict(node=node,training=is_train,protocol_sha256=sha(ROOT/'protocol.json'),
        source='generated TRAIN only' if is_train else 'read-only historical validation edits',
        source_model_state=pc.state_sha(s.model),audit=audit)
    for arm in ['attack','random','ordinary']:
        entry=audit[arm]
        if entry is None or not entry['valid']:
            row[arm]=dict(valid=False);continue
        flips=entry['flips'];assert 0<len(flips)<=10 and len(set(flips))==len(flips)
        d=pc.subset(support,attacks.weights(base,flips).bool())
        with torch.no_grad():
            physical=float(pc.forward(s.model,pc.official(d,pc.DEVICE))[d['target']].softmax(0)[1])
            adapted=float(s.predict(support,attacks.weights(base,flips))[support['target']].softmax(0)[1])
        full=s.full_probability(support,base,flips)
        assert abs(full-physical)<5e-5 and abs(adapted-physical)<5e-5 and full>.5
        row[arm]=dict(valid=True,p=full,flips=flips,deletions=sum(i<m for i in flips),additions=sum(i>=m for i in flips),
            physical_replay_error=abs(full-physical),binary_error=abs(adapted-physical))
    row['paired_training_edits_valid']=row['attack']['valid'] and row['ordinary']['valid']
    if row['paired_training_edits_valid']:
        assert (row['attack']['deletions'],row['attack']['additions'])==(row['ordinary']['deletions'],row['ordinary']['additions'])
    save_torch(torch,ROOT/f'bank/{node}_contexts.pt',dict(clean=cpu_context(pc,c),support=cpu_context(pc,support),base=base.cpu(),donors=donors,protocol_sha256=sha(ROOT/'protocol.json')))
    write(ROOT/f'bank/{node}.json',row)
    assert pc.state_sha(s.model)==pc.MODEL_HASH
    validate();guard.close();print('BANK COMPLETE',node,row['paired_training_edits_valid'],flush=True)
if __name__=='__main__':
    try:main(int(sys.argv[1]))
    except BaseException:write(ROOT/'failure.json',dict(stage='bank',traceback=traceback.format_exc()));raise
