"""New training-only attack bank; import historical validation edits without tuning."""
import sys,time,traceback
from common import *
def main(node):
    p,lock,guard,pc,attacks,s=start(f'Retraining bank account{node}')
    torch=pc.torch;is_train=False
    assert node in p['cohort']['validation_nodes']
    assert bool(s.masks['validation'][node]) and not bool(s.masks['test'][node])
    c,support,base,donors=s.context(node)
    def progress(**kw):status(stage='validation_bank',node=node,**{k:v for k,v in kw.items() if k!='stage'})
    audit=attacks.gx(s,support,base,progress)
    assert len(audit['attack_proposals'])==len(audit['random_proposals'])
    m=int(base.sum())
    for a,b in zip(audit['attack_proposals'],audit['random_proposals']):
        assert len(a['flips'])==len(b['flips']) and sum(i<m for i in a['flips'])==sum(i<m for i in b['flips'])
    row=dict(node=node,training=is_train,protocol_sha256=sha(ROOT/'protocol.json'),
        source='New validation GX and matched controls generated once against ORIGINAL checkpoint, shared unchanged by all four models',
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
