"""Isolated pilot; historical checkpoints, experiments and data are read-only."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
import sys,json,hashlib,importlib.util,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
spec=importlib.util.spec_from_file_location('retraining_guard',WORK/'mega_attack_10_v1/resource_guard.py')
rg=importlib.util.module_from_spec(spec);spec.loader.exec_module(rg);rg.ROOT=ROOT
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
write=rg.write_json
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def validate():
    p=read(ROOT/'protocol.json')
    for name,value in p['source_hashes'].items():assert sha(name)==value,f'Frozen input changed: {name}'
    assert read(WORK/'mega_attack_eval/results_v1/status.json')['state']=='paused'
    assert read(WORK/'mega_attack_10_v1/controller_status.json')['state']=='stopped'
    assert not (ROOT/'STOP').exists(),'Stop requested'
    return p
def lock_file(path):
    import msvcrt
    stream=open(path,'a+b');stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
    return stream
def science():
    sys.path.insert(0,str(WORK/'mega_priority_repair'))
    import repair_core as pc
    sys.path.insert(0,str(WORK/'mega_attack_audit_v1'))
    import audited_attacks as attacks
    return pc,attacks
def start(label):
    p=validate();lock=lock_file(WORK/'mega_attack_eval/worker.lock');guard=rg.Guard(label)
    pc,attacks=science();s=pc.RepairStudy();pc.load_pg(s);guard.install(s)
    assert pc.state_sha(s.model)==pc.MODEL_HASH
    return p,lock,guard,pc,attacks,s
def save_torch(torch,path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def state_for(pc,arm):
    if arm=='baseline':return pc.torch.load(pc.CHECKPOINT,map_location='cpu',weights_only=False)['state_dict']
    assert read(ROOT/f'models/{arm}.json')['checkpoint_sha256']==sha(ROOT/f'models/{arm}.pt')
    row=pc.torch.load(ROOT/f'models/{arm}.pt',map_location='cpu',weights_only=False)
    assert row['protocol_sha256'] in [sha(ROOT/'protocol.json'),read(ROOT/'protocol.json').get('preserved_training_protocol_sha256')]
    return row['state_dict']
def cpu_context(pc,c):return pc.move_context(c,'cpu')
def status(**kwargs):write(ROOT/'status.json',dict(timestamp=time.time(),**kwargs))
