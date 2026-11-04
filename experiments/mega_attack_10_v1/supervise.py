"""Small CPU-only controller. Serial workers, memory cooldowns, no sweeps."""
import json
import hashlib
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from resource_guard import POLICY, snapshot, gpu_temperature, hold_reason, wait_for_headroom, write_json

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'mega_attack_audit_v1'
PYTHON=Path(r'C:\habib dissertation\Projects\AML-GNN-Replication-v2\.venv\Scripts\python.exe')
STATUS=ROOT/'controller_status.json'
read=lambda p:json.loads(Path(p).read_text())


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate():
    p=read(ROOT/'results/protocol.json')
    changed=[str(path) for path,h in p['source_hashes'].items() if sha(path)!=h]
    assert not changed, f'Frozen sources/checkpoints changed: {changed}'
    assert read(ROOT.parent/'mega_attack_eval/results_v1/status.json')['state']=='paused'
    assert len(p['cohorts']['validation_pilot'])==10 and p['expected']==40
    for path in [ROOT/'worker_failure.json',ROOT/'results/failure.json',OLD/'results/failure.json']:
        assert not path.exists(), f'Recorded failure requires diagnosis: {path}'


def workers():
    found=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            cmd=' '.join(p.info['cmdline'] or []).replace('\\','/').lower()
            if (p.info['name'] or '').lower().startswith('python') and 'work/mega_' in cmd:
                if any(x in cmd for x in ['/run.py','/verify.py','/guarded.py','/pipeline.py','/gnn_convergence.py','/fit_pg.py']):found.append(p)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found


def stop_known(processes,reason):
    # The caller supplies only processes whose full command lines were inspected.
    write_json(ROOT/'resource_hold.json',dict(reason=reason,timestamp=time.time(),
        memory=snapshot(),stopped_pids=[p.pid for p in processes],saved_results_retained=True))
    for p in processes:
        try:p.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(processes,timeout=5)
    for p in alive:
        try:p.kill()
        except psutil.NoSuchProcess:pass


def await_previous_worker():
    while True:
        live=workers()
        if not live:return
        for p in live:
            cmd=' '.join(p.cmdline()).replace('\\','/').lower()
            assert 'mega_attack_audit_v1/run.py' in cmd, 'Unexpected GPU worker; refusing to overlap'
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        mem=snapshot();temperature=gpu_temperature()
        reason=hold_reason(mem,temperature)
        write_json(STATUS,dict(state='waiting_for_current_pilot',current_worker_pids=[p.pid for p in live],
            planned_nodes=10,expected_new_records=40,memory=mem,temperature_c=temperature,resource_policy=POLICY,timestamp=time.time()))
        if reason:
            stop_known(live,reason)
            wait_for_headroom(STATUS,POLICY['cooldown_seconds'])
            return
        time.sleep(2)


def launch(script,*args):
    assert not workers(), 'No parallel GPU workers allowed'
    validate()
    wait_for_headroom(STATUS)
    command=[str(PYTHON),'-B','-u',str(ROOT/'guarded.py'),str(script),*map(str,args)]
    while True:
        write_json(STATUS,dict(state='running',script=str(script),arguments=list(args),timestamp=time.time()))
        with (ROOT/'run.log').open('a',encoding='utf8') as out,(ROOT/'errors.log').open('a',encoding='utf8') as err:
            p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=0x08000000)
            write_json(ROOT/'active_worker.json',dict(launcher_pid=p.pid,command=command,timestamp=time.time()))
            code=p.wait()
        write_json(ROOT/'last_exit.json',dict(code=code,script=str(script),timestamp=time.time()))
        if code==75:
            wait_for_headroom(STATUS,POLICY['cooldown_seconds'])
            validate()
            # A resource monitor can fire just after the last atomic result.
            # Do not rerun a completed worker after the cooldown.
            if script.name=='verify.py' and (script.parent/'results/independent_verification.json').exists():return
            if script==OLD/'run.py' and (OLD/'results/runner_verification.json').exists():return
            if script==ROOT/'run.py' and args:
                node=int(args[-1])
                done=[ROOT/'results/records'/f'{a}_{e}_{node}.json' for a in ['deduction','gx'] for e in ['pgexplainer','gnnexplainer']]
                if all(x.exists() for x in done):
                    # The final worker must also finish aggregate verification.
                    # If interrupted before that, leave a reviewable resource hold.
                    if len(list((ROOT/'results/records').glob('*.json')))==40 and not (ROOT/'results/runner_verification.json').exists():
                        raise RuntimeError('All cases saved but final aggregate interrupted; verify saved results before proceeding')
                    return
            continue
        if code!=0:raise RuntimeError(f'Worker exited {code}; inspect errors.log and failure files. No blind retry.')
        return


def main():
    lock=open(ROOT/'controller.lock','a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    validate();await_previous_worker();validate()
    if not (OLD/'results/runner_verification.json').exists():launch(OLD/'run.py')
    if not (OLD/'results/independent_verification.json').exists():launch(OLD/'verify.py')
    assert read(OLD/'results/independent_verification.json')['passed']
    p=read(ROOT/'results/protocol.json')
    for node in p['cohorts']['validation_pilot']:
        paths=[ROOT/'results/records'/f'{a}_{e}_{node}.json' for a in ['deduction','gx'] for e in ['pgexplainer','gnnexplainer']]
        if all(x.exists() for x in paths):continue
        launch(ROOT/'run.py','--node',node)
        # Exit of the per-node process releases all CPU/GPU allocation and caches.
        time.sleep(POLICY['cooldown_seconds'])
    if not (ROOT/'results/independent_verification.json').exists():launch(ROOT/'verify.py')
    assert read(ROOT/'results/independent_verification.json')['passed']
    write_json(STATUS,dict(state='complete',verified=True,nodes=10,records=40,timestamp=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        write_json(ROOT/'controller_failure.json',dict(timestamp=time.time(),traceback=traceback.format_exc()))
        write_json(STATUS,dict(state='failed',timestamp=time.time()))
        raise
