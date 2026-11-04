"""Single serial worker; resumable saved stages without changing settings."""
import subprocess,sys,traceback,time
from common import *
def worker(script,args):
    for attempt in range(4):
        validate();rg.wait_for_headroom(ROOT/'controller_status.json',rg.POLICY['cooldown_seconds'] if attempt else 0)
        write(ROOT/'controller_status.json',dict(state='running',script=script,args=args,timestamp=time.time()))
        with (ROOT/'run.log').open('a',encoding='utf8') as out,(ROOT/'errors.log').open('a',encoding='utf8') as err:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/script),*map(str,args)],cwd=ROOT,stdout=out,stderr=err,creationflags=0x08000000)
            write(ROOT/'active_worker.json',dict(pid=proc.pid,script=script,args=args,timestamp=time.time()));code=proc.wait()
        if code==75:continue
        assert code==0,f'{script} failed; inspect failure.json, no blind retry'
        return
    raise RuntimeError('Repeated resource holds; further automatic retries stopped')
def main():
    lock=lock_file(ROOT/'controller.lock');p=validate()
    assert read(ROOT/'selfcheck.json')['passed']
    assert not (ROOT/'failure.json').exists() and not (ROOT/'controller_failure.json').exists()
    for node in p['cohort']['attack_training_nodes']+p['cohort']['validation_nodes']:
        if not (ROOT/f'bank/{node}.json').exists():worker('bank.py',[node])
    for arm in ['clean','random','adversarial']:
        if not (ROOT/f'models/{arm}.json').exists():worker('train.py',[arm])
    for arm in p['arms']:
        if not (ROOT/f'detection/{arm}.json').exists():worker('detect.py',[arm])
    for arm in p['arms']:
        for node in p['cohort']['validation_nodes']:
            if not (ROOT/f'completed/eval_{arm}_{node}.json').exists():worker('evaluate.py',[arm,node])
    subprocess.run([sys.executable,'-B','-u',str(ROOT/'report.py')],check=True,cwd=ROOT,creationflags=0x08000000)
    assert read(ROOT/'independent_verification.json')['passed']
    write(ROOT/'controller_status.json',dict(state='complete',timestamp=time.time()))
if __name__=='__main__':
    try:main()
    except BaseException:
        write(ROOT/'controller_failure.json',dict(traceback=traceback.format_exc(),timestamp=time.time()))
        write(ROOT/'controller_status.json',dict(state='failed',timestamp=time.time()));raise
