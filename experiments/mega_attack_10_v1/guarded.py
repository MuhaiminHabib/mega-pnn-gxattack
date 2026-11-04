"""Apply runtime protection before importing Torch; preserve scientific sources."""
import os
import sys
import runpy
import traceback
from pathlib import Path
from resource_guard import Guard, write_json

ROOT = Path(__file__).resolve().parent
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '2'


def main():
    target = Path(sys.argv[1]).resolve()
    allowed = {(ROOT/name).resolve() for name in ['run.py','verify.py']}
    allowed |= {(ROOT.parent/'mega_attack_audit_v1'/name).resolve() for name in ['run.py','verify.py']}
    assert target in allowed
    guard = Guard(str(target))
    sys.path.insert(0, str(ROOT.parent/'mega_attack_eval'))
    import bridge
    original = bridge.Study.__init__
    def protected_init(study, *args, **kwargs):
        guard.check()
        original(study, *args, **kwargs)
        guard.install(study)
        guard.check()
    bridge.Study.__init__ = protected_init
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target)] + sys.argv[2:]
    try:
        runpy.run_path(str(target), run_name='__main__')
    except BaseException:
        write_json(ROOT/'worker_failure.json', dict(target=str(target), pid=os.getpid(), traceback=traceback.format_exc()))
        raise
    finally:
        guard.close()


if __name__ == '__main__': main()
