"""Runtime-only pacing and memory protection; never changes model arithmetic."""
import ctypes
from ctypes import wintypes
import json
import mmap
import os
from pathlib import Path
import subprocess
import threading
import time
import psutil

ROOT = Path(__file__).resolve().parent
GB = 1_000_000_000
POLICY = dict(system_ram_ceiling_bytes=30*GB, stop_system_used_bytes=int(29.5*GB),
              resume_system_used_bytes=25*GB, process_commit_limit_bytes=23*GB,
              cooldown_seconds=60, stable_seconds=30, gpu_active_fraction=1.0,
              max_gpu_temperature_c=75, resume_gpu_temperature_c=65,
              cpu_threads=2, poll_seconds=2)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf8')
    try: tmp.replace(path)
    except PermissionError: tmp.unlink(missing_ok=True)


def snapshot():
    mem = psutil.virtual_memory()
    return dict(total_bytes=mem.total, available_bytes=mem.available,
                used_bytes=mem.total-mem.available, timestamp=time.time())


def gpu_temperature():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, timeout=5, creationflags=0x08000000)
        if result.returncode == 0:
            return max(float(x.strip()) for x in result.stdout.splitlines() if x.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


class _BasicLimit(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong), ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD), ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]


class _IOCounters(ctypes.Structure):
    _fields_ = [(x, ctypes.c_ulonglong) for x in
                ['ReadOperationCount','WriteOperationCount','OtherOperationCount',
                 'ReadTransferCount','WriteTransferCount','OtherTransferCount']]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', _BasicLimit), ('IoInfo', _IOCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


def cap_own_process(limit_bytes):
    """Windows kernel-enforced committed-memory limit, queried back for proof."""
    if os.name != 'nt': raise RuntimeError('This resource policy requires Windows')
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    api.CreateJobObjectW.restype = wintypes.HANDLE
    api.GetCurrentProcess.restype = wintypes.HANDLE
    api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    api.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    handle = api.CreateJobObjectW(None, None)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    limits = _ExtendedLimit()
    limits.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    # Windows accounts committed memory in pages and rounds byte limits down.
    enforced_bytes = (limit_bytes // mmap.PAGESIZE) * mmap.PAGESIZE
    assert 0 < enforced_bytes <= limit_bytes
    limits.ProcessMemoryLimit = enforced_bytes
    for ok in [api.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)),
               api.AssignProcessToJobObject(handle, api.GetCurrentProcess())]:
        if not ok: raise ctypes.WinError(ctypes.get_last_error())
    actual = _ExtendedLimit()
    if not api.QueryInformationJobObject(handle, 9, ctypes.byref(actual), ctypes.sizeof(actual), None):
        raise ctypes.WinError(ctypes.get_last_error())
    assert actual.ProcessMemoryLimit == enforced_bytes and actual.BasicLimitInformation.LimitFlags & 0x100
    return handle, actual.ProcessMemoryLimit


def hold_reason(mem, temperature=None):
    if mem['used_bytes'] >= POLICY['stop_system_used_bytes']: return 'system_ram_ceiling'
    if temperature is not None and temperature >= POLICY['max_gpu_temperature_c']: return 'gpu_temperature'
    return None


class Guard:
    def __init__(self, label):
        self.label = label; self.started = time.time(); self.stopped = threading.Event()
        self.job, self.cap = cap_own_process(POLICY['process_commit_limit_bytes'])
        self.process = psutil.Process()
        self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.torch = None; self.window = time.perf_counter(); self.sleep_seconds = 0.; self.ticks = 0
        self.max_private = 0; self.max_rss = 0; self.max_system = 0
        self._hooks = []; self._installed = set(); self.temperature = None
        self.check()
        threading.Thread(target=self._monitor, daemon=True).start()

    def check(self):
        mem = snapshot(); info = self.process.memory_info()
        self.max_private = max(self.max_private, info.private)
        self.max_rss = max(self.max_rss, info.rss)
        self.max_system = max(self.max_system, mem['used_bytes'])
        reason = hold_reason(mem, self.temperature)
        data = dict(**mem, label=self.label, pid=os.getpid(), process_private_bytes=info.private,
                    process_rss_bytes=info.rss, peak_private_bytes=self.max_private,
                    peak_rss_bytes=self.max_rss, peak_system_used_bytes=self.max_system,
                    enforced_process_commit_cap=self.cap, gpu_temperature_c=self.temperature,
                    pacing_sleep_seconds=self.sleep_seconds, forward_checks=self.ticks,
                    policy=POLICY, state='resource_hold' if reason else 'running')
        write_json(ROOT/'resources.json', data)
        if reason:
            write_json(ROOT/'resource_hold.json', dict(**data, reason=reason,
                note='Worker exits to release memory. Completed atomic results are retained.'))
            os._exit(75)

    def _monitor(self):
        last_temperature = 0
        while not self.stopped.wait(POLICY['poll_seconds']):
            if time.monotonic()-last_temperature >= 15:
                self.temperature = gpu_temperature(); last_temperature = time.monotonic()
            self.check()

    def tick(self, *_):
        self.ticks += 1
        elapsed = time.perf_counter()-self.window
        if elapsed < .5: return
        pacing = POLICY['gpu_active_fraction'] < 1.0
        # CUDA completion is needed only to measure a deliberate pacing delay.
        # RAM/temperature checks use host observations and remain at the same cadence.
        if pacing and self.torch is not None and self.torch.cuda.is_initialized():
            self.torch.cuda.synchronize()
        elapsed = time.perf_counter()-self.window
        self.check()
        if pacing:
            delay = min(elapsed, 30.) * (1/POLICY['gpu_active_fraction']-1)
            if delay > 0:
                time.sleep(delay); self.sleep_seconds += delay
        self.window = time.perf_counter()

    def install(self, study):
        import torch
        self.torch = torch
        for module in [study.adapter, study.model]:
            if id(module) not in self._installed:
                self._hooks.append(module.register_forward_pre_hook(self.tick))
                self._installed.add(id(module))

    def close(self):
        self.stopped.set(); self.check()
        write_json(ROOT/'resource_runs'/f'{os.getpid()}.json', dict(label=self.label,
            started=self.started, finished=time.time(), peak_private_bytes=self.max_private,
            peak_rss_bytes=self.max_rss, peak_system_used_bytes=self.max_system,
            enforced_process_commit_cap=self.cap, pacing_sleep_seconds=self.sleep_seconds,
            forward_checks=self.ticks, policy=POLICY))


def wait_for_headroom(status, initial_cooldown=0):
    deadline = time.monotonic()+initial_cooldown; stable = None
    while True:
        mem = snapshot(); temp = gpu_temperature()
        safe = mem['used_bytes'] < POLICY['resume_system_used_bytes'] and (temp is None or temp <= POLICY['resume_gpu_temperature_c'])
        stable = (stable or time.monotonic()) if safe else None
        if stable is not None and time.monotonic()-stable >= POLICY['stable_seconds'] and time.monotonic() >= deadline: return
        write_json(status, dict(state='cooldown', memory=mem, temperature_c=temp, timestamp=time.time()))
        time.sleep(5)
