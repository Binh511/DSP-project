"""One-shot Windows controller: finish the active worker, dispatch no next run.

Only the queue dispatcher is terminated. The worker, frozen training code,
checkpoints and data are left intact. No scheduled or recurring job is created.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'reports/research_review/studies/classifier_source_clean_context_study_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding='utf-8-sig'))
        except json.JSONDecodeError:
            if attempt == 19:
                raise
            time.sleep(.1)


def write(path, value):
    temporary = path.with_name(path.name + '.stop-controller.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def control(expected_run):
    if sys.platform != 'win32':
        raise RuntimeError('This controller is for Windows only')
    status_path = STUDY / 'queue_status.json'
    receipt_path = STUDY / 'stop_after_current.json'
    state = read(status_path)
    if state.get('stop_after_current'):
        raise RuntimeError('A stop request is already registered; do not dispatch a second controller')
    if state['state'] != 'running' or state['current'] != expected_run:
        raise RuntimeError('Queue state changed; inspect before taking action')
    dispatcher_pid = state['pid']
    dispatcher = psutil.Process(dispatcher_pid)
    command = dispatcher.cmdline()
    if '--queue' not in command or not any(Path(arg).name == 'classifier_context_experiment.py' for arg in command):
        raise RuntimeError('Dispatcher identity mismatch')
    workers = []
    for process in dispatcher.children(recursive=False):
        command = process.cmdline()
        if '--run' in command and command[command.index('--run') + 1] == expected_run:
            workers.append(process)
    if len(workers) != 1:
        raise RuntimeError('Exactly one matching active training worker is required')
    worker = workers[0]
    lock = STUDY / 'queue.lock'
    if lock.read_text(encoding='utf-8').strip() != str(dispatcher_pid):
        raise RuntimeError('Queue lock owner mismatch')
    plan = read(STUDY / 'experiment_plan.json')
    for filename, expected_hash in plan['code_sha256'].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError('Frozen source changed: ' + filename)

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel.SetThreadExecutionState.restype = wintypes.DWORD
    query_sync = 0x1000 | 0x100000
    dispatcher_handle = kernel.OpenProcess(query_sync | 1, False, dispatcher_pid)
    worker_handle = kernel.OpenProcess(query_sync, False, worker.pid)
    if not dispatcher_handle or not worker_handle:
        if dispatcher_handle:
            kernel.CloseHandle(dispatcher_handle)
        if worker_handle:
            kernel.CloseHandle(worker_handle)
        raise ctypes.WinError()
    receipt = dict(requested_utc=now(), reason='explicit_user_request_finish_current_then_stop',
                   controller_pid=os.getpid(), dispatcher_pid=dispatcher_pid,
                   worker_pid=worker.pid, finish_run=expected_run, phase='preflight',
                   frozen_code_unchanged=True)
    awake = False
    try:
        if not dispatcher.is_running() or not worker.is_running() or worker.ppid() != dispatcher_pid:
            raise RuntimeError('Process identities changed')
        latest = read(status_path)
        if latest != state or kernel.WaitForSingleObject(worker_handle, 0) != 258:
            raise RuntimeError('Worker or queue changed during preflight')
        # Own the idle-sleep request before the old queue helper releases it.
        awake = bool(kernel.SetThreadExecutionState(0x80000000 | 1))
        write(receipt_path, receipt)
        if not kernel.TerminateProcess(dispatcher_handle, 0):
            raise ctypes.WinError()
        if kernel.WaitForSingleObject(dispatcher_handle, 10000) != 0:
            raise RuntimeError('Dispatcher termination was not confirmed')
        receipt.update(phase='waiting_for_current_worker', dispatcher_stopped_utc=now(),
                       worker_alive_after_dispatcher_stop=kernel.WaitForSingleObject(worker_handle, 0) == 258)
        state.update(pid=os.getpid(), original_queue_pid=dispatcher_pid, worker_pid=worker.pid,
                     stop_after_current=expected_run, stop_requested_utc=receipt['requested_utc'],
                     dispatcher_stopped=True)
        # Keep running=true while the worker is active for the demo's GPU guard.
        write(status_path, state)
        write(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
        while True:
            result = kernel.WaitForSingleObject(worker_handle, 15000)
            if result == 0:
                break
            if result != 258:
                raise ctypes.WinError()
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(worker_handle, ctypes.byref(exit_code)):
            raise ctypes.WinError()
        folder = STUDY / 'experiments' / expected_run
        complete = (exit_code.value == 0 and (folder / 'run_status.json').is_file()
                    and read(folder / 'run_status.json').get('state') == 'complete'
                    and (folder / 'validation_metrics.json').is_file())
        if complete and expected_run not in state['completed']:
            state['completed'].append(expected_run)
        state.update(state='stopped', current=None, stopped_utc=now(),
                     stop_reason='user_requested_after_current', worker_exit_code=exit_code.value,
                     skipped=[name for name in state['planned'] if name not in state['completed']])
        if not complete:
            state['error'] = 'Current worker exited before a complete export/evaluation; no next run started'
        write(status_path, state)
        if lock.exists() and lock.read_text(encoding='utf-8').strip() == str(dispatcher_pid):
            lock.unlink()
        # Only aggregate saved validation results; never starts training.
        cpu_python = ROOT / '.venv/classifier_demo_cpu/Scripts/python.exe'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTORCH_NVML_BASED_CUDA_CHECK='1')
        summary = subprocess.run([str(cpu_python if cpu_python.is_file() else Path(sys.executable)),
                                  '-u', '-c', 'import classifier_context_experiment as c; c.summarize()'],
                                 cwd=ROOT, env=env, check=False)
        receipt.update(phase='stopped', completed_utc=now(), worker_exit_code=exit_code.value,
                       current_run_complete=complete, completed_runs=state['completed'],
                       skipped_runs=state['skipped'], summary_exit_code=summary.returncode)
        write(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        receipt.update(phase='controller_error', error=str(error), updated_utc=now())
        write(receipt_path, receipt)
        raise
    finally:
        if awake:
            kernel.SetThreadExecutionState(0x80000000)
        kernel.CloseHandle(worker_handle)
        kernel.CloseHandle(dispatcher_handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-run', required=True)
    arguments = parser.parse_args()
    control(arguments.expected_run)
