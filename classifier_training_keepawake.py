"""Temporary Windows idle-sleep request while one classifier queue is running.

Does not change power-plan settings, keep the display on, or block manual sleep.
The request is released when the queue completes, stops, exits, or times out.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys
import time


def watch(status_file: Path, max_hours=24):
    if sys.platform != 'win32':
        raise RuntimeError('This helper is only for Windows')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel.SetThreadExecutionState.restype = wintypes.DWORD
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    state = json.loads(status_file.read_text(encoding='utf-8'))
    if state['state'] != 'running':
        return
    handle = kernel.OpenProcess(0x1000, False, state['pid'])
    if not handle:
        raise OSError(ctypes.get_last_error(), 'Cannot observe queue process')
    continuous, system_required = 0x80000000, 0x00000001
    deadline = time.monotonic() + max_hours * 3600
    try:
        if not kernel.SetThreadExecutionState(continuous | system_required):
            raise OSError(ctypes.get_last_error(), 'Cannot request temporary system wakefulness')
        print('Temporary idle-sleep request active for classifier queue.', flush=True)
        while time.monotonic() < deadline:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
                break
            latest = json.loads(status_file.read_text(encoding='utf-8'))
            if latest.get('state') != 'running' or latest.get('pid') != state['pid']:
                break
            time.sleep(15)
    finally:
        kernel.SetThreadExecutionState(continuous)
        kernel.CloseHandle(handle)
        print('Temporary idle-sleep request released.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status-file', type=Path, required=True)
    parser.add_argument('--max-hours', type=float, default=24)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 48:
        parser.error('--max-hours must be in (0,48]')
    watch(args.status_file, args.max_hours)
