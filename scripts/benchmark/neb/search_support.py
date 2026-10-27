"""Shared helpers for the NEB path-search tool."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def runtime_inventory(binaries):
    """Pin resolved shared-library contents without copying the system runtime."""
    records = {}
    for name, binary in binaries.items():
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(binary.parent)+':'+env.get('LD_LIBRARY_PATH', '')
        result = subprocess.run(['ldd', str(binary)], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        libraries = {}
        for line in result.stdout.splitlines():
            fields = line.split()
            candidate = fields[2] if len(fields) > 2 and fields[1] == '=>' else fields[0] if fields else ''
            if candidate.startswith('/') and Path(candidate).is_file():
                libraries[str(Path(candidate).resolve())] = sha(candidate)
        records[name] = dict(returncode=result.returncode, ldd=result.stdout, libraries=libraries)
    return records


def settings(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            key, value = line.split(None, 1)
            result[key] = value
    return result


def physical_cores():
    allowed = os.sched_getaffinity(0)
    lines = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET'], text=True).splitlines()
    selected = {}
    for line in lines:
        if line.startswith('#'): continue
        cpu, core, socket = map(int, line.split(','))
        if cpu in allowed: selected.setdefault((socket, core), cpu)
    return list(selected.values())


def execute(command, folder, env, timeout):
    start = time.perf_counter()
    with (folder/'run.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try: code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            code = 'timeout'
    return code, time.perf_counter()-start
