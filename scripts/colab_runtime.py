"""Small stdlib-only live subprocess wrapper for resumable Colab experiments."""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path


def run_live(command, label, *, repo_dir, log_root, timeout_seconds=None):
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    if Path(label).name != label:
        raise ValueError("log label must be a file basename")
    path = log_root / f"{label}.log"
    print("Running:", " ".join(map(str, command)), flush=True)
    print("Persistent log:", path, flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    started = time.monotonic()
    with path.open("a", buffering=1) as sink, path.open("r") as reader:
        reader.seek(0, 2)
        process = subprocess.Popen(command, cwd=repo_dir, env=env, stdout=sink,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        print({"pid": process.pid, "tail_command": f'tail -f "{path}"'}, flush=True)
        heartbeat = started
        try:
            while True:
                text = reader.read()
                if text:
                    print(text, end="", flush=True)
                if process.poll() is not None:
                    print(reader.read(), end="", flush=True)
                    break
                now = time.monotonic()
                if timeout_seconds and now - started > timeout_seconds:
                    raise TimeoutError(f"{label} exceeded {timeout_seconds}s; see {path}")
                if now - heartbeat >= 30:
                    print(f"notebook heartbeat {label}: {(now-started)/60:.1f}m; pid={process.pid}", flush=True)
                    heartbeat = now
                time.sleep(1)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    return path
