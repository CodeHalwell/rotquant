from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.colab_runtime import run_live


@pytest.mark.skipif(os.name != "posix", reason="POSIX process containment")
def test_timeout_terminates_workers_in_nested_sessions(tmp_path):
    pid_file = tmp_path / "pid"
    worker = f"import os,time;from pathlib import Path;Path({str(pid_file)!r}).write_text(str(os.getpid()));time.sleep(60)"
    nested = (
        "import sys;from scripts.colab_runtime import run_live;"
        f"run_live([sys.executable,'-c',{worker!r}],'inner',"
        f"repo_dir={os.getcwd()!r},log_root={str(tmp_path / 'inner')!r})"
    )
    environment = os.environ.copy()
    os.environ["PYTHONPATH"] = os.getcwd()
    try:
        with pytest.raises(TimeoutError):
            run_live([sys.executable, "-c", nested], "outer", repo_dir=Path.cwd(),
                     log_root=tmp_path / "outer", timeout_seconds=2)
        pid = int(pid_file.read_text())
        state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                               capture_output=True, text=True, check=False).stdout.strip()
        assert not state or state.startswith("Z")
    finally:
        os.environ.clear()
        os.environ.update(environment)
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
