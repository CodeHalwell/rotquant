"""Contain nested workers even when they start separate process sessions."""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time


class ProcessTree:
    def __init__(self, process):
        self.process = process
        self.members = set()

    def snapshot(self):
        output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,stat="], text=True)
        return {int(pid): (int(parent), state) for pid, parent, state in
                (line.split() for line in output.splitlines())}

    def send(self, pid, sig):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, sig)

    def freeze(self, pid):
        if pid in self.members:
            return
        self.members.add(pid)
        self.send(pid, signal.SIGSTOP)
        for child, (parent, _) in self.snapshot().items():
            if parent == pid:
                self.freeze(child)

    def running(self):
        return {pid for pid, (_, state) in self.snapshot().items()
                if pid in self.members and not state.startswith("Z")}

    def terminate(self):
        self.freeze(self.process.pid)
        for sig in (signal.SIGTERM, signal.SIGCONT):
            for pid in self.members:
                self.send(pid, sig)
        deadline = time.monotonic() + 10
        while self.running() and time.monotonic() < deadline:
            self.process.poll()
            time.sleep(.05)
        for pid in self.running():
            self.send(pid, signal.SIGKILL)
        self.process.wait(timeout=10)
        deadline = time.monotonic() + 10
        while self.running():
            if time.monotonic() >= deadline:
                raise RuntimeError("Nested workers did not terminate")
            time.sleep(.05)
