"""Stdlib-only, persistent stage runner for the native GPU notebook.

Only completed stages with unchanged requests and artifact hashes are reused.
Failed attempts are retained. Active stage time, not notebook idle time, consumes
the configured execution allowance; neither this runner nor its cap stops billing.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import tempfile
import time
import zipfile
from pathlib import Path

from scripts.colab_runtime import run_live
from scripts.native_hashing import digest


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def artifacts(paths):
    return {str(Path(p).resolve(strict=True)): {"bytes": Path(p).stat().st_size, "sha256": digest(p)}
            for p in paths}


def artifacts_match(record):
    try:
        return all(Path(p).is_file() and not Path(p).is_symlink()
                   and Path(p).stat().st_size == v["bytes"] and digest(p) == v["sha256"]
                   for p, v in record.items())
    except OSError:
        return False


class Stage:
    def __init__(self, workflow, directory, limit):
        self.workflow, self.directory, self.limit = workflow, directory, limit
        self.started = time.monotonic()

    def command(self, command, label):
        remaining = self.limit - (time.monotonic() - self.started)
        if remaining <= 0:
            raise TimeoutError("Active execution allowance exhausted; no new command started")
        return run_live(list(map(str, command)), label, repo_dir=self.workflow.repo,
                        log_root=self.directory / "logs", timeout_seconds=remaining)


class Workflow:
    def __init__(self, root, repo, controls, budget_minutes=90):
        if not 1 <= budget_minutes <= 180:
            raise ValueError("active execution budget must be 1..180 minutes")
        self.root, self.repo = Path(root).resolve(), Path(repo).resolve()
        self.budget = budget_minutes * 60
        self.controls = {**controls, "active_budget_minutes": budget_minutes}
        self.path = self.root / "workflow.json"
        self.lock = None

    def __enter__(self):
        # A local advisory lock releases automatically after a killed kernel.
        # Do not rely on FUSE/Drive advisory-lock support.
        directory = Path(tempfile.gettempdir()) / "rotquant-native-workflow-locks"
        directory.mkdir(exist_ok=True)
        self.lock = (directory / (fingerprint(str(self.root)) + ".lock")).open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.root.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                if self.state["controls"] != self.controls:
                    raise ValueError("Run controls changed: choose a new RUN_NAME; previous results are preserved")
            else:
                self.state = {"protocol": "rq3-gpu-e2e-v2", "controls": self.controls,
                              "active_seconds": 0., "attempts": [], "status": "ready"}
            # A process killed without cleanup left a running receipt. Charge
            # conservatively, bounded by the declared phase timeout.
            for row in self.state["attempts"]:
                if row["status"] == "running":
                    row["status"] = "interrupted"
                    elapsed = min(row["limit_seconds"], max(0., time.time() - row["started_utc_epoch"]))
                    row["elapsed_seconds"] = elapsed
                    self.state["active_seconds"] += elapsed
            self.state["status"] = "running"
            self.save()
            return self
        except BaseException:
            self.lock.close()
            raise

    def save(self):
        write_json(self.path, self.state)

    def stage(self, name, action, *, signature=None, minutes=10, reuse=True):
        if not name or Path(name).name != name:
            raise ValueError("stage name must be a basename")
        key = fingerprint(signature)
        previous = [r for r in self.state["attempts"] if r["name"] == name]
        if reuse and previous:
            row = previous[-1]
            if row["status"] == "passed" and row["signature"] == key and artifacts_match(row["artifacts"]):
                print(f"RESUME {name}: verified completed artifacts", flush=True)
                return row["value"]
        remaining = self.budget - self.state["active_seconds"]
        if remaining <= 0:
            raise TimeoutError("Active execution allowance exhausted; completed work is preserved. Stop the GPU runtime.")
        directory = self.root / "stages" / name / f"attempt-{len(previous) + 1:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        limit = min(minutes * 60, remaining)
        row = {"name": name, "status": "running", "signature": key, "directory": str(directory),
               "started_utc_epoch": time.time(), "limit_seconds": limit}
        self.state["attempts"].append(row)
        self.save()
        print(f"\nSTART {name}: phase cap {limit/60:.1f}m; active allowance remaining {remaining/60:.1f}m", flush=True)
        context = Stage(self, directory, limit)
        try:
            value, files = action(context)
            row.update(value=value, artifacts=artifacts(files), status="passed")
            print(f"PASS {name}", flush=True)
            return value
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            print(f"FAIL {name}: {row['error']}", flush=True)
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - context.started
            self.state["active_seconds"] += row["elapsed_seconds"]
            self.save()

    def __exit__(self, kind, error, traceback):
        try:
            self.state["status"] = "passed" if kind is None else "failed"
            if error:
                self.state["error"] = f"{type(error).__name__}: {error}"
            else:
                self.state.pop("error", None)
            self.save()
            summary = {"protocol": self.state["protocol"], "status": self.state["status"],
                       "active_minutes": self.state["active_seconds"] / 60,
                       "stages": [{k: r[k] for k in ("name", "status", "elapsed_seconds") if k in r}
                                  for r in self.state["attempts"]], "error": self.state.get("error"),
                       "boundary": "Validation workflow status, not a quality improvement or serving speedup."}
            write_json(self.root / "summary.json", summary)
            archive = self.root.parent / f"{self.root.name}-reports-{time.time_ns()}.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                for path in sorted(self.root.rglob("*")):
                    if path.is_file() and not path.is_symlink() and path.suffix in {".json", ".log", ".txt"}:
                        output.write(path, path.relative_to(self.root))
            print(f"\n{self.state['status'].upper()}: reports {self.root}\nDownload: {archive}\n"
                  "Disconnect/delete the GPU runtime to stop billing. No automatic billing shutdown was performed.", flush=True)
        finally:
            self.lock.close()
