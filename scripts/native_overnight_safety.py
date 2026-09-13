"""Notebook supervision and optional Colab release. No provisioning or keepalive."""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

from scripts.colab_runtime import run_live
from scripts.native_gpu_workflow import write_json
from scripts.native_hashing import digest


def verified_archive(root):
    root = Path(root)
    if not (root / "summary.json").is_file() or not (root / "workflow.json").is_file():
        raise ValueError("Final workflow/summary receipts missing; runtime will not be released")
    candidates = sorted(root.parent.glob(root.name + "-reports-*.zip"), key=lambda p: p.stat().st_mtime_ns)
    for path in reversed(candidates):
        if path.is_symlink():
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                if (archive.read("summary.json") != (root / "summary.json").read_bytes()
                        or archive.read("workflow.json") != (root / "workflow.json").read_bytes()
                        or archive.testzip() is not None):
                    continue
                # Every current report/log must be included, not just the two
                # summary files. Check CRC and exact bytes before release.
                for source in root.rglob("*"):
                    if source.is_file() and not source.is_symlink() and source.suffix in {".json", ".log", ".txt"}:
                        # Launch log is still open until the supervisor returns.
                        if "launch-logs" in source.relative_to(root).parts:
                            continue
                        if archive.read(str(source.relative_to(root))) != source.read_bytes():
                            raise ValueError("Archive omitted or changed an evidence file")
                return path
        except (zipfile.BadZipFile, KeyError, ValueError):
            continue
    raise ValueError("No complete verified archive; keep the runtime and recover reports manually")


def finish(root, auto_release=False, *, release=None):
    archive = verified_archive(root)
    receipt = {"archive": str(archive), "sha256": digest(archive), "verified": True,
               "auto_release_requested": bool(auto_release), "billing_stop_confirmed": False}
    # Outside root: do not invalidate the archive we just verified.
    path = Path(root).parent / (Path(root).name + "-release.json")
    write_json(path, receipt)
    print("Verified persistent reports:", archive, flush=True)
    if auto_release:
        if release is None:
            from google.colab import runtime
            release = runtime.unassign
        try:
            print("Requesting Colab runtime release. This API is best effort, not a billing guarantee.", flush=True)
            release()
            receipt["release_call_returned"] = True
        except Exception as error:
            receipt["release_error"] = f"{type(error).__name__}: {error}"
            print("RELEASE FAILED: disconnect/delete the runtime manually.", flush=True)
        # A successful unassign may terminate us before we can write this.
        write_json(path, receipt)
    else:
        print("AUTO_RELEASE_RUNTIME=False: disconnect/delete the GPU runtime manually.", flush=True)
    return receipt


def supervise(command, root, repo, minutes, *, auto_release=False):
    if not 10 <= minutes <= 480:
        raise ValueError("Expected 10..480 minute cap")
    root = Path(root)
    state_path = root / "workflow.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state and state.get("controls", {}).get("overnight_wall_minutes") != minutes:
        raise ValueError("Cannot renew a run's wall allowance; choose a new run name")
    remaining = (state["wall_deadline_epoch"] - time.time()) if state else minutes * 60
    try:
        if remaining <= 0:
            raise TimeoutError("Original wall deadline expired; no new worker launched")
        # Independent parent bound catches hangs even outside a driver stage.
        run_live(command, "overnight", repo_dir=repo, log_root=root / "launch-logs", timeout_seconds=remaining)
    finally:
        try:
            finish(root, auto_release=auto_release)
        except Exception as error:
            print(f"Reports/release check failed: {error}\nRecover files and disconnect/delete the GPU manually.", flush=True)
            # Never mask the original worker failure; no release without evidence.
