import json
import zipfile

import pytest

from scripts import build_w4_research_bundle as bundle
from scripts.build_qwen35_w4_research_notebook import unpack_source_bundle
from scripts.build_rq3_runtime import root_identity


def test_working_source_bundle_extracts_and_build_identity_needs_no_git(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "rotquant/__pycache__").mkdir(parents=True)
    (root / "rotquant/__init__.py").write_text("__version__ = '0.1'\n")
    (root / "rotquant/__pycache__/cached.pyc").write_bytes(b"not-source")
    (root / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (root / ".secret").write_text("excluded")
    monkeypatch.setattr(bundle.subprocess, "check_output", lambda *a, **kw: "a" * 40 + "\n")
    output = tmp_path / "source.zip"
    result = bundle.build_bundle(output, root)
    assert result["files"] == 2 and result["includes_working_tree_changes"]
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"rotquant/__init__.py", "pyproject.toml", "SOURCE_BUNDLE.json"}
        manifest = json.loads(archive.read("SOURCE_BUNDLE.json"))
        assert manifest["revision"] == "a" * 40
    extracted, identity = unpack_source_bundle(output, tmp_path / "extracted")
    assert not (extracted / ".git").exists()
    assert root_identity(extracted)["source_bundle_sha256"] == identity
    (extracted / "rotquant/__init__.py").write_text("changed")
    with pytest.raises(ValueError, match="mismatch"):
        root_identity(extracted)


def test_bundle_never_overwrites_or_writes_inside_checkout(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        bundle.build_bundle(root / "bundle.zip", root)
    output = tmp_path / "existing.zip"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        bundle.build_bundle(output, root)
    assert output.read_bytes() == b"keep"
