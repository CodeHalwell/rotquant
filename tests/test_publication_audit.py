from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_publication.py"
SPEC = importlib.util.spec_from_file_location("audit_publication", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_publication)


def _manifest() -> dict:
    return json.loads((ROOT / "paper" / "data" / "publication_results.json").read_text())


def test_publication_numbers_recompute() -> None:
    checks, derived = audit_publication.validate_numerical_claims(_manifest())
    assert checks
    assert all(check["passed"] for check in checks)
    assert derived["joint_mean_ppl"] == 14.5548
    assert 59.3 < derived["actual_tensor_storage_reduction_pct"] < 59.4


def test_tex_macros_are_deterministic() -> None:
    manifest = _manifest()
    _, derived = audit_publication.validate_numerical_claims(manifest)
    rendered = audit_publication.render_tex_macros(manifest, derived)
    assert "\\newcommand{\\QwenJointMeanPPL}{14.5548}" in rendered
    assert "\\newcommand{\\ActualTensorReductionPct}{59.34}" in rendered
    assert "[TODO" not in rendered


def test_local_checkpoint_auditors(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 4}, "weight_map": {}})
    )
    (source / "weights.bin").write_bytes(b"1234")
    source_bytes = sum(path.stat().st_size for path in source.iterdir())
    source_checks = audit_publication.audit_source_checkpoint(
        source, {"tensor_bytes": 4, "complete_snapshot_bytes": source_bytes}
    )
    assert all(check["passed"] for check in source_checks)

    packed = tmp_path / "packed"
    packed.mkdir()
    (packed / "model.safetensors").write_bytes(b"12")
    (packed / "packed.safetensors").write_bytes(b"345")
    (packed / "rotquant_config.json").write_text(
        json.dumps({"quantized_modules": [{"lora_rank": 0}]})
    )
    packed_bytes = sum(path.stat().st_size for path in packed.iterdir())
    packed_checks = audit_publication.audit_rotquant_checkpoint(
        packed,
        {
            "tensor_files": ["model.safetensors", "packed.safetensors"],
            "tensor_bytes": 5,
            "complete_snapshot_bytes": packed_bytes,
            "quantized_modules": 1,
        },
    )
    assert all(check["passed"] for check in packed_checks)


def _write_safetensors(path: Path, tensors: dict[str, int]) -> None:
    """Write a minimal safetensors file whose tensors have the given byte sizes."""
    import struct

    header: dict[str, dict] = {}
    offset = 0
    for name, size in tensors.items():
        header[name] = {
            "dtype": "F16",
            "shape": [size // 2],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)


def test_source_audit_derives_mtp_head_bytes_from_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_safetensors(source / "shard-1.safetensors", {"model.a": 4, "mtp.b": 6})
    _write_safetensors(source / "shard-2.safetensors", {"model.c": 8, "mtp.d": 2})
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 20},
        "weight_map": {
            "model.a": "shard-1.safetensors",
            "mtp.b": "shard-1.safetensors",
            "model.c": "shard-2.safetensors",
            "mtp.d": "shard-2.safetensors",
        },
    }))
    snapshot_bytes = sum(path.stat().st_size for path in source.iterdir())
    entry = {
        "tensor_bytes": 20,
        "complete_snapshot_bytes": snapshot_bytes,
        "mtp_head_bytes": 8,
        "tensor_bytes_excluding_mtp": 12,
    }
    checks = audit_publication.audit_source_checkpoint(source, entry)
    assert {check["check"] for check in checks} >= {
        "source header tensor bytes",
        "source MTP head bytes",
        "source loaded tensor bytes",
    }
    assert all(check["passed"] for check in checks)

    stale = dict(entry, mtp_head_bytes=7, tensor_bytes_excluding_mtp=13)
    failed = [
        check["check"]
        for check in audit_publication.audit_source_checkpoint(source, stale)
        if not check["passed"]
    ]
    assert failed == ["source MTP head bytes", "source loaded tensor bytes"]


def test_manifest_like_for_like_fields_are_internally_consistent() -> None:
    manifest = _manifest()
    checks, derived = audit_publication.validate_numerical_claims(manifest)
    labels = {check["check"] for check in checks}
    assert "source loaded bytes equal total minus MTP head" in labels
    assert "like-for-like tensor storage reduction" in labels
    assert 58.2 < derived["like_for_like_tensor_storage_reduction_pct"] < 58.3

    stale = json.loads(json.dumps(manifest))
    stale["models"]["source"]["tensor_bytes_excluding_mtp"] += 1
    stale_checks, _ = audit_publication.validate_numerical_claims(stale)
    assert any(
        not check["passed"] and check["check"] == "source loaded bytes equal total minus MTP head"
        for check in stale_checks
    )
