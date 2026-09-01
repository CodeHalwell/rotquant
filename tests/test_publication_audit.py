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
