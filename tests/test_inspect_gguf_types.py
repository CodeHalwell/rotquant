"""The GGUF header inspector must reproduce tensor types and byte shares."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "inspect_gguf_types.py"
    spec = importlib.util.spec_from_file_location("inspect_gguf_types", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    body = struct.pack("<I", 8) + struct.pack("<Q", len(values)) + b"".join(_string(v) for v in values)
    return _string(key) + struct.pack("<I", 9) + body


def _tensor(name: str, dims: tuple[int, ...], type_id: int) -> bytes:
    body = _string(name) + struct.pack("<I", len(dims))
    body += b"".join(struct.pack("<Q", d) for d in dims)
    return body + struct.pack("<I", type_id) + struct.pack("<Q", 0)


def _synthetic_gguf() -> bytes:
    kv = [
        _kv_string("general.architecture", "qwen35"),
        _kv_u32("general.file_type", 15),
        # Large string arrays (tokenizer vocab) must be skipped without being kept.
        _kv_string_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(50)]),
    ]
    tensors = [
        _tensor("token_embd.weight", (64, 1000), 14),       # Q6_K
        _tensor("output_norm.weight", (64,), 0),            # F32, 1-D, not backbone
        _tensor("blk.0.attn_qkv.weight", (64, 128), 13),    # Q5_K
        _tensor("blk.0.ffn_down.weight", (128, 64), 14),    # Q6_K
        _tensor("blk.0.ssm_out.weight", (64, 64), 8),       # Q8_0
        _tensor("blk.0.ssm_a", (32,), 0),                   # F32 vector, excluded
        _tensor("blk.1.attn_q.weight", (64, 128), 12),      # Q4_K
    ]
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    return header + b"".join(kv) + b"".join(tensors) + b"\0" * 16


def test_parse_and_summarise_synthetic_header():
    module = _load()
    data = _synthetic_gguf()
    metadata, tensors, header_bytes = module.parse_header(data)
    assert metadata["general.architecture"] == "qwen35"
    assert metadata["general.file_type"] == 15
    # Tokenizer arrays are stepped over, not decoded, and never reported.
    assert metadata["tokenizer.ggml.tokens"] == module.SKIPPED_VALUE
    assert header_bytes == len(data) - 16
    assert [t.type_name for t in tensors] == ["Q6_K", "F32", "Q5_K", "Q6_K", "Q8_0", "F32", "Q4_K"]

    summary = module.summarise(tensors)
    backbone_params = 64 * 128 + 128 * 64 + 64 * 64 + 64 * 128
    assert summary["backbone_params"] == backbone_params
    expected_bytes = (64 * 128 * 5.5 + 128 * 64 * 6.5625 + 64 * 64 * 8.5 + 64 * 128 * 4.5) / 8
    assert summary["backbone_nominal_bytes"] == pytest.approx(expected_bytes)
    assert summary["backbone_bits_per_weight"] == pytest.approx(expected_bytes * 8 / backbone_params)
    embedding_rows = [row for row in summary["rows"] if row["group"] == "token_embd"]
    assert embedding_rows == [
        {"group": "token_embd", "type": "Q6_K", "tensors": 1, "params": 64000,
         "bytes": pytest.approx(64000 * 6.5625 / 8)}
    ]

    columns, layers = module.per_layer_table(tensors)
    assert columns == ["attn_qkv", "ffn_down", "ssm_out", "attn_q"]
    assert layers[0] == {"attn_qkv": "Q5_K", "ffn_down": "Q6_K", "ssm_out": "Q8_0"}
    assert layers[1] == {"attn_q": "Q4_K"}


def test_truncated_header_fails_closed():
    module = _load()
    data = _synthetic_gguf()
    with pytest.raises(EOFError):
        module.parse_header(data[: len(data) // 2])


def test_skipped_values_are_not_decoded(monkeypatch):
    module = _load()
    data = _synthetic_gguf()
    decoded: list[str] = []
    original = module._Reader.string

    def spy(self):
        value = original(self)
        decoded.append(value)
        return value

    monkeypatch.setattr(module._Reader, "string", spy)
    metadata, _, _ = module.parse_header(data)
    assert metadata["general.architecture"] == "qwen35"
    # Keys and kept string values are decoded; no tokenizer token string is.
    tokens = {f"tok{i}" for i in range(50)}
    assert tokens.isdisjoint(decoded)
    assert "tokenizer.ggml.tokens" in decoded
    assert "qwen35" in decoded


def test_head_bytes_must_be_positive(tmp_path):
    module = _load()
    with pytest.raises(ValueError, match="positive"):
        module.fetch_head("https://example.invalid/model.gguf", 0)
    path = tmp_path / "tiny.gguf"
    path.write_bytes(_synthetic_gguf())
    with pytest.raises(SystemExit):
        module.main([str(path), "--head-bytes", "0"])
    with pytest.raises(SystemExit):
        module.main(["--url", "https://example.invalid/model.gguf", "--head-bytes", "-5"])


def test_cli_prints_summary(tmp_path, capsys):
    module = _load()
    path = tmp_path / "tiny.gguf"
    path.write_bytes(_synthetic_gguf())
    assert module.main([str(path), "--per-layer"]) == 0
    out = capsys.readouterr().out
    assert "token_embd  Q6_K" in out
    assert "bits/weight" in out
    assert "attn_qkv" in out and "Q8_0" in out
