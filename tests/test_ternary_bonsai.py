"""Offline acquisition integrity, tokenizer identity and direct bridge contracts."""
import ctypes as C
import hashlib
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import ternary_research_bonsai as bonsai


def gguf_string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def gguf_value(value):
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, str):
        return 8, gguf_string(value)
    if isinstance(value, list):
        entries = [gguf_value(item) for item in value]
        return 9, struct.pack("<IQ", entries[0][0], len(entries)) + b"".join(v for _, v in entries)
    return 4, struct.pack("<I", value)


def write_gguf(path, changes=None):
    fields = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35",
              "tokenizer.ggml.tokens": ["a", "b", "end", "padding"],
              "tokenizer.ggml.token_type": [1, 1, 3, 5],
              "tokenizer.ggml.merges": ["a b"], "tokenizer.ggml.eos_token_id": 2,
              "tokenizer.chat_template": "reference template"}
    fields.update(changes or {})
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(fields))
    for key, value in fields.items():
        kind, encoded = gguf_value(value)
        data += gguf_string(key) + struct.pack("<I", kind) + encoded
    # Unknown custom tensor enum and payload are intentionally never parsed.
    path.write_bytes(data + b"unread custom tensor directory and model payload")
    return path


def test_tokenizer_gate_accepts_same_ids_with_different_vendor_chat_template(tmp_path):
    candidate = write_gguf(tmp_path / "bonsai.gguf", {"tokenizer.chat_template": "vendor template"})
    reference = write_gguf(tmp_path / "fp16.gguf")
    report = bonsai.validate_tokenizer(candidate, reference)
    assert report["passed"] and report["vocabulary_size"] == 4
    assert len(report["tokenizer_metadata_sha256"]) == 64


def test_tokenizer_gate_reports_only_unused_bos_and_padding_defaults(tmp_path):
    candidate = write_gguf(tmp_path / "bonsai.gguf", {
        "tokenizer.ggml.add_bos_token": False, "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.padding_token_id": 2})
    reference = write_gguf(tmp_path / "fp16.gguf", {
        "tokenizer.ggml.add_bos_token": False, "tokenizer.ggml.padding_token_id": 3})
    report = bonsai.validate_tokenizer(candidate, reference)
    assert set(report["unused_frozen_input_defaults"]) == {
        "tokenizer.ggml.bos_token_id", "tokenizer.ggml.padding_token_id"}
    write_gguf(candidate, {"tokenizer.ggml.add_bos_token": True,
                           "tokenizer.ggml.bos_token_id": 2})
    with pytest.raises(ValueError, match="KL/tasks forbidden"):
        bonsai.validate_tokenizer(candidate, reference)


@pytest.mark.parametrize("change", [
    {"tokenizer.ggml.tokens": ["b", "a", "end", "padding"]},
    {"tokenizer.ggml.eos_token_id": 1},
    {"tokenizer.ggml.merges": ["b a"]},
    {"tokenizer.ggml.token_type": [1, 1, 1, 5]},
])
def test_tokenizer_gate_rejects_semantic_or_special_id_mismatch(tmp_path, change):
    candidate = write_gguf(tmp_path / "bonsai.gguf", change)
    reference = write_gguf(tmp_path / "fp16.gguf")
    with pytest.raises(ValueError, match="KL/tasks forbidden"):
        bonsai.validate_tokenizer(candidate, reference)


def test_metadata_reader_rejects_truncated_or_oversized_fields(tmp_path):
    path = tmp_path / "broken.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 100_000_000))
    with pytest.raises(ValueError, match="bounded header"):
        bonsai.tokenizer_fields(path)
    path.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="truncated"):
        bonsai.tokenizer_fields(path)


@pytest.fixture
def tiny_acquisition(monkeypatch):
    bodies = {"PTQ1_0": b"ternary", "mmproj-BF16": b"vision"}
    monkeypatch.setattr(bonsai, "PINS", {
        key: (len(data), hashlib.sha256(data).hexdigest()) for key, data in bodies.items()})
    calls = []

    def download(repo, filename, revision, local_dir):
        from pathlib import Path
        assert repo == bonsai.REPOSITORY and revision == bonsai.REVISION
        kind = filename.removeprefix(bonsai.PREFIX).removesuffix(".gguf")
        path = Path(local_dir) / filename
        path.write_bytes(bodies[kind])
        calls.append(filename)
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    return calls


def test_acquisition_pins_hashes_counts_vision_and_resumes_without_overwriting(tmp_path, tiny_acquisition):
    report = bonsai.acquire(tmp_path)
    assert report["language_model_bytes"] == 7
    assert report["complete_model_payload_bytes"] == 13
    assert report["vision_execution"] is False
    assert len(tiny_acquisition) == 2
    assert bonsai.acquire(tmp_path) == report
    assert len(tiny_acquisition) == 2
    (tmp_path / f"{bonsai.PREFIX}PTQ1_0.gguf").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        bonsai.acquire(tmp_path)
    assert len(tiny_acquisition) == 2


def test_acquisition_rejects_symlink_even_to_matching_file(tmp_path, tiny_acquisition):
    other = tmp_path / "other"
    other.write_bytes(b"ternary")
    (tmp_path / f"{bonsai.PREFIX}PTQ1_0.gguf").symlink_to(other)
    with pytest.raises(ValueError, match="hash/size mismatch"):
        bonsai.acquire(tmp_path)
    assert not tiny_acquisition


def test_build_requires_pinned_pristine_source_and_cuda_toolkit(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(bonsai, "git", lambda source, *args: "changed")
    with pytest.raises(ValueError, match="pinned revision"):
        bonsai.prepare_source(source)
    monkeypatch.setattr(bonsai, "git", lambda source, *args:
                        bonsai.RUNTIME_REVISION if args[0] == "rev-parse" else " M file")
    with pytest.raises(ValueError, match="pristine"):
        bonsai.prepare_source(source)
    monkeypatch.setattr(bonsai.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="no CPU fallback"):
        bonsai.build_runtime(source, tmp_path / "build", backend="CUDA")


def test_build_receipt_binds_runtime_and_detects_library_changes(tmp_path, monkeypatch):
    source, build = tmp_path / "source", tmp_path / "build"
    source.mkdir()
    monkeypatch.setattr(bonsai, "prepare_source", lambda _: source)
    commands = []

    def run(command, **kwargs):
        commands.append(list(map(str, command)))
        if command[:2] == ["cmake", "--build"]:
            (build / "bin").mkdir()
            (build / "bin/libbonsai_measure.so").write_bytes(b"compiled bridge")

    monkeypatch.setattr(bonsai, "run", run)
    report = bonsai.build_runtime(source, build, backend="CPU", jobs=1)
    assert report["runtime_revision"] == bonsai.RUNTIME_REVISION
    assert report["load_validated"] and not report["gpu_validated"]
    assert "-DGGML_CUDA=OFF" in commands[0]
    assert "verify=False" in commands[-1][2]
    assert bonsai.verify_runtime(report["library"]) == report
    (build / "bin/libbonsai_measure.so").write_bytes(b"replaced bridge")
    with pytest.raises(ValueError, match="pinned build receipt"):
        bonsai.verify_runtime(report["library"])


class FakeNative:
    def __init__(self):
        self.calls, self.closed = [], False

    def bonsai_open(self, path, backend, context):
        return 17

    def bonsai_vocab(self, handle):
        return 4

    def bonsai_evaluate(self, handle, data, count, reset, selected, output):
        ids = np.ctypeslib.as_array(C.cast(data, C.POINTER(C.c_int32)), shape=(count,))
        values = np.ctypeslib.as_array(C.cast(output, C.POINTER(C.c_float)), shape=(selected, 4))
        self.calls.append((ids.tolist(), reset, selected))
        values[:] = np.eye(4, dtype=np.float32)[ids[-selected:]]
        return 0

    def bonsai_is_eog(self, handle, token):
        return token == 2

    def bonsai_close(self, handle):
        self.closed = True


def test_native_model_preserves_requested_positions_reset_and_closed_state(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"fixture")
    native = FakeNative()
    with bonsai.BonsaiModel(SimpleNamespace(lib=native), path, "CPU", 32) as model:
        output = model.evaluate([0, 1, 2], reset=True, selected=2)
        assert output.shape == (2, 4)
        assert output.argmax(-1).tolist() == [1, 2]
        assert native.calls[-1] == ([0, 1, 2], True, 2)
        assert model.is_eog(2) and not model.is_eog(1)
        assert bonsai.smoke_model(model, [0, 1, 2])["reset_exact"]
        for invalid in ([4], [-1], [1.5], []):
            with pytest.raises(ValueError, match="invalid Bonsai"):
                model.evaluate(invalid)
    assert native.closed
    with pytest.raises(ValueError, match="closed model"):
        model.evaluate([1])


def test_native_runtime_requires_expected_bridge_abi(tmp_path, monkeypatch):
    path = tmp_path / "lib.so"
    path.write_bytes(b"fixture")

    class Function:
        def __call__(self):
            return b"wrong-version"

    lib = SimpleNamespace(**{name: Function() for name in (
        "bonsai_bridge_version", "bonsai_error", "bonsai_init", "bonsai_open", "bonsai_vocab",
        "bonsai_is_eog", "bonsai_evaluate", "bonsai_close")})
    monkeypatch.setattr(bonsai.C, "CDLL", lambda _: lib)
    with pytest.raises(ValueError, match="bridge version"):
        bonsai.BonsaiRuntime(path, verify=False)
