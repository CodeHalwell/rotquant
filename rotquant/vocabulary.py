"""Bounded, tied-vocabulary PTQ and its explicitly dense quality prototype.

The initial contract is intentionally narrow: tied nn.Embedding/nn.Linear,
Gaussian W6/W8, g128 (or smaller test groups), FP16 scales and fixed FWHT.
No source parameter is mutated during quantization. The prototype context
restores both aliases even when evaluation raises; projected bytes are never
reported as measured compressed model storage.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .pack import PackedTensor
from .quantize import QuantConfig, QuantizedWeight, Quantizer
from .rotate import RandomizedHadamard
from .utils import write_result

VOCABULARY_PROTOCOL = "rotquant-tied-vocabulary-v1"


@dataclass(frozen=True)
class VocabularyConfig:
    bits: int = 8
    group_size: int = 128
    block: int = 128
    chunk_rows: int = 1024
    seed: int = 0

    def __post_init__(self):
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or self.bits not in (6, 8):
            raise ValueError("vocabulary bits must be 6 or 8")
        for name in ("group_size", "block", "chunk_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.block & (self.block - 1):
            raise ValueError("vocabulary rotation block must be a power of two")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("vocabulary seed must be a nonnegative integer")

    def quant_config(self):
        return QuantConfig(bits=self.bits, codebook="gaussian", scale="mse_search",
                           group_size=self.group_size, scale_bits=16, error_comp="none",
                           seed=self.seed, collect_scale_diagnostics=True)


def tied_modules(model):
    """Resolve and validate both uses; discovery alone does not imply support."""
    try:
        embedding = model.get_input_embeddings()
        head = model.get_output_embeddings()
    except AttributeError as exc:
        raise TypeError("model must expose input/output embedding accessors") from exc
    if not isinstance(embedding, nn.Embedding) or not isinstance(head, nn.Linear):
        raise TypeError("vocabulary prototype requires nn.Embedding and nn.Linear")
    if embedding.weight is not head.weight:
        raise ValueError("input embedding and output head must share one Parameter")
    if head.bias is not None or embedding.max_norm is not None:
        raise ValueError("biased output heads and max_norm embeddings are unsupported")
    if embedding.weight.ndim != 2 or not embedding.weight.is_contiguous():
        raise ValueError("vocabulary must be a contiguous matrix")
    if tuple(head.weight.shape) != (embedding.num_embeddings, embedding.embedding_dim):
        raise ValueError("tied vocabulary dimensions disagree")
    return embedding, head


def tensor_digest(tensor: torch.Tensor, chunk_rows: int = 1024) -> str:
    digest = hashlib.sha256()
    digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    for start in range(0, tensor.shape[0], chunk_rows):
        chunk = tensor[start:start + chunk_rows].detach().cpu().contiguous()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_to_cpu(qweight):
    qweight.packed.data = qweight.packed.data.cpu()
    qweight.scales = qweight.scales.cpu()
    return qweight


class PackedVocabulary(nn.Module):
    """One packed owner usable by both lookup and projection reference wrappers.

    Chunks have FP16 group scales and share a codebook and rotation. Projection
    is tiled and has no persistent dense head. This is not a fused fast kernel.
    """

    def __init__(self, chunks, config: VocabularyConfig, source_shape, source_dtype,
                 source_digest: str):
        super().__init__()
        self.chunks = chunks
        self.config = config
        self.vocab_size, self.hidden_size = map(int, source_shape)
        self.execution_dtype = source_dtype
        self.source_digest = source_digest
        self.rotation = RandomizedHadamard(self.hidden_size, block=config.block,
                                          seed=config.seed)
        if not chunks or self.hidden_size % config.group_size or self.hidden_size % config.block:
            raise ValueError("invalid packed vocabulary dimensions")
        if self.hidden_size * config.bits % 32:
            raise ValueError("packed vocabulary rows must end on word boundaries")
        shared_codebook = chunks[0].codebook
        for chunk in chunks:
            if (chunk.in_features != self.hidden_size or chunk.out_features < 1
                    or chunk.packed.bits != config.bits or chunk.group_size != config.group_size
                    or chunk.scales is None or chunk.scales.dtype != torch.float16
                    or tuple(chunk.scales.shape) != (
                        chunk.out_features, self.hidden_size // config.group_size)
                    or chunk.codebook.levels != 2 ** config.bits
                    or tuple(chunk.packed.shape) not in (
                        (chunk.out_features, self.hidden_size), (chunk.out_features * self.hidden_size,))
                    or chunk.packed.numel != chunk.out_features * self.hidden_size
                    or chunk.packed.data.dtype != torch.int32
                    or chunk.packed.data.numel() != math.ceil(chunk.packed.numel * config.bits / 32)
                    or not torch.isfinite(chunk.scales).all() or (chunk.scales < 0).any()
                    or not torch.equal(chunk.codebook.centroids, shared_codebook.centroids)
                    or chunk.residual_packed is not None or chunk.sketch is not None):
                raise ValueError("unsupported or malformed packed vocabulary chunk")
            chunk.codebook = shared_codebook
        if sum(q.out_features for q in chunks) != self.vocab_size:
            raise ValueError("packed vocabulary has incomplete rows")
        self.fingerprint = hashlib.sha256(json.dumps({
            "protocol": VOCABULARY_PROTOCOL, "config": asdict(config),
            "source_digest": source_digest,
            "centroids": tensor_digest(shared_codebook.centroids),
            "chunks": [{"codes": tensor_digest(q.packed.data, chunk_rows=4 * 1024 * 1024),
                        "scales": tensor_digest(q.scales)} for q in chunks],
        }, sort_keys=True).encode()).hexdigest()

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        probe = fn(torch.empty(0, device=self.chunks[0].packed.data.device,
                               dtype=self.execution_dtype))
        self.execution_dtype = probe.dtype
        for chunk in self.chunks:
            chunk.packed.data = chunk.packed.data.to(probe.device)
            chunk.scales = chunk.scales.to(probe.device)  # storage dtype is invariant
        return self

    def packed_bytes(self):
        payload = sum(q.packed.data.numel() * q.packed.data.element_size()
                      + q.scales.numel() * q.scales.element_size() for q in self.chunks)
        centroids = self.chunks[0].codebook.centroids
        return payload + centroids.numel() * centroids.element_size() + self.hidden_size

    def reconstructed_chunks(self, device=None):
        start = 0
        for chunk in self.chunks:
            dense = chunk.dequantize()
            if device is not None:
                dense = dense.to(device)
            yield start, self.rotation.inverse_activation(dense).to(self.execution_dtype)
            start += chunk.out_features

    def lookup(self, ids):
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("embedding ids must be int32 or int64")
        flat = ids.reshape(-1)
        if flat.numel() and (flat.min() < 0 or flat.max() >= self.vocab_size):
            raise IndexError("vocabulary index out of range")
        result = torch.empty((flat.numel(), self.hidden_size), device=ids.device,
                             dtype=self.execution_dtype)
        start = 0
        for chunk in self.chunks:
            positions = ((flat >= start) & (flat < start + chunk.out_features)).nonzero().flatten()
            # Bound lookup temporaries to the same row-chunk limit as projection.
            for offset in range(0, len(positions), self.config.chunk_rows):
                selected = positions[offset:offset + self.config.chunk_rows]
                rows = (flat[selected] - start).to(chunk.packed.data.device)
                cols = torch.arange(self.hidden_size, device=rows.device)
                bits = (rows[:, None] * self.hidden_size + cols) * self.config.bits
                word, shift = bits // 32, bits % 32
                words = chunk.packed.data
                low = (words[word].long() & 0xFFFFFFFF) >> shift
                spill = shift + self.config.bits > 32
                if spill.any():
                    low[spill] |= ((words[word[spill] + 1].long() & 0xFFFFFFFF)
                                   << (32 - shift[spill]))
                indices = low & ((1 << self.config.bits) - 1)
                centroids = chunk.codebook.centroids.to(rows.device)
                scales = chunk.scales[rows].float().repeat_interleave(
                    self.config.group_size, dim=1)
                rotated = centroids[indices] * scales
                result[selected] = self.rotation.inverse_activation(
                    rotated.to(ids.device)).to(result.dtype)
            start += chunk.out_features
        return result.reshape(*ids.shape, self.hidden_size)

    def project(self, x):
        rotated = self.rotation.rotate_activation(x)
        # Allocate output once, not all chunk logits plus a second concatenation.
        result = torch.empty((*x.shape[:-1], self.vocab_size), device=x.device, dtype=x.dtype)
        start = 0
        for chunk in self.chunks:
            dense = chunk.dequantize().to(device=x.device, dtype=x.dtype)
            result[..., start:start + chunk.out_features] = torch.nn.functional.linear(rotated, dense)
            start += chunk.out_features
        return result


class PackedEmbedding(nn.Module):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.num_embeddings = owner.vocab_size
        self.embedding_dim = owner.hidden_size

    def forward(self, input_ids):
        return self.owner.lookup(input_ids)


class PackedOutputHead(nn.Module):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.in_features = owner.hidden_size
        self.out_features = owner.vocab_size

    def forward(self, hidden):
        return self.owner.project(hidden)


def install_packed_vocabulary(model, owner: PackedVocabulary):
    """Replace both source aliases with one packed owner (explicit mutation).

    No weight attribute is exposed: resizing/retie APIs must fail instead of
    silently recreating a dense source tensor. Generation is tested separately.
    """
    from ._internal import get_parent

    embedding, head = tied_modules(model)
    if tuple(embedding.weight.shape) != (owner.vocab_size, owner.hidden_size):
        raise ValueError("packed vocabulary shape differs from the source model")
    names = {id(module): name for name, module in model.named_modules()}
    embedding_name, head_name = names[id(embedding)], names[id(head)]
    if not embedding_name or not head_name:
        raise ValueError("vocabulary aliases must be named children")
    parent, attribute = get_parent(model, embedding_name)
    setattr(parent, attribute, PackedEmbedding(owner))
    parent, attribute = get_parent(model, head_name)
    setattr(parent, attribute, PackedOutputHead(owner))
    return {"embedding": embedding_name, "head": head_name}


def quantize_vocabulary(weight, config: VocabularyConfig, *, cache_dir=None,
                        progress: Callable[[dict], None] | None = None) -> PackedVocabulary:
    """Chunk and checkpoint Gaussian quantization without mutating the source."""
    from safetensors.torch import load_file, save_file

    if weight.ndim != 2 or min(weight.shape) < 1 or not weight.is_floating_point():
        raise ValueError("vocabulary must be a nonempty floating-point matrix")
    if weight.shape[1] % config.group_size or weight.shape[1] % config.block:
        raise ValueError("vocabulary width must be divisible by group and rotation block")
    if weight.shape[1] * config.bits % 32:
        raise ValueError("vocabulary rows must end on packed word boundaries")
    source_hash = tensor_digest(weight, config.chunk_rows)
    identity = {"protocol": VOCABULARY_PROTOCOL, "config": asdict(config),
                "quantizer": asdict(config.quant_config()),
                "source_digest": source_hash, "shape": list(weight.shape)}
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir) / cache_key if cache_dir else None
    if root:
        root.mkdir(parents=True, exist_ok=True)
    quantizer = Quantizer(config.quant_config())
    rotation = RandomizedHadamard(weight.shape[1], block=config.block, seed=config.seed,
                                  device=weight.device)
    chunks = []
    started = time.monotonic()
    for start in range(0, weight.shape[0], config.chunk_rows):
        stop = min(start + config.chunk_rows, weight.shape[0])
        path = root / f"rows-{start:08d}-{stop:08d}.safetensors" if root else None
        record = path.with_suffix(".json") if path else None
        resumed = False
        if path and path.exists() and record.exists():
            metadata = json.loads(record.read_text())
            if metadata.get("identity") != identity or metadata.get("sha256") != file_digest(path):
                raise ValueError(f"corrupt or mismatched vocabulary chunk: {path}")
            data = load_file(str(path))
            shape = (stop - start, weight.shape[1])
            qw = QuantizedWeight(
                packed=PackedTensor(data["codes"], shape, config.bits, math.prod(shape)),
                scales=data["scales"], codebook=quantizer.codebook,
                group_size=config.group_size, out_features=shape[0], in_features=shape[1],
                scale_diagnostics=metadata.get("scale_diagnostics"))
            resumed = True
        else:
            source = weight[start:stop].detach().float()
            if not torch.isfinite(source).all():
                raise ValueError(f"nonfinite source vocabulary rows at {start}")
            qw = _chunk_to_cpu(quantizer.quantize_weight(rotation.rotate_weight(source)))
            if path:
                temporary = path.with_suffix(".safetensors.tmp")
                save_file({"codes": qw.packed.data, "scales": qw.scales}, str(temporary))
                os.replace(temporary, path)
                write_result(str(record), {"identity": identity, "sha256": file_digest(path),
                                           "scale_diagnostics": qw.scale_diagnostics})
        chunks.append(qw)
        elapsed = time.monotonic() - started
        if progress:
            progress({"phase": "vocabulary", "bits": config.bits, "rows": stop,
                      "total_rows": weight.shape[0], "resumed": resumed,
                      "elapsed_seconds": elapsed,
                      "eta_seconds": elapsed / stop * (weight.shape[0] - stop)})
    return PackedVocabulary(chunks, config, weight.shape, weight.dtype, source_hash)


@contextmanager
def vocabulary_prototype(model, owner: PackedVocabulary):
    """Install reconstructed values in one tied Parameter, restoring on failure."""
    embedding, head = tied_modules(model)
    weight = embedding.weight
    if tensor_digest(weight) != owner.source_digest:
        raise ValueError("vocabulary prototype requires the unchanged source matrix")
    source = weight.detach().to(device="cpu", copy=True)
    try:
        with torch.no_grad():
            for start, chunk in owner.reconstructed_chunks(weight.device):
                weight[start:start + len(chunk)].copy_(chunk)
        if embedding.weight is not head.weight:
            raise RuntimeError("vocabulary tie was broken")
        yield {
            "protocol": VOCABULARY_PROTOCOL, "mode": "dense_quality_prototype",
            "config": asdict(owner.config), "fingerprint": owner.fingerprint,
            "source_digest": owner.source_digest,
            "source_dense_bytes": weight.numel() * weight.element_size(),
            "packed_vocabulary_payload_bytes": owner.packed_bytes(),
            "scale_storage_diagnostics_by_chunk": [q.scale_diagnostics for q in owner.chunks],
            "artifact_bytes": None, "packed_artifact_verified": False,
            "source_verified_before_mutation": True, "tied_parameter": True,
        }
    finally:
        with torch.no_grad():
            for start in range(0, len(source), owner.config.chunk_rows):
                weight[start:start + owner.config.chunk_rows].copy_(
                    source[start:start + owner.config.chunk_rows])
        if tensor_digest(weight) != owner.source_digest:
            raise RuntimeError("source vocabulary restoration failed")
