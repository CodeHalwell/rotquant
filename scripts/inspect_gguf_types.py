#!/usr/bin/env python
"""Summarise the tensor types and byte shares of a GGUF file from its header.

Only the header is needed, so a multi-gigabyte artifact can be inspected from
its first few megabytes. Pass a local ``.gguf`` (complete or truncated to its
header) or ``--url`` to fetch a byte range over HTTP. The parser is
dependency-free so it can run anywhere the Python interpreter does.

Example (the pinned Unsloth Qwen3.5-4B comparator artifact):

    python scripts/inspect_gguf_types.py --url \\
      https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-UD-Q4_K_XL.gguf \\
      --head-bytes 33554432 --per-layer

Byte shares use the nominal bits-per-weight of each ggml type (block scales
included). They are exact for the K-quant, legacy and float types listed in
``BITS_PER_WEIGHT`` and are reported as zero for unknown types.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

GGUF_MAGIC = b"GGUF"

_SCALAR_FORMATS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
    10: "Q", 11: "q", 12: "d",
}
_STRING = 8
_ARRAY = 9

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}

# Nominal storage rate per weight including block scales/mins.
BITS_PER_WEIGHT = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0, "F64": 64.0,
    "Q4_0": 4.5, "Q4_1": 5.0, "Q5_0": 5.5, "Q5_1": 6.0, "Q8_0": 8.5,
    "Q2_K": 2.625, "Q3_K": 3.4375, "Q4_K": 4.5, "Q5_K": 5.5, "Q6_K": 6.5625,
    "IQ4_XS": 4.25, "IQ4_NL": 4.5, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625,
    "IQ2_S": 2.5, "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625, "IQ1_S": 1.5625,
    "IQ1_M": 1.75, "I8": 8.0, "I16": 16.0, "I32": 32.0, "I64": 64.0,
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    type_name: str
    numel: int

    @property
    def nominal_bytes(self) -> float:
        return self.numel * BITS_PER_WEIGHT.get(self.type_name, 0.0) / 8.0


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def scalar(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        if self.pos + size > len(self.data):
            raise EOFError("GGUF header is truncated; fetch more bytes")
        value = struct.unpack_from("<" + fmt, self.data, self.pos)[0]
        self.pos += size
        return value

    def string(self) -> str:
        length = self.scalar("Q")
        if self.pos + length > len(self.data):
            raise EOFError("GGUF header is truncated; fetch more bytes")
        raw = self.data[self.pos:self.pos + length]
        self.pos += length
        return raw.decode("utf-8", "replace")

    def value(self, value_type: int, *, keep: bool = True):
        if value_type in _SCALAR_FORMATS:
            return self.scalar(_SCALAR_FORMATS[value_type])
        if value_type == _STRING:
            return self.string()
        if value_type == _ARRAY:
            element_type = self.scalar("I")
            count = self.scalar("Q")
            if element_type in _SCALAR_FORMATS:
                size = struct.calcsize("<" + _SCALAR_FORMATS[element_type])
                if self.pos + size * count > len(self.data):
                    raise EOFError("GGUF header is truncated; fetch more bytes")
                self.pos += size * count
                return f"<array of {count} scalars>"
            preview = []
            for index in range(count):
                item = self.value(element_type, keep=index < 3)
                if index < 3:
                    preview.append(item)
            return f"<array of {count}: {preview}...>"
        raise ValueError(f"unknown GGUF value type {value_type}")


def parse_header(data: bytes) -> tuple[dict[str, object], list[TensorInfo], int]:
    """Return (metadata, tensor infos, header byte length) for a GGUF prefix."""
    if data[:4] != GGUF_MAGIC:
        raise ValueError("not a GGUF file (bad magic)")
    reader = _Reader(data)
    reader.pos = 4
    version = reader.scalar("I")
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    n_tensors = reader.scalar("Q")
    n_kv = reader.scalar("Q")
    metadata: dict[str, object] = {"gguf.version": version}
    for _ in range(n_kv):
        key = reader.string()
        value = reader.value(reader.scalar("I"))
        metadata[key] = value
    tensors: list[TensorInfo] = []
    for _ in range(n_tensors):
        name = reader.string()
        n_dims = reader.scalar("I")
        dims = tuple(reader.scalar("Q") for _ in range(n_dims))
        type_id = reader.scalar("I")
        reader.scalar("Q")  # data offset, unused here
        numel = 1
        for dim in dims:
            numel *= dim
        tensors.append(TensorInfo(name, dims, GGML_TYPES.get(type_id, f"type{type_id}"), numel))
    return metadata, tensors, reader.pos


def tensor_group(name: str) -> str:
    if name.startswith("token_embd"):
        return "token_embd"
    if name.startswith("output"):
        return "output"
    if "norm" in name:
        return "norm"
    if "ffn_" in name:
        return "ffn"
    if "attn_" in name:
        return "attn"
    if "ssm_" in name or "linear" in name:
        return "ssm"
    return "other"


BACKBONE_GROUPS = ("attn", "ffn", "ssm")


def summarise(tensors: list[TensorInfo]) -> dict[str, object]:
    by_group_type: dict[tuple[str, str], list[float]] = collections.defaultdict(lambda: [0, 0, 0.0])
    for tensor in tensors:
        row = by_group_type[(tensor_group(tensor.name), tensor.type_name)]
        row[0] += 1
        row[1] += tensor.numel
        row[2] += tensor.nominal_bytes
    rows = [
        {"group": group, "type": type_name, "tensors": count, "params": params, "bytes": nbytes}
        for (group, type_name), (count, params, nbytes) in sorted(by_group_type.items())
    ]
    # Matrices only: 1-D tensors (norms, biases, SSM decay vectors) are not weights.
    backbone = [t for t in tensors if tensor_group(t.name) in BACKBONE_GROUPS and len(t.dims) >= 2]
    backbone_params = sum(t.numel for t in backbone)
    backbone_bytes = sum(t.nominal_bytes for t in backbone)
    return {
        "rows": rows,
        "total_params": sum(t.numel for t in tensors),
        "total_nominal_bytes": sum(t.nominal_bytes for t in tensors),
        "backbone_params": backbone_params,
        "backbone_nominal_bytes": backbone_bytes,
        "backbone_bits_per_weight": (
            backbone_bytes * 8 / backbone_params if backbone_params else 0.0),
    }


def per_layer_table(tensors: list[TensorInfo]) -> tuple[list[str], dict[int, dict[str, str]]]:
    layers: dict[int, dict[str, str]] = collections.defaultdict(dict)
    columns: list[str] = []
    for tensor in tensors:
        match = re.match(r"blk\.(\d+)\.(\w+)\.weight$", tensor.name)
        if not match or len(tensor.dims) < 2:
            continue
        layer, projection = int(match.group(1)), match.group(2)
        layers[layer][projection] = tensor.type_name
        if projection not in columns:
            columns.append(projection)
    return columns, dict(sorted(layers.items()))


def fetch_head(url: str, head_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{head_bytes - 1}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read(head_bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("path", nargs="?", type=Path, help="local GGUF file or header prefix")
    source.add_argument("--url", help="HTTP(S) URL to range-download the header from")
    parser.add_argument("--head-bytes", type=int, default=32 * 1024 * 1024,
                        help="bytes to read; must cover the header (default 32 MiB)")
    parser.add_argument("--per-layer", action="store_true", help="print the per-layer type table")
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = parser.parse_args(argv)

    if args.url:
        data = fetch_head(args.url, args.head_bytes)
    else:
        with args.path.open("rb") as handle:
            data = handle.read(args.head_bytes)
    metadata, tensors, header_bytes = parse_header(data)
    summary = summarise(tensors)

    if args.json:
        payload = {
            "header_bytes": header_bytes,
            "metadata": {k: v for k, v in metadata.items() if not k.startswith("tokenizer.")},
            **summary,
        }
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    print(f"GGUF v{metadata['gguf.version']}: {len(tensors)} tensors, header {header_bytes:,} bytes")
    for key in ("general.architecture", "general.name", "general.file_type",
                "quantize.imatrix.file", "quantize.imatrix.dataset", "quantize.imatrix.chunks_count"):
        if key in metadata:
            print(f"  {key} = {metadata[key]}")
    print()
    print(f"{'group':<12}{'type':<9}{'tensors':>8}{'params':>16}{'nominal bytes':>16}")
    for row in summary["rows"]:
        print(f"{row['group']:<12}{row['type']:<9}{row['tensors']:>8}{row['params']:>16,}{row['bytes']:>16,.0f}")
    print()
    print(f"total params {summary['total_params']:,}; nominal tensor bytes "
          f"{summary['total_nominal_bytes']:,.0f}")
    print(f"backbone matrices ({', '.join(BACKBONE_GROUPS)}): {summary['backbone_params']:,} params, "
          f"{summary['backbone_nominal_bytes']:,.0f} bytes, "
          f"{summary['backbone_bits_per_weight']:.3f} bits/weight")

    if args.per_layer:
        columns, layers = per_layer_table(tensors)
        print()
        print("layer " + " ".join(f"{column:<12}" for column in columns))
        for layer, types in layers.items():
            print(f"{layer:>5} " + " ".join(f"{types.get(column, '-'):<12}" for column in columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
