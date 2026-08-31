"""Calibration: forward-hook activation capture and incremental layer Hessians.

For each target ``nn.Linear`` we accumulate the (input) Hessian estimate
``H = mean_t x_t^T x_t``
incrementally over a calibration set -- we never store the activations themselves.
A damping term (default 1% of the mean diagonal) is added, auto-increasing on
Cholesky failure, which is the instability the spec warns about.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .utils import get_logger

logger = get_logger(__name__)


class HessianAccumulator:
    """Incrementally accumulates ``H = mean_t x_t^T x_t`` for one linear layer."""

    def __init__(self, in_features: int, device=None, dtype=torch.float32):
        self.in_features = in_features
        self.H = torch.zeros(in_features, in_features, device=device, dtype=dtype)
        self.mean = torch.zeros(in_features, device=device, dtype=dtype)
        self.n_samples = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        # x: (..., in_features) -> flatten leading dims to tokens
        x = x.reshape(-1, self.in_features).to(self.H.dtype)
        n = x.shape[0]
        if n == 0:
            return
        # Running mean of x^T x keeps the scale stable across batches.
        total = self.n_samples + n
        self.H *= self.n_samples / total
        self.H += (x.transpose(0, 1) @ x) / total
        self.mean *= self.n_samples / total
        self.mean += x.sum(dim=0) / total
        self.n_samples += n

    def finalize(self, damp_frac: float = 0.01) -> torch.Tensor:
        H = self.H.clone()
        mean_diag = torch.diag(H).mean().clamp_min(1e-8)
        H[range(self.in_features), range(self.in_features)] += damp_frac * mean_diag
        return H


@dataclass
class CalibrationResult:
    hessians: Mapping[str, torch.Tensor] = field(default_factory=dict)
    n_samples: dict[str, int] = field(default_factory=dict)
    means: dict[str, torch.Tensor] = field(default_factory=dict)


class DiskHessianStore(Mapping[str, torch.Tensor]):
    """Small in-memory index over Hessians individually offloaded to disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._paths: dict[str, Path] = {}

    def put(self, name: str, tensor: torch.Tensor) -> None:
        digest = hashlib.sha256(name.encode()).hexdigest()[:20]
        path = self.root / f"{digest}.pt"
        torch.save(tensor.detach().to(device="cpu", dtype=torch.float32), path)
        self._paths[name] = path

    def __getitem__(self, name: str) -> torch.Tensor:
        return torch.load(
            self._paths[name], map_location="cpu", weights_only=True
        )

    def __iter__(self):
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


def _forward_calibration_batch(model: nn.Module, batch, device) -> None:
    if isinstance(batch, dict):
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        model(**batch)
    elif isinstance(batch, (list, tuple)):
        model(*[
            value.to(device) if torch.is_tensor(value) else value
            for value in batch
        ])
    else:
        model(batch.to(device))


@dataclass
class ActivationResult:
    """Bounded source-model input samples for layerwise reconstruction."""

    activations: dict[str, torch.Tensor] = field(default_factory=dict)
    n_samples: dict[str, int] = field(default_factory=dict)
    seen_samples: dict[str, int] = field(default_factory=dict)


def _iter_linears(model: nn.Module, include: Sequence[str] | None = None,
                  exclude: Sequence[str] | None = None):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if include is not None and not any(k in name for k in include):
                continue
            if exclude and any(k in name for k in exclude):
                continue
            yield name, mod


@torch.no_grad()
def collect_hessians(model: nn.Module, dataloader: Iterable, device,
                     include: Sequence[str] | None = None,
                     exclude: Sequence[str] | None = None,
                     max_batches: int | None = None,
                     damp_frac: float = 0.01,
                     offload_device: str | None = "cpu") -> CalibrationResult:
    """Run the model over calibration batches, capturing per-linear input Hessians.

    ``dataloader`` yields tensors / dicts suitable for ``model(**batch)`` or
    ``model(batch)``. Use 128-512 sequences of the model's context length.

    ``include``/``exclude`` are substring filters over layer names; pass the same
    values as :class:`~rotquant.patch.PatchConfig` so Hessians are only collected
    for layers that will actually be quantised.

    Finalised Hessians are moved to ``offload_device`` (default CPU) so they do
    not pile up in VRAM before patching -- for a 7B model, keeping every fp32
    ``[in, in]`` Hessian on-GPU costs ~25 GB on top of the model. The patcher
    moves each one back next to its weight when it is consumed. Note the
    *accumulators* still live on the GPU during the calibration forward passes;
    restrict ``include`` if that exceeds your VRAM.
    """
    accums: dict[str, HessianAccumulator] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str, in_features: int):
        def hook(_module, inputs, _output):
            x = inputs[0]
            if name not in accums:
                accums[name] = HessianAccumulator(in_features, device=x.device)
            accums[name].update(x)
        return hook

    include_terms = tuple(include) if include is not None else None
    exclude_terms = tuple(exclude) if exclude is not None else None
    for name, mod in _iter_linears(model, include_terms, exclude_terms):
        handles.append(mod.register_forward_hook(make_hook(name, mod.in_features)))

    model.eval()
    try:
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            _forward_calibration_batch(model, batch, device)
    finally:
        for h in handles:
            h.remove()

    result = CalibrationResult()
    for name, acc in accums.items():
        H = acc.finalize(damp_frac=damp_frac)
        result.hessians[name] = H.to(offload_device) if offload_device else H
        result.n_samples[name] = acc.n_samples
        result.means[name] = (
            acc.mean.to(offload_device) if offload_device else acc.mean.clone()
        )
    logger.info("Collected Hessians for %d linear layers", len(result.hessians))
    return result


@torch.no_grad()
def collect_hessians_streamed(
    model: nn.Module,
    dataloader: Iterable,
    device,
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    max_batches: int | None = None,
    damp_frac: float = 0.01,
    layers_per_pass: int = 1,
    offload_device: str | None = "cpu",
    offload_dir: str | Path | None = None,
) -> CalibrationResult:
    """Replay calibration with only ``layers_per_pass`` Hessians resident.

    GPU memory is bounded by the active layer group.  When ``offload_dir`` is
    provided, each finalized Hessian is written independently and loaded lazily
    by the patcher, also bounding host memory.
    """

    if layers_per_pass < 1:
        raise ValueError("layers_per_pass must be >= 1")
    if offload_dir is not None and offload_device not in (None, "cpu"):
        raise ValueError("disk-offloaded Hessians must be finalized on CPU")
    # Replaying requires a finite/reiterable sequence; materialize generators.
    batches = dataloader if isinstance(dataloader, Sequence) else list(dataloader)
    targets = list(_iter_linears(model, include, exclude))
    store: Mapping[str, torch.Tensor]
    disk_store = DiskHessianStore(offload_dir) if offload_dir is not None else None
    collected: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    means: dict[str, torch.Tensor] = {}
    model.eval()
    for start in range(0, len(targets), layers_per_pass):
        group = targets[start:start + layers_per_pass]
        accums: dict[str, HessianAccumulator] = {}
        handles = []

        def make_hook(name: str, in_features: int, *, accumulators=accums):
            def hook(_module, inputs, _output):
                x = inputs[0]
                if name not in accumulators:
                    accumulators[name] = HessianAccumulator(
                        in_features, device=x.device
                    )
                accumulators[name].update(x)
            return hook

        for name, module in group:
            handles.append(module.register_forward_hook(
                make_hook(name, module.in_features)
            ))
        try:
            for index, batch in enumerate(batches):
                if max_batches is not None and index >= max_batches:
                    break
                _forward_calibration_batch(model, batch, device)
        finally:
            for handle in handles:
                handle.remove()
        for name, _module in group:
            accumulator = accums.get(name)
            if accumulator is None:
                raise RuntimeError(f"linear layer {name} was not invoked during calibration")
            hessian = accumulator.finalize(damp_frac=damp_frac)
            hessian = hessian.to(offload_device) if offload_device else hessian
            if disk_store is not None:
                disk_store.put(name, hessian)
            else:
                collected[name] = hessian
            counts[name] = accumulator.n_samples
            means[name] = accumulator.mean.detach().to("cpu")
        logger.info(
            "Streamed Hessians for %d/%d linear layers",
            min(start + layers_per_pass, len(targets)),
            len(targets),
        )
    store = disk_store if disk_store is not None else collected
    return CalibrationResult(hessians=store, n_samples=counts, means=means)


@torch.no_grad()
def collect_activation_means(
    model: nn.Module,
    dataloader: Iterable,
    device,
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    max_batches: int | None = None,
    offload_device: str | None = "cpu",
) -> dict[str, torch.Tensor]:
    """Collect only per-linear input means, without allocating Hessians."""

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str, in_features: int):
        def hook(_module, inputs, _output):
            values = inputs[0].detach().reshape(-1, in_features).float()
            if values.numel() == 0:
                return
            if name not in sums:
                sums[name] = torch.zeros(
                    in_features, device=values.device, dtype=torch.float64
                )
                counts[name] = 0
            sums[name] += values.to(torch.float64).sum(dim=0)
            counts[name] += values.shape[0]
        return hook

    for name, module in _iter_linears(model, include, exclude):
        handles.append(module.register_forward_hook(
            make_hook(name, module.in_features)
        ))
    model.eval()
    try:
        for index, batch in enumerate(dataloader):
            if max_batches is not None and index >= max_batches:
                break
            _forward_calibration_batch(model, batch, device)
    finally:
        for handle in handles:
            handle.remove()
    means = {
        name: (total / counts[name]).to(dtype=torch.float32)
        for name, total in sums.items()
    }
    if offload_device is not None:
        means = {name: value.to(offload_device) for name, value in means.items()}
    return means


@torch.no_grad()
def collect_activations(model: nn.Module, dataloader: Iterable, device,
                        include: Sequence[str] | None = None,
                        exclude: Sequence[str] | None = None,
                        max_tokens: int = 64,
                        offload_device: str | None = "cpu",
                        storage_dtype: torch.dtype = torch.float16,
                        seed: int = 0) -> ActivationResult:
    """Reservoir-sample at most ``max_tokens`` inputs per targeted linear.

    Unlike a full ``[in, in]`` Hessian, this bounded sample costs O(tokens * d)
    storage and permits the rotation trainer to optimise the actual layer-output
    reconstruction loss. Samples are offloaded immediately (CPU fp16 by default)
    and promoted to fp32 next to a layer only while that layer is trained.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    reservoirs: dict[str, torch.Tensor] = {}
    priorities: dict[str, torch.Tensor] = {}
    seen: dict[str, int] = {}
    generators: dict[str, torch.Generator] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str, in_features: int):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().reshape(-1, in_features)
            if x.numel() == 0:
                return
            if offload_device is not None:
                x = x.to(device=offload_device, dtype=storage_dtype)
            else:
                x = x.to(dtype=storage_dtype)
            generator = generators.get(name)
            if generator is None:
                name_seed = int.from_bytes(
                    hashlib.sha256(name.encode()).digest()[:8], "little"
                )
                generator = torch.Generator(device="cpu").manual_seed(
                    (seed + name_seed) % (2**63 - 1)
                )
                generators[name] = generator
            new_priorities = torch.rand(
                x.shape[0], generator=generator, dtype=torch.float64
            )
            previous = reservoirs.get(name)
            if previous is None:
                combined = x
                combined_priorities = new_priorities
            else:
                combined = torch.cat((previous, x), dim=0)
                combined_priorities = torch.cat(
                    (priorities[name], new_priorities), dim=0
                )
            if combined.shape[0] > max_tokens:
                keep = torch.topk(
                    combined_priorities, max_tokens, sorted=False
                ).indices
                reservoirs[name] = combined[keep.to(combined.device)]
                priorities[name] = combined_priorities[keep]
            else:
                reservoirs[name] = combined
                priorities[name] = combined_priorities
            seen[name] = seen.get(name, 0) + x.shape[0]
        return hook

    include_terms = tuple(include) if include is not None else None
    exclude_terms = tuple(exclude) if exclude is not None else None
    targets = list(_iter_linears(model, include_terms, exclude_terms))
    for name, mod in targets:
        handles.append(mod.register_forward_hook(make_hook(name, mod.in_features)))

    model.eval()
    try:
        for batch in dataloader:
            if isinstance(batch, dict):
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                model(**batch)
            elif isinstance(batch, (list, tuple)):
                model(*[b.to(device) if torch.is_tensor(b) else b for b in batch])
            else:
                model(batch.to(device))
    finally:
        for handle in handles:
            handle.remove()

    result = ActivationResult(
        activations=reservoirs,
        n_samples={name: tensor.shape[0] for name, tensor in reservoirs.items()},
        seen_samples=seen,
    )
    logger.info(
        "Reservoir-sampled up to %d activation tokens for %d linear layers",
        max_tokens,
        len(result.activations),
    )
    return result
