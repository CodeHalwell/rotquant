"""Assignment-changing whole-model hard ternary QAT with independent behaviour gates."""
from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

from ._internal import get_parent
from .ternary import TernaryLinear, packed_ternary_state
from .ternary_global_audit import audit_dense_kl, prepare_dense_audit
from .ternary_global_config import (
    TernaryGlobalConfig,
    behaviour_acceptance,
    validate_behaviour,
)
from .ternary_global_data import (
    call_mask,
    evaluate_forward,
    evaluate_loss,
    model_loss,
    prepare_calls,
)
from .ternary_global_state import global_contract, protected_digest, restore_rng, rng_state
from .ternary_latents import retained_site_latents, save_selected_latents, ternary_sites
from .ternary_numerics import stable_qwen_fallbacks
from .ternary_state import (
    load_training_state,
    parameter_snapshot,
    restore_parameters,
    save_training_state,
)


def replace_sites(model, modules):
    for name, module in modules.items():
        owner, attribute = get_parent(model, name)
        setattr(owner, attribute, module)


def make_trainable(model, sites, latents, *, learn_assignments=True):
    model.requires_grad_(False)
    replacements = {name: TernaryLinear.from_quant_linear(module, latents.get(name))
                    for name, module in sites.items()}
    if not learn_assignments:
        for module in replacements.values():
            module.freeze_assignments()
    replace_sites(model, replacements)
    return replacements


def make_optimizer(model, config):
    latent, scales = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (scales if name.endswith("log_scales") else latent).append(parameter)
    if not scales or (config.learn_assignments and not latent):
        raise ValueError("global QAT requires the configured trainable latent/scales parameters")
    return torch.optim.AdamW([{"params": latent, "lr": config.lr},
                              {"params": scales, "lr": config.scale_lr}], weight_decay=0)


class GlobalTernaryTrainer:
    def __init__(self, model, train, validation, config, evaluator, path, contract, device, audits):
        self.model, self.train, self.validation = model, train, validation
        self.config, self.evaluator, self.path = config, evaluator, path
        self.contract, self.device = contract, device
        self.audits = audits
        self.optimizer = make_optimizer(model, config)
        self.step, self.stale, self.best_step, self.optimiser_seen_tokens = 0, 0, 0, 0
        self.initial_loss = evaluate_loss(model, validation, config, device)
        self.best_loss = self.initial_loss
        self.baseline_behaviour = self.behaviour()
        self.selected_behaviour = self.baseline_behaviour
        self.best = parameter_snapshot(model)
        self.history = []
        self.initial_dense_audit = audit_dense_kl(model, audits, config, device)

    def behaviour(self):
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return validate_behaviour(self.evaluator(self.model), self.config)
        finally:
            self.model.train(was_training)

    def save(self):
        fields = ("step", "stale", "best_step", "optimiser_seen_tokens", "initial_loss", "best_loss",
                  "baseline_behaviour", "selected_behaviour", "best", "history", "initial_dense_audit")
        state = {name: getattr(self, name) for name in fields}
        state.update(contract=self.contract, complete=False, parameters=parameter_snapshot(self.model),
                     optimizer=self.optimizer.state_dict(), rng=rng_state())
        save_training_state(self.path, state)

    def restore(self, state):
        restore_parameters(self.model, state["parameters"])
        self.optimizer.load_state_dict(state["optimizer"])
        fields = ("step", "stale", "best_step", "optimiser_seen_tokens", "initial_loss", "best_loss",
                  "baseline_behaviour", "selected_behaviour", "best", "history", "initial_dense_audit")
        for name in fields:
            setattr(self, name, state[name])
        current = parameter_snapshot(self.model)
        restore_parameters(self.model, self.best)
        restore_parameters(self.model, current)
        restore_rng(state["rng"])

    def update(self):
        self.optimizer.zero_grad(set_to_none=True)
        record = self.train[self.step % len(self.train)]
        self.model.train(bool(getattr(self.model, "is_gradient_checkpointing", False)))
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d,
                                   torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()
        with stable_qwen_fallbacks(self.model):
            loss = model_loss(self.model, record, self.config, self.device)
            if not torch.isfinite(loss):
                raise FloatingPointError("global ternary training loss is nonfinite")
            loss.backward()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm or math.inf)
        if not torch.isfinite(norm):
            raise FloatingPointError("global ternary training gradient is nonfinite")
        self.optimizer.step()
        self.step += 1
        self.optimiser_seen_tokens += int(call_mask(record[0])[:, 1:].sum())

    def validate(self):
        loss = evaluate_loss(self.model, self.validation, self.config, self.device)
        behaviour = self.behaviour()
        accepted, reasons = behaviour_acceptance(behaviour, self.baseline_behaviour, self.config)
        improved = loss < self.best_loss - self.config.min_improvement
        self.history.append({"step": self.step, "validation_loss": loss, "behaviour": behaviour,
                             "behaviour_accepted": accepted, "rejected_metrics": reasons})
        if self.step % self.config.dense_audit_interval == 0 or self.step == self.config.steps:
            self.history[-1]["dense_audit"] = audit_dense_kl(
                self.model, self.audits, self.config, self.device)
        if improved and accepted:
            self.best_loss, self.best_step, self.stale = loss, self.step, 0
            self.best, self.selected_behaviour = parameter_snapshot(self.model), behaviour
        else:
            self.stale += 1

    def finished(self):
        return self.step >= self.config.steps or bool(
            self.config.early_stopping_patience and self.stale >= self.config.early_stopping_patience)

    def run(self):
        while not self.finished():
            self.update()
            if (self.step % self.config.validation_interval == 0
                    or self.step % self.config.dense_audit_interval == 0
                    or self.step == self.config.steps):
                self.validate()
            if self.step % self.config.checkpoint_interval == 0 or self.finished():
                self.save()
        attempted = parameter_snapshot(self.model)
        restore_parameters(self.model, self.best)
        return {"steps_completed": self.step, "best_step": self.best_step,
                "accepted": self.best_step > 0,
                "selection": "candidate" if self.best_step else "restored_baseline",
                "initial_validation_loss": self.initial_loss,
                "selected_validation_loss": self.best_loss, "history": self.history,
                "baseline_behaviour": self.baseline_behaviour,
                "selected_behaviour": self.selected_behaviour,
                "initial_dense_audit": self.initial_dense_audit,
                "optimiser_seen_tokens": self.optimiser_seen_tokens,
                "attempted_validation": self.history[-1] if self.history else None,
                "rejection_reason": [] if self.best_step else ["no_improving_candidate_passed_behaviour"],
                "attempted_parameter_digest": parameter_digest(attempted),
                "selected_parameter_digest": parameter_digest(self.best)}


def parameter_digest(parameters):
    from .ternary_state import content_digest
    return content_digest(parameters)


def assignment_changes(current, initial):
    from .pack import unpack_indices
    changed, total = 0, 0
    for name, module in current.items():
        left = unpack_indices(module.to_quant_linear().qweight.packed)
        right = unpack_indices(initial[name].qweight.packed).to(left.device)
        changed, total = changed + int((left != right).sum()), total + left.numel()
    return {"changed": changed, "total": total, "rate": changed / total}


def commit_packed(model, trainable, validation, config, device):
    before, before_logits = evaluate_forward(model, validation, config, device, exact=True)
    packed = {name: module.to_quant_linear() for name, module in trainable.items()}
    replace_sites(model, packed)
    after, after_logits = evaluate_forward(model, validation, config, device, exact=True)
    if before != after or before_logits != after_logits:
        replace_sites(model, trainable)
        raise RuntimeError("global hard ternary and packed validation forwards differ")
    return packed, after, after_logits


def restore_complete(model, sites, state, validation, config, device):
    from .ternary_latents import load_selected_latents
    load_selected_latents(state["stats"]["selected_latents"], state["packed"], state["contract"])
    if sites.keys() != state["packed"].keys():
        raise ValueError("global QAT completed checkpoint sites differ")
    trainable = make_trainable(model, sites, {})
    from .ternary import restore_packed_ternary
    restored = {}
    for name, module in trainable.items():
        source = SimpleNamespace(in_features=module.in_features, out_features=module.out_features,
                                 bias=module.bias, weight=torch.empty(0, device=device))
        restored[name] = restore_packed_ternary(state["packed"][name], source)
    replace_sites(model, restored)
    loss, digests = evaluate_forward(model, validation, config, device, exact=True)
    if loss != state["stats"]["packed_validation_loss"] or digests != state["stats"]["packed_logits"]:
        raise ValueError("global QAT completed packed reload validation differs")
    return state["stats"]


def train_global_ternary(model, train_calls, validation_calls, config: TernaryGlobalConfig, *,
                         behavioural_evaluator, training_artifact_receipts=(), device=None,
                         stats_out=None, dense_audit_calls=()):
    """Mutate only ternary sites; return a packed model, reverting on any failed gate."""
    train_calls, validation_calls = list(train_calls), list(validation_calls)
    train, validation = prepare_calls(train_calls, validation_calls, config)
    sites = ternary_sites(model)
    if not sites:
        raise ValueError("global QAT found no packed ternary matrices")
    latents, identities = retained_site_latents(model, training_artifact_receipts)
    audits, audit_identity = prepare_dense_audit(train_calls, dense_audit_calls)
    contract = global_contract(model, train, validation, config, identities, audit_identity)
    protected = protected_digest(model, sites)
    path = None if config.checkpoint_dir is None else Path(config.checkpoint_dir) / "global.pt"
    state = load_training_state(path, contract, config.resume)
    device = device or next(model.parameters()).device
    initial_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    try:
        stats = run_global(model, sites, latents, train, validation, config,
                           behavioural_evaluator, path, contract, device, state, audits)
        if protected_digest(model, sites) != protected:
            raise RuntimeError("global ternary QAT changed a protected non-ternary tensor")
    except BaseException:
        replace_sites(model, sites)
        raise
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(initial_flags.get(name, False))
        model.eval()
    if stats_out is not None:
        stats_out["ternary_global"] = stats
    return model


def run_global(model, sites, latents, train, validation, config, evaluator, path, contract, device, state, audits):
    if state is not None and state["complete"]:
        return restore_complete(model, sites, state, validation, config, device)
    trainable = make_trainable(model, sites, latents, learn_assignments=config.learn_assignments)
    trainer = GlobalTernaryTrainer(model, train, validation, config, evaluator, path, contract, device, audits)
    if state is not None:
        trainer.restore(state)
    with checkpointed_training(model, config) as checkpointing:
        stats = trainer.run()
    model.eval()
    stats["gradient_checkpointing"] = checkpointing
    stats["assignments"] = assignment_changes(trainable, sites)
    stats["learn_assignments"] = config.learn_assignments
    stats["initialisation"] = {name: "retained_selected_latent" if name in latents
                               else "packed_reconstruction" for name in sites}
    stats["training_token_presentations"] = stats["optimiser_seen_tokens"]
    stats["distinct_training_target_positions"] = sum(int(call_mask(call)[:, 1:].sum())
                                                     for call, _ in train)
    stats["training_context_lengths"] = sorted({call.inputs["input_ids"].shape[1] for call, _ in train})
    packed_state = {name: packed_ternary_state(module.to_quant_linear())
                    for name, module in trainable.items()}
    if path is not None:
        stats["selected_latents"] = save_selected_latents(
            model, packed_state, contract, "", path.parent / "selected_latents" / "global.latents.pt")
    packed, stats["packed_validation_loss"], stats["packed_logits"] = commit_packed(
        model, trainable, validation, config, device)
    stats["deployed_dense_latents"] = False
    save_training_state(path, {"contract": contract, "complete": True,
                               "packed": {name: packed_ternary_state(module)
                                          for name, module in packed.items()}, "stats": stats})
    return stats


@contextmanager
def checkpointed_training(model, config):
    supported = callable(getattr(model, "gradient_checkpointing_enable", None))
    original = bool(getattr(model, "is_gradient_checkpointing", False))
    enabled = supported and config.gradient_checkpointing
    if enabled and not original:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        yield {"requested": config.gradient_checkpointing, "supported": supported,
               "enabled": enabled or original, "use_reentrant": False}
    finally:
        if enabled and not original:
            model.gradient_checkpointing_disable()
